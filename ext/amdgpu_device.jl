# Device boundary for KADeviceBackend: the storage guard, synchronization,
# failure mapping and the explicit host-to-device transfer.

_ka_sync!(b::KADeviceBackend) = KA.synchronize(b.ka)

Gesso.Inference._profile_sync(b::KADeviceBackend) = _ka_sync!(b)

function Gesso.Inference._engine_boundary!(b::KADeviceBackend)
    _ka_sync!(b)
    return nothing
end

function Gesso.Inference._engine_failure(err::AMDGPU.HIP.HIPError)
    oom = err.code in (AMDGPU.HIP.hipErrorOutOfMemory, AMDGPU.HIP.hipErrorMemoryAllocation)
    return gesso_error(
        oom ? ERR_ALLOCATION : ERR_RUNTIME,
        "primary AMDGPU operation failed";
        cause=sprint(showerror, err),
        cause_type=string(typeof(err)),
        interrupted=false,
        backend=:amdgpu,
    )
end

# --- device guard (the interpreter does not copy) -----------------------------

_ka_on_device(b::KADeviceBackend{B}, s) where {B} =
    s isa AbstractArray && KA.get_backend(s) isa B

# Device views are legal (prefill slices device buffers); host storage handed to
# a device backend is ERR_INVALID_PLAN, because the caller skipped to_device.
# On KADeviceBackend(KA.CPU()) host Arrays ARE the device, so this guard cannot
# catch a skipped transfer there; the AMDGPU backend is what it protects.
function _ka_device_storage!(op::Symbol, b::KADeviceBackend, t)
    s = t.storage
    root = Gesso.Inference._storage_root(s)
    _ka_on_device(b, root) && eltype(s) === Float32 && return s
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$op: $(backend_name(b)) backend requires Float32 storage on its own device; " *
            "host Array, host views, other devices and other element types are rejected; " *
            "call to_device explicitly";
            op=op,
            backend=backend_name(b),
            storage_type=string(typeof(s)),
        ),
    )
end

# Routed expert banks stay host-resident GgufQuantSlice storage and run on the
# host MoE route, the same split as CUDA and Lava. Only expert banks are exempt.
function Gesso.Inference._check_device_storage(op::Symbol, b::KADeviceBackend, t)
    t isa Gesso.ExpertWeight &&
        t.storage isa Gesso.Inference.GgufQuantSlice &&
        return nothing
    t.storage isa AmdDenseQuant && _ka_on_device(b, t.storage.data) && return nothing
    _ka_device_storage!(op, b, t)
    return nothing
end

# --- to_device ----------------------------------------------------------------

function _ka_upload(b::KADeviceBackend, s)
    root = Gesso.Inference._storage_root(s)
    !(root isa Array) && _ka_on_device(b, root) && eltype(s) === Float32 && return s
    host =
        s isa Gesso.Inference.QuantMatrix ? Gesso.Inference._quant_host_f32(s) :
        s isa AbstractArray ? Array{Float32}(s) :
        throw(
            gesso_error(
                ERR_INVALID_PLAN,
                "to_device: $(backend_name(b)) cannot transfer this storage";
                backend=backend_name(b),
                storage_type=string(typeof(s)),
            ),
        )
    dst = KA.allocate(b.ka, Float32, size(host))
    copyto!(dst, host)
    return dst
end

# ArchitectureSpec, RoPEPolicy, mixer tags — anything without .storage.
const _TO_DEVICE_STORAGE_KEYS = (:embedding, :blocks, :lm_head, :final_rms)

function _keep_tensor_meta(tensors)
    ks = Symbol[]
    vs = Any[]
    for k in keys(tensors)
        k in _TO_DEVICE_STORAGE_KEYS && continue
        push!(ks, k)
        push!(vs, getproperty(tensors, k))
    end
    isempty(ks) && return (;)
    return NamedTuple{Tuple(ks)}(Tuple(vs))
end

"""
    to_device(::KADeviceBackend, tensors) -> tensors'

Copy weight storage to Float32 device arrays and return a new named tuple of
the same shape. A projection whose GGUF format has a tested device kernel
(see `_AMDQ_TYPES`) keeps its block bytes on device; every other quant matrix
dequantizes to Float32 on the host first. Routed expert banks stay on the
host. NEVER mutates the CPU tensors. Tied heads stay tied: `lm_head === embedding`.
"""
function _ka_to_device(b::KADeviceBackend, tensors)
    _conv_t(t) = begin
        hasproperty(t, :storage) || return t
        t.storage === nothing && return t
        if t isa Gesso.ProjectionWeight && _amdq_eligible(t.storage)
            return typeof(t)(; shape=t.shape, storage=_amdq_upload(b, t.storage))
        end
        typeof(t)(; shape=t.shape, storage=_ka_upload(b, t.storage))
    end
    _host_expert_key(k) = k in (:wexp_gate, :wexp_up, :wexp_gate_up, :wexp_down)
    _host_expert_t(t) =
        hasproperty(t, :storage) && t.storage isa Gesso.Inference.GgufQuantSlice
    _conv_blk(blk) = NamedTuple{keys(blk)}(
        map(keys(blk)) do k
            t = getproperty(blk, k)
            _host_expert_key(k) && _host_expert_t(t) ? t : _conv_t(t)
        end,
    )
    embedding = _conv_t(tensors.embedding)
    blocks = [_conv_blk(blk) for blk in tensors.blocks]
    lm_head = tensors.lm_head === tensors.embedding ? embedding : _conv_t(tensors.lm_head)
    fr = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    final_rms = fr === nothing ? nothing : _conv_t(fr)
    _ka_sync!(b)
    result = (; embedding, blocks, lm_head, final_rms)
    return merge(result, _keep_tensor_meta(tensors))
end

# ext-local dispatch wrapper (bound as Gesso.to_device when this extension is
# the first backend extension to load; otherwise __init__ attaches the worker)
to_device(b::KADeviceBackend, tensors) = _ka_to_device(b, tensors)
