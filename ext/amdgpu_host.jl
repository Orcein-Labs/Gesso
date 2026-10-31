# Host views of device-resident weights for the ops that run on the CPU oracle.
#
# The MoE router and the expert bank stay on the CPU path (expert banks are
# GGUF slices that never leave the host; the router is a small dense matvec
# whose CPU kernel is the reference). Those ops receive host copies of the
# device weights. A quantized projection is copied back as raw GGUF bytes, so
# the CPU kernel decodes exactly the bytes the device GEMV reads. The copy is
# cached per weight so a decode loop pays the transfer once.

const _KA_HOST_QUANT = Dict{UInt,Tuple{WeakRef,Gesso.Inference.QuantMatrix}}()
const _KA_HOST_QUANT_LOCK = ReentrantLock()

function _ka_host_quant(q::AmdDenseQuant)
    key = objectid(q.data)
    lock(_KA_HOST_QUANT_LOCK) do
        e = get(_KA_HOST_QUANT, key, nothing)
        if e !== nothing && e[1].value === q.data
            return e[2]
        end
        filter!(p -> p.second[1].value !== nothing, _KA_HOST_QUANT)
        blck = Int64(Gesso.Inference._GGML_KNOWN_BLOCKS[q.ggml_type][1])
        host = Gesso.Inference.QuantMatrix(
            Array(q.data), 0, q.ggml_type, q.ne0, q.nrows, q.row_bytes, blck,
        )
        _KA_HOST_QUANT[key] = (WeakRef(q.data), host)
        return host
    end
end

_ka_host_storage(s) = s
_ka_host_storage(::Nothing) = nothing
_ka_host_storage(s::AbstractArray) = s isa Array ? s : Array(s)
_ka_host_storage(q::AmdDenseQuant) = _ka_host_quant(q)

_ka_host_wrap(w) = w
_ka_host_wrap(::Nothing) = nothing
_ka_host_wrap(w::ProjectionWeight) = ProjectionWeight(w.shape, _ka_host_storage(w.storage))
_ka_host_wrap(w::FrozenParameter) = FrozenParameter(w.shape, _ka_host_storage(w.storage))
_ka_host_wrap(w::ExpertWeight) = ExpertWeight(w.shape, _ka_host_storage(w.storage))

# Replace the named fields of a block's parameter tuple with host views;
# fields the block does not carry are skipped.
function _ka_host_bt(bt, fields)
    ks = Tuple(f for f in fields if haskey(bt, f))
    vals = Tuple(_ka_host_wrap(getproperty(bt, f)) for f in ks)
    return merge(bt, NamedTuple{ks}(vals))
end

# Activation storage, whether the caller passed the Activation or its array.
_ka_act_storage(x::Activation) = x.storage
_ka_act_storage(x) = x
