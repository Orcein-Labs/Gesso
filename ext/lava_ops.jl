# Lava operator methods + device transfer (§LXXXI item B).
#
# Same operators, more-specific methods (§CIX): these extend the EXISTING
# op! functions from src/backends.jl, dispatching on Gesso's LavaBackend ×
# semantic families × workload — exactly the CPU reference's and the CUDA
# extension's shape, on Lava.LavaArray{Float32} storage (§LXXXI math law:
# F32 on device; the CPU oracle stays F64 and parity is the declared
# atol=1e-3 for the micro models, not bit-identity).
#
# Implementation is GPUArrays broadcasting + Lava's `mul!` (Lava
# array/gemm.jl) ONLY — no KernelAbstractions kernels were needed: RoPE is
# strided views over the pairwise feature planes with a small angle table,
# causal softmax is an index-comparison mask plus dims-mapreduce max/sum.
# Fewer moving parts than the CUDA path (which needed two raw kernels) and
# no handwritten SPIR-V, no coopmat, no graphics (§LXXXI "not this sprint").
#
# Prefill still waits after every public op (host-visible logits). Decode
# submits without a per-op wait; `_engine_boundary!` waits once. Logits
# finite-check stays in `_greedy_id`. Host D2H (argmax, collect) still
# implicit-syncs inside Lava download.

using LinearAlgebra: mul!
using Gesso: ERR_NUMERICAL_INSTABILITY

const KA = Lava.KernelAbstractions
const KALava = Lava.LavaBackend   # the KA backend instance type (§LXXXI alias)
using Lava.KernelAbstractions: @kernel, @index

# Lava bakes SPIR-V LocalSize from workgroupsize. KA's default is
# `min(prod(ndrange), 64)`, so decode kernels whose ndrange grows with K
# (batched scores (H,K), GQA repeat (K, n_kv)) compiled a new pipeline
# every token until 9K ≥ 64 (spiral 3 Q1: 50 MB host alloc + 26 SPIR-V
# jobs on the first decode after prefill). Pin 64 threads. Kernels
# bound-check extra lanes.
const _LAVA_WG1 = (64,)
# Match Lava's default once prod(ndrange) ≥ 64: 64 threads on dim 1.
# Pinning it for small K too shares that SPIR-V instead of compiling
# (18,1), (27,1), … per token. (8,8) tiling lost ~17% on live fox.
const _LAVA_WG2 = (64, 1)

# Regime II arm 1: decode attribution. Disabled unless a probe enables it.
# Counts and nanosecond totals are host-visible; they do not change op math.
mutable struct LavaDecodeAudit
    enabled::Bool
    syncs::Int
    finites::Int
    sync_ns::UInt64
    finite_ns::UInt64
    embed_n::Int
    rms_n::Int
    rope_n::Int
    matmul_n::Int
    softmax_n::Int
    swiglu_n::Int
    embed_ns::UInt64
    rms_ns::UInt64
    rope_ns::UInt64
    matmul_ns::UInt64
    softmax_ns::UInt64
    swiglu_ns::UInt64
end
const _LAVA_DECODE_AUDIT = LavaDecodeAudit(
    false, 0, 0, UInt64(0), UInt64(0),
    0, 0, 0, 0, 0, 0,
    UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
)

function _lava_audit_reset!()
    a = _LAVA_DECODE_AUDIT
    a.syncs = 0
    a.finites = 0
    a.sync_ns = UInt64(0)
    a.finite_ns = UInt64(0)
    a.embed_n = 0
    a.rms_n = 0
    a.rope_n = 0
    a.matmul_n = 0
    a.softmax_n = 0
    a.swiglu_n = 0
    a.embed_ns = UInt64(0)
    a.rms_ns = UInt64(0)
    a.rope_ns = UInt64(0)
    a.matmul_ns = UInt64(0)
    a.softmax_ns = UInt64(0)
    a.swiglu_ns = UInt64(0)
    return a
end

function _lava_audit_enable!(on::Bool=true)
    _LAVA_DECODE_AUDIT.enabled = on
    on && _lava_audit_reset!()
    return _LAVA_DECODE_AUDIT
end

function _lava_audit_snapshot()
    a = _LAVA_DECODE_AUDIT
    return (;
        enabled=a.enabled,
        syncs=a.syncs,
        finites=a.finites,
        sync_ns=a.sync_ns,
        finite_ns=a.finite_ns,
        embed_n=a.embed_n,
        rms_n=a.rms_n,
        rope_n=a.rope_n,
        matmul_n=a.matmul_n,
        softmax_n=a.softmax_n,
        swiglu_n=a.swiglu_n,
        embed_ns=a.embed_ns,
        rms_ns=a.rms_ns,
        rope_ns=a.rope_ns,
        matmul_ns=a.matmul_ns,
        softmax_ns=a.softmax_ns,
        swiglu_ns=a.swiglu_ns,
    )
end

function _lava_note_op!(op::Symbol, dt::UInt64)
    a = _LAVA_DECODE_AUDIT
    a.enabled || return
    if op === :embed
        a.embed_n += 1
        a.embed_ns += dt
    elseif op === :rmsnorm
        a.rms_n += 1
        a.rms_ns += dt
    elseif op === :rope
        a.rope_n += 1
        a.rope_ns += dt
    elseif op === :matmul
        a.matmul_n += 1
        a.matmul_ns += dt
    elseif op === :softmax
        a.softmax_n += 1
        a.softmax_ns += dt
    elseif op === :swiglu
        a.swiglu_n += 1
        a.swiglu_ns += dt
    end
    return
end

function _lava_sync!()
    a = _LAVA_DECODE_AUDIT
    if a.enabled
        t = time_ns()
        KA.synchronize(KALava())
        a.sync_ns += time_ns() - t
        a.syncs += 1
        return
    end
    KA.synchronize(KALava())
    return
end

# Prefill keeps the Phase 8 per-op wait (host-visible logits). Decode
# submits and waits once at the engine boundary — CUDA never waited per op.
# Arm 1 receipt: 605 waits/token, but KA.synchronize was 11% of decode wall.
_lava_after_op!(::Gesso.PrefillWorkload) = _lava_sync!()
_lava_after_op!(::Gesso.DecodeWorkload) = nothing

# Decode never waits per op, so without this a stage timer reads only the
# submit cost and the queued work is charged to the next stage that blocks.
Gesso.Inference._profile_sync(::LavaBackend) = _lava_sync!()

# A completed engine action leaves all primary device work synchronized.
function Gesso.Inference._engine_boundary!(::LavaBackend)
    _lava_sync!()
    return nothing
end

function Gesso.Inference._engine_failure(err::Lava.LavaError)
    code=err.operation in ("memory allocation","pool block allocation") ? Gesso.ERR_ALLOCATION : Gesso.ERR_RUNTIME
    return gesso_error(code,"primary Lava operation failed";cause=sprint(showerror,err),cause_type=string(typeof(err)),interrupted=false,backend=:lava)
end

# --- device guard (the interpreter does not copy) -----------------------------

# every op first checks that storage actually lives on the device: handing
# Lava ops CPU memory is ERR_INVALID_PLAN — the caller skipped to_device
# (§LXXXI: "the interpreter does not copy"; §LXX: loud, typed failure).
# Mirrors the CUDA extension's guard: only the host/device BOUNDARY matters;
# device-side views are legal (the interpreter legitimately slices device
# buffers — generate's prefill rows).
function _lava_device_storage!(op::Symbol, t)
    s = t.storage
    Gesso.Inference._storage_root(s) isa Lava.LavaArray && eltype(s) === Float32 && return s
    # Quantized dense weights keep their bytes in a device LavaArray{UInt8}
    # (lava_quant.jl); that is device storage at the same boundary.
    s isa LavaDenseQuant && s.data isa Lava.LavaArray && return s
    throw(gesso_error(ERR_INVALID_PLAN,
        "$op: LavaBackend requires its own device storage (Float32 or LavaDenseQuant); host Array, " *
        "host views, other devices and unsupported arithmetic are rejected; call to_device explicitly";
        op=op, storage_type=string(typeof(s))))
end
# Expert banks stay host-resident GgufQuantSlice storage: the MoE route runs
# them on the host (GESSO_MOE_HOST_EXPERTS=1), the same split as CUDA. Only
# expert banks are exempt; a dense weight still on the host fails the guard.
# A device expert bank is refused where it would be used (_moe_topk_swiglu!).
function Gesso.Inference._check_device_storage(op::Symbol, backend::LavaBackend, t)
    t isa Gesso.ExpertWeight && t.storage isa Gesso.Inference.GgufQuantSlice && return nothing
    return _lava_device_storage!(op, t)
end

# On the tested Vulkan compiler path, the floating `isfinite` predicate
# reports NaN and Inf as finite. Classify their IEEE exponent bits instead;
# all data stays on-device and only the UInt32 reduction result is read.
_lava_nonfinite_flag(x::Float32) = UInt32((reinterpret(UInt32, x) & 0x7f800000) == 0x7f800000)
_lava_nonfinite_flag(x::Float64) = UInt32((reinterpret(UInt64, x) & 0x7ff0000000000000) == 0x7ff0000000000000)
function Gesso.Inference._all_finite(a::Lava.AnyLavaArray{T}) where {T <: Union{Float32,Float64}}
    # Explicit ordinary device temporary avoids KA's tiny BAR allocation.
    a_audit = _LAVA_DECODE_AUDIT
    t0 = a_audit.enabled ? time_ns() : UInt64(0)
    temp=Lava.LavaArray{UInt32}(undef,(max(2,2*cld(length(a),128)),))
    ok = Lava.AK.mapreduce(_lava_nonfinite_flag,max,a,KA.get_backend(a);
        init=UInt32(0),neutral=UInt32(0),temp,block_size=64,switch_below=0)==UInt32(0)
    if a_audit.enabled
        a_audit.finite_ns += time_ns() - t0
        a_audit.finites += 1
    end
    return ok
end

function _lava_dimreduce(f,op,a,dim,init)
    shape=ntuple(i -> i==dim ? 1 : size(a,i),ndims(a))
    prod(shape)>1 && return mapreduce(f,op,a;dims=dim)
    # Preserve Lava's native scalar Float32 sum and the legacy 1D tree.
    f===identity && op===(+) && eltype(a)===Float32 && return sum(a;dims=dim)
    result=Lava.LavaArray{typeof(init)}(undef,shape)
    temp=Lava.LavaArray{typeof(init)}(undef,(max(2,2*cld(length(a),128)),))
    value=Lava.AK.mapreduce(f,op,a,KA.get_backend(a);init,neutral=init,temp,block_size=64,switch_below=0)
    copyto!(result,1,[value],1,1)
    return result
end

# --- op bodies -----------------------------------------------------------------

function _lava_embedding_lookup!(dst, table, tokens)
    a = _LAVA_DECODE_AUDIT
    t0 = a.enabled ? time_ns() : UInt64(0)
    Gesso._validate_embedding_inputs(dst,table,tokens)
    tab = table.storage
    # 0-based token ids (§LXXV) → 1-based rows on the HOST (cheap), then one
    # broadcast gather on device: rows land in dst in token order.
    tok = Lava.LavaArray{Int64}(collect(Int64, tokens) .+ 1)
    dst.storage .= tab[tok, :]
    a.enabled && _lava_note_op!(:embed, time_ns() - t0)
    return dst
end

# Single-row decode normalization. Matches `_cuda_rmsnorm_row_kernel!` in
# arithmetic: F32 accumulate, F64 rms, F32 write. The sum is split across one
# workgroup so a feature dim of several thousand is not walked by one lane.
const _LAVA_RMS_LANES = 64

@kernel cpu=false function _lava_rmsnorm_row_kernel!(dst, xs, scales, eps, d::Int)
    tl = @index(Local, Linear)
    part = @localmem Float32 (_LAVA_RMS_LANES,)
    acc = Float32(0)
    for j in tl:_LAVA_RMS_LANES:d
        acc += abs2(xs[1, j])
    end
    part[tl] = acc
    # every lane reads all partials after this barrier, so no lane can see a stale slot
    @synchronize
    total = Float32(0)
    for i in 1:_LAVA_RMS_LANES
        total += part[i]
    end
    rms = sqrt(Float64(total) / d + Float64(eps))
    for j in tl:_LAVA_RMS_LANES:d
        dst[1, j] = Float32((Float64(xs[1, j]) / rms) * Float64(scales[j]))
    end
end

# Single-token per-head normalization for (1, heads, d_head) inputs (QK-norm in
# decode). One workgroup per head runs the same lane-split reduction as the row
# kernel, so each head gets the row kernel's arithmetic without a host loop.
@kernel cpu=false function _lava_rmsnorm_head_kernel!(dst, xs, scales, eps, d::Int)
    g = @index(Group, Linear)
    tl = @index(Local, Linear)
    part = @localmem Float32 (_LAVA_RMS_LANES,)
    acc = Float32(0)
    for j in tl:_LAVA_RMS_LANES:d
        acc += abs2(xs[1, g, j])
    end
    part[tl] = acc
    @synchronize
    total = Float32(0)
    for i in 1:_LAVA_RMS_LANES
        total += part[i]
    end
    rms = sqrt(Float64(total) / d + Float64(eps))
    for j in tl:_LAVA_RMS_LANES:d
        dst[1, g, j] = Float32((Float64(xs[1, g, j]) / rms) * Float64(scales[j]))
    end
end

function _lava_rmsnorm!(dst, x, scale; eps = 1e-6, check_finite = true)
    a = _LAVA_DECODE_AUDIT
    t0 = a.enabled ? time_ns() : UInt64(0)
    isfinite(eps) && eps > 0 || throw(gesso_error(ERR_INVALID_PLAN,
        "rmsnorm!: eps must be finite and positive"))
    xs = x.storage
    native_eps=eltype(xs)(eps)
    isfinite(native_eps) && native_eps>0 || throw(gesso_error(ERR_INVALID_PLAN,
        "rmsnorm!: eps must be representable, finite and positive in storage dtype"))
    d = size(xs, ndims(xs))                       # last dim is the feature dim
    length(scale.storage) == d ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(scale.storage)) ≠ feature dim $d"; op = :rmsnorm!))
    if ndims(xs) == 2 && size(xs, 1) == 1
        kern = _lava_rmsnorm_row_kernel!(KA.get_backend(xs))
        kern(dst.storage, xs, scale.storage, Float64(eps), Int(d);
            ndrange=_LAVA_RMS_LANES, workgroupsize=(_LAVA_RMS_LANES,))
        if check_finite
            Gesso.Inference._all_finite(dst.storage) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
                "rmsnorm!: nonfinite normalization denominator"))
        end
        a.enabled && _lava_note_op!(:rmsnorm, time_ns() - t0)
        return dst
    end
    if ndims(xs) == 3 && size(xs, 1) == 1
        nheads = size(xs, 2)
        kern = _lava_rmsnorm_head_kernel!(KA.get_backend(xs))
        kern(dst.storage, xs, scale.storage, Float64(eps), Int(d);
            ndrange=_LAVA_RMS_LANES * nheads, workgroupsize=(_LAVA_RMS_LANES,))
        if check_finite
            Gesso.Inference._all_finite(dst.storage) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
                "rmsnorm!: nonfinite normalization denominator"))
        end
        a.enabled && _lava_note_op!(:rmsnorm, time_ns() - t0)
        return dst
    end
    rms = sqrt.(_lava_dimreduce(abs2,+,xs,ndims(xs),zero(eltype(xs))) ./ eltype(xs)(d) .+ native_eps)
    if check_finite
        Gesso.Inference._all_finite(rms) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
            "rmsnorm!: nonfinite normalization denominator"))
    end
    # scale indexes the LAST axis: trailing 1s so it broadcasts on device too
    stail = reshape(scale.storage, (ntuple(_ -> 1, ndims(xs) - 1)..., d))
    dst.storage .= (xs ./ rms) .* stail
    a.enabled && _lava_note_op!(:rmsnorm, time_ns() - t0)
    return dst
end

# RoPE: adjacent pairs by default; imported HF policy selects half-split
# law), Q over its head axis and K over its own (GQA-safe, Phase 3 fix).
# Broadcast cannot express the PAIR COUPLING with a single fused statement,
# but strided views over the odd/even feature planes express it exactly:
# rotate by an angle table that depends on (position, feature pair) — never
# on the head — so it broadcasts across heads (GQA for free).
#
# x1/x2 are views into the SAME array; both outputs read the ORIGINAL
# values, so the second result is staged in a temporary before either view
# is written. Angles are computed in F32 (the declared device math, §LXXXI;
# the F64 CPU oracle differs by ~1e-7 relative here, far inside atol=1e-3).
# On-device RoPE: one (seq, head) thread, angles from theta like
# `_cuda_rope_kernel!`. Scaled `inv_freq` policies keep the host table.
# `rd` is the rotary width: only the first rd features rotate, and the rest
# pass through unchanged, as in `_cpu_rope_storage!`. rd == d is the full-head
# case and reproduces the previous kernels exactly.
@kernel cpu=false function _lava_rope_kernel!(x, positions, theta, interleaved::Bool, rd::Int)
    idx = @index(Global, Cartesian)
    t = idx[1]
    h = idx[2]
    seq, nheads, _ = size(x)
    (t > seq || h > nheads) && return
    m = Float32(positions[t])
    half = rd ÷ 2
    for i in 0:(half - 1)
        θv = m * (theta ^ (-2 * i / rd))
        c = cos(θv)
        s = sin(θv)
        a = interleaved ? (2 * i + 1) : (i + 1)
        b = interleaved ? (2 * i + 2) : (i + 1 + half)
        x1 = x[t, h, a]
        x2 = x[t, h, b]
        x[t, h, a] = x1 * c - x2 * s
        x[t, h, b] = x1 * s + x2 * c
    end
end

# Decode seq=1: position is a kernel scalar so we do not allocate a 4-byte
# device buffer 30 times per token (arm 5c: that upload ate the kernel win).
@kernel cpu=false function _lava_rope_decode_kernel!(x, pos::Int32, theta, interleaved::Bool, rd::Int)
    h = @index(Global)
    _, nheads, _ = size(x)
    h > nheads && return
    m = Float32(pos)
    half = rd ÷ 2
    for i in 0:(half - 1)
        θv = m * (theta ^ (-2 * i / rd))
        c = cos(θv)
        s = sin(θv)
        a = interleaved ? (2 * i + 1) : (i + 1)
        b = interleaved ? (2 * i + 2) : (i + 1 + half)
        x1 = x[1, h, a]
        x2 = x[1, h, b]
        x[1, h, a] = x1 * c - x2 * s
        x[1, h, b] = x1 * s + x2 * c
    end
end

function _lava_rope!(q, k, positions; theta = 10000.0, inv_freq = nothing, interleaved = true, rotary_dim = 0)
    a = _LAVA_DECODE_AUDIT
    t0 = a.enabled ? time_ns() : UInt64(0)
    size(q.storage, 3) == size(k.storage, 3) ||
        throw(gesso_error(ERR_INVALID_PLAN, "rope!: q d_head ≠ k d_head"; op = :rope!))
    Gesso._validate_rope_inputs(q,k,positions,theta,inv_freq,rotary_dim)
    tθ = Float64(theta)
    d = size(q.storage, 3)
    rd = rotary_dim == 0 ? d : Int(rotary_dim)
    if inv_freq === nothing && length(positions) == 1 && size(q.storage, 1) == 1
        pos = Int32(positions[1])
        for x in (q.storage, k.storage)
            nheads = size(x, 2)
            kern = _lava_rope_decode_kernel!(KA.get_backend(x))
            kern(x, pos, tθ, interleaved, rd; ndrange=nheads, workgroupsize=_LAVA_WG1)
        end
        a.enabled && _lava_note_op!(:rope, time_ns() - t0)
        return q
    end
    if inv_freq === nothing
        pos = Lava.LavaArray{Int32}(Int32.(positions))
        for x in (q.storage, k.storage)
            seq, nheads, _ = size(x)
            kern = _lava_rope_kernel!(KA.get_backend(x))
            kern(x, pos, tθ, interleaved, rd; ndrange=(seq, nheads), workgroupsize=_LAVA_WG2)
        end
        a.enabled && _lava_note_op!(:rope, time_ns() - t0)
        return q
    end
    if length(positions) == 1 && size(q.storage, 1) == 1
        row = Int32(Int(positions[1]) + 1)
        cs, sn = _lava_rope_table(Float32.(inv_freq), rd, Int(row))
        for x in (q.storage, k.storage)
            nheads = size(x, 2)
            kern = _lava_rope_table_kernel!(KA.get_backend(x))
            kern(x, cs, sn, row, interleaved, rd; ndrange=nheads, workgroupsize=_LAVA_WG1)
        end
        a.enabled && _lava_note_op!(:rope, time_ns() - t0)
        return q
    end
    seq = size(q.storage, 1)
    half = rd ÷ 2
    freq = Float32.(inv_freq)
    # Match the native Float32 phase, then reduce metadata angles accurately.
    # Only derived constants are uploaded; Q/K and their rotations stay GPU.
    phase = reshape(Float32.(positions), seq, 1) .* reshape(freq, 1, half)
    reduced = Float32.(rem.(Float64.(phase), 2π))
    ang = Lava.LavaArray{Float32}(reshape(reduced, seq, 1, half))
    c = cos.(ang)
    s = sin.(ang)
    for x in (q.storage, k.storage)
        x1 = view(x, :, :, interleaved ? (1:2:rd) : (1:half))          # odd features  (2i+1)
        x2 = view(x, :, :, interleaved ? (2:2:rd) : ((half+1):rd))     # even features (2i+2)
        n2 = x1 .* s .+ x2 .* c                                        # staged: reads ORIGINAL x1, x2
        x1 .= x1 .* c .- x2 .* s
        x2 .= n2
    end
    a.enabled && _lava_note_op!(:rope, time_ns() - t0)
    return q
end

# Scaled inv_freq policies on single-token decode. The general path below
# uploads a per-call angle table; decode instead reads one row of a device
# cos/sin table keyed by (rotary width, inv_freq). Rows are built with the
# same F32 phase and F64 reduction as the general path, so both agree to
# rounding. The table grows by doubling, so steady-state decode uploads
# nothing and allocates nothing.
@kernel cpu=false function _lava_rope_table_kernel!(x, cos_t, sin_t, row::Int32, interleaved::Bool, rd::Int)
    h = @index(Global)
    _, nheads, _ = size(x)
    h > nheads && return
    half = rd ÷ 2
    for i in 1:half
        c = cos_t[row, i]
        s = sin_t[row, i]
        a = interleaved ? (2 * i - 1) : i
        b = interleaved ? (2 * i) : (i + half)
        x1 = x[1, h, a]
        x2 = x[1, h, b]
        x[1, h, a] = x1 * c - x2 * s
        x[1, h, b] = x1 * s + x2 * c
    end
end

const _LAVA_ROPE_TABLES = Dict{Tuple{Int, Vector{Float32}}, Any}()

function _lava_rope_table(freq::Vector{Float32}, rd::Int, row::Int)
    key = (rd, freq)
    ent = get(_LAVA_ROPE_TABLES, key, nothing)
    if ent === nothing || ent.cap < row
        cap = max(row, ent === nothing ? 256 : 2 * ent.cap)
        half = length(freq)
        cosm = Matrix{Float32}(undef, cap, half)
        sinm = Matrix{Float32}(undef, cap, half)
        for p in 0:(cap - 1), i in 1:half
            phase = Float32(p) * freq[i]
            r = Float32(rem(Float64(phase), 2π))
            cosm[p + 1, i] = cos(r)
            sinm[p + 1, i] = sin(r)
        end
        if ent !== nothing
            Lava.unsafe_free!(ent.cos)
            Lava.unsafe_free!(ent.sin)
        end
        ent = (cap = cap, cos = Lava.LavaArray{Float32}(cosm), sin = Lava.LavaArray{Float32}(sinm))
        _LAVA_ROPE_TABLES[key] = ent
    end
    return ent.cos, ent.sin
end

# Causal softmax over (L, K) score rows, rows aligned to the LAST K keys
# (key offset = K − L; prefill L==K masks the upper triangle, a decode row
# attends to everything) — same contract as the CPU kernel, §LXXV.
# max-subtract for stability; exp(−Inf) = 0 makes the mask implicit in the
# exponential; every row has at least one unmasked key (its own), so the
# denominator is never zero.
# One thread per query row. Mask, max, exp, sum, write dst — CUDA's
# `_cuda_softmax_kernel!` contract, on Lava storage. Decode (L=1) is the
# measured 62% wall; the kernel also covers prefill L>1.
@kernel cpu=false function _lava_softmax_row_kernel!(dst, scores, L::Int, K::Int)
    i = @index(Global)
    i > L && return
    offset = K - L
    j0 = offset + i
    row_max = Float32(-Inf)
    for j in 1:K
        v = j > j0 ? Float32(-Inf) : scores[i, j]
        row_max = ifelse(v > row_max, v, row_max)
    end
    acc = Float32(0)
    for j in 1:K
        v = j > j0 ? Float32(-Inf) : scores[i, j]
        e = exp(v - row_max)
        scores[i, j] = e
        acc += e
    end
    inv = acc == Float32(0) ? Float32(0) : (Float32(1) / acc)
    for j in 1:K
        p = scores[i, j] * inv
        scores[i, j] = p
        dst[i, j] = p
    end
end

function _lava_softmax_fused!(dst, scores)
    s = scores.storage
    d = dst.storage
    L, K = size(s)
    L == 0 && return dst
    kern = _lava_softmax_row_kernel!(KA.get_backend(s))
    kern(d, s, Int(L), Int(K); ndrange=L, workgroupsize=_LAVA_WG1)
    return dst
end

function _lava_softmax!(dst, scores; check_finite = true)
    a = _LAVA_DECODE_AUDIT
    t0 = a.enabled ? time_ns() : UInt64(0)
    if !check_finite
        _lava_softmax_fused!(dst, scores)
        a.enabled && _lava_note_op!(:softmax, time_ns() - t0)
        return dst
    end
    s = scores.storage
    L, K = size(s)
    offset = K - L
    qi = reshape(Lava.LavaArray{Int64}(1:L), L, 1)
    kj = reshape(Lava.LavaArray{Int64}(1:K), 1, K)
    masked = ifelse.(kj .> (qi .+ offset), -Inf32, s)
    rowmax = _lava_dimreduce(identity,max,masked,2,-Inf32)
    Gesso.Inference._all_finite(rowmax) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
        "softmax!: no finite unmasked maximum"))
    e = exp.(masked .- rowmax)
    den = _lava_dimreduce(identity,+,e,2,0f0)
    dst.storage .= e ./ den
    a.enabled && _lava_note_op!(:softmax, time_ns() - t0)
    return dst
end

function _lava_swiglu!(dst, gate, up)
    a = _LAVA_DECODE_AUDIT
    t0 = a.enabled ? time_ns() : UInt64(0)
    g = gate.storage
    dst.storage .= (g ./ (1f0 .+ exp.(-g))) .* up.storage
    a.enabled && _lava_note_op!(:swiglu, time_ns() - t0)
    return dst
end

function _lava_matmul!(dst, x, w)
    a = _LAVA_DECODE_AUDIT
    t0 = a.enabled ? time_ns() : UInt64(0)
    if w.storage isa LavaDenseQuant
        _lq_matmul!(dst.storage, x.storage, w.storage)
        a.enabled && _lava_note_op!(:matmul, time_ns() - t0)
        return dst
    end
    T = eltype(dst.storage)
    mul!(dst.storage, x.storage, transpose(w.storage), one(T), zero(T))
    a.enabled && _lava_note_op!(:matmul, time_ns() - t0)
    return dst
end

# --- dispatch surface: LavaBackend × families × workload ------------------------

function embedding_lookup!(
    ::LavaBackend,
    dst::Activation,
    table::EmbeddingTable,
    tokens::AbstractVector{Int},
    workload::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:embedding_lookup!, dst)
    _lava_device_storage!(:embedding_lookup!, table)
    r = _lava_embedding_lookup!(dst, table, tokens)
    _lava_after_op!(workload)
    return r
end

function embedding_lookup!(
    ::LavaBackend,
    dst::Activation,
    table::EmbeddingTable,
    tokens::AbstractVector{Int},
    workload::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:embedding_lookup!, dst)
    _lava_device_storage!(:embedding_lookup!, table)
    r = _lava_embedding_lookup!(dst, table, tokens)
    _lava_after_op!(workload)
    return r
end

function rmsnorm!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    workload::Gesso.PrefillWorkload;
    eps::Real = 1e-6,
)
    _lava_device_storage!(:rmsnorm!, dst)
    _lava_device_storage!(:rmsnorm!, x)
    r = _lava_rmsnorm!(dst, x, scale; eps)
    _lava_after_op!(workload)
    return r
end

function rmsnorm!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    workload::Gesso.DecodeWorkload;
    eps::Real = 1e-6,
)
    _lava_device_storage!(:rmsnorm!, dst)
    _lava_device_storage!(:rmsnorm!, x)
    r = _lava_rmsnorm!(dst, x, scale; eps, check_finite=false)
    _lava_after_op!(workload)
    return r
end

function rope!(
    ::LavaBackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    workload::Gesso.PrefillWorkload;
    theta::Real = 10000.0,
    inv_freq = nothing,
    interleaved::Bool = true,
    rotary_dim::Integer = 0,
)
    # Imported default, linear and Llama3 frequency metadata travels with Q/K.
    # Validate and rotate on the requested Lava storage (§LXX: no silent
    # representation change).
    _lava_device_storage!(:rope!, q)
    _lava_device_storage!(:rope!, k)
    r = _lava_rope!(q, k, positions; theta, inv_freq, interleaved, rotary_dim)
    _lava_after_op!(workload)
    return r
end

function rope!(
    ::LavaBackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    workload::Gesso.DecodeWorkload;
    theta::Real = 10000.0,
    inv_freq = nothing,
    interleaved::Bool = true,
    rotary_dim::Integer = 0,
)
    # Imported default, linear and Llama3 frequency metadata travels with Q/K.
    # Validate and rotate on the requested Lava storage (§LXX: no silent
    # representation change).
    _lava_device_storage!(:rope!, q)
    _lava_device_storage!(:rope!, k)
    r = _lava_rope!(q, k, positions; theta, inv_freq, interleaved, rotary_dim)
    _lava_after_op!(workload)
    return r
end

function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::ProjectionWeight,
    workload::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_after_op!(workload)
    return r
end

function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::ProjectionWeight,
    workload::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_after_op!(workload)
    return r
end

# tied head (§LXXV): the embedding table IS the lm_head — same function,
# no second vocabulary, no second table
function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::EmbeddingTable,
    workload::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_after_op!(workload)
    return r
end

function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::EmbeddingTable,
    workload::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_after_op!(workload)
    return r
end

function softmax!(
    ::LavaBackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    workload::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:softmax!, dst)
    _lava_device_storage!(:softmax!, scores)
    r = _lava_softmax!(dst, scores)
    _lava_after_op!(workload)
    return r
end

function softmax!(
    ::LavaBackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    workload::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:softmax!, dst)
    _lava_device_storage!(:softmax!, scores)
    r = _lava_softmax!(dst, scores; check_finite=false)
    _lava_after_op!(workload)
    return r
end

function swiglu!(
    ::LavaBackend,
    dst::Activation,
    gate::Activation,
    up::Activation,
    workload::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:swiglu!, dst)
    _lava_device_storage!(:swiglu!, gate)
    _lava_device_storage!(:swiglu!, up)
    r = _lava_swiglu!(dst, gate, up)
    _lava_after_op!(workload)
    return r
end

function swiglu!(
    ::LavaBackend,
    dst::Activation,
    gate::Activation,
    up::Activation,
    workload::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:swiglu!, dst)
    _lava_device_storage!(:swiglu!, gate)
    _lava_device_storage!(:swiglu!, up)
    r = _lava_swiglu!(dst, gate, up)
    _lava_after_op!(workload)
    return r
end

# --- 10F storage seams: LavaArray methods (CUDA already had these) ------------
# Generic AbstractArray bodies in Inference.jl still serve CPU. These
# replace per-head broadcasts and copied slices on Lava decode.

@kernel cpu=false function _lava_split_kernel!(dst, src, d_head::Int)
    idx = @index(Global, Cartesian)
    r = idx[1]
    h = idx[2]
    (r > size(dst, 1) || h > size(dst, 2)) && return
    coloff = (h - 1) * d_head
    for f in 1:d_head
        dst[r, h, f] = src[r, coloff + f]
    end
end

function _split_heads!(dst::Lava.LavaArray, src::Lava.LavaArray, n_heads, d_head)
    rows = size(dst, 1)
    rows == 0 && return dst
    kern = _lava_split_kernel!(KA.get_backend(dst))
    kern(dst, src, Int(d_head); ndrange=(rows, Int(n_heads)), workgroupsize=_LAVA_WG2)
    return dst
end

@kernel cpu=false function _lava_merge_kernel!(dst, src, d_head::Int)
    idx = @index(Global, Cartesian)
    r = idx[1]
    h = idx[2]
    (r > size(dst, 1) || h > size(dst, 2)) && return
    coloff = (h - 1) * d_head
    for f in 1:d_head
        dst[r, coloff + f] = src[r, h, f]
    end
end

function _merge_heads!(dst::Lava.LavaArray, src::Lava.LavaArray, n_heads, d_head)
    rows = size(dst, 1)
    rows == 0 && return dst
    kern = _lava_merge_kernel!(KA.get_backend(dst))
    kern(dst, src, Int(d_head); ndrange=(rows, Int(n_heads)), workgroupsize=_LAVA_WG2)
    return dst
end

@kernel cpu=false function _lava_repeat_kernel!(dst, src, group::Int, K::Int)
    idx = @index(Global, Cartesian)
    r = idx[1]
    sh = idx[2]
    (r > K || sh > size(src, 2)) && return
    Dh = size(dst, 3)
    for g in 1:group, f in 1:Dh
        dst[r, (sh - 1) * group + g, f] = src[r, sh, f]
    end
end

function _repeat_heads!(dst::Lava.LavaArray, src::Lava.LavaArray, group::Int, K::Int)
    group == 1 && return dst
    K == 0 && return dst
    nkv = size(src, 2)
    kern = _lava_repeat_kernel!(KA.get_backend(dst))
    kern(dst, src, group, K; ndrange=(K, nkv), workgroupsize=_LAVA_WG2)
    return dst
end

@kernel cpu=false function _lava_add_kernel!(dst, src, n::Int)
    i = @index(Global)
    i > n && return
    dst[i] += src[i]
end

function _add_storage!(dst::Lava.LavaArray, src::Lava.LavaArray)
    n = length(dst)
    n == 0 && return dst
    kern = _lava_add_kernel!(KA.get_backend(dst))
    kern(dst, src, n; ndrange=n, workgroupsize=_LAVA_WG1)
    return dst
end

# Per-head attention without copied slices. Decode L=1 is (K, Dh) · q.
@kernel cpu=false function _lava_attn_scores_kernel!(sc, q, k, hh, scale, L::Int, K::Int, Dh::Int)
    idx = @index(Global, Cartesian)
    t = idx[1]
    j = idx[2]
    (t > L || j > K) && return
    acc = Float32(0)
    for d in 1:Dh
        acc += q[t, hh, d] * k[j, hh, d]
    end
    sc[t, j] = acc * scale
end

function _attention_scores_device!(::LavaBackend, sc, q::AbstractArray, k::AbstractArray, hh, d_head, K)
    L = size(q, 1)
    L == 0 && return sc
    scale = Float32(1 / sqrt(d_head))
    kern = _lava_attn_scores_kernel!(KA.get_backend(q))
    kern(sc, q, k, Int(hh), scale, Int(L), Int(K), Int(d_head); ndrange=(L, K), workgroupsize=_LAVA_WG2)
    return sc
end

@kernel cpu=false function _lava_attn_values_kernel!(attn, probs, v, hh, L::Int, K::Int, Dh::Int)
    idx = @index(Global, Cartesian)
    t = idx[1]
    d = idx[2]
    (t > L || d > Dh) && return
    acc = Float32(0)
    for j in 1:K
        acc += probs[t, j] * v[j, hh, d]
    end
    attn[t, hh, d] = acc
end

function _attention_values_device!(::LavaBackend, attn::AbstractArray, probs, v::AbstractArray, hh, K)
    L = size(attn, 1)
    Dh = size(attn, 3)
    L == 0 && return attn
    kern = _lava_attn_values_kernel!(KA.get_backend(attn))
    kern(attn, probs, v, Int(hh), Int(L), Int(K), Int(Dh); ndrange=(L, Dh), workgroupsize=_LAVA_WG2)
    return attn
end

# Decode L=1: one scores kernel, one softmax, one values kernel per layer.
# Each score row is a full key prefix (no causal offset — K already is the
# filled cache). Prefill keeps the per-head loop in `_attention_heads!`.
@kernel cpu=false function _lava_attn_scores_batched_kernel!(sc, q, k, scale, H::Int, K::Int, Dh::Int)
    idx = @index(Global, Cartesian)
    h = idx[1]
    j = idx[2]
    (h > H || j > K) && return
    acc = Float32(0)
    for d in 1:Dh
        acc += q[1, h, d] * k[j, h, d]
    end
    sc[h, j] = acc * scale
end

@kernel cpu=false function _lava_softmax_heads_kernel!(dst, scores, H::Int, K::Int)
    h = @index(Global)
    h > H && return
    row_max = Float32(-Inf)
    for j in 1:K
        v = scores[h, j]
        row_max = ifelse(v > row_max, v, row_max)
    end
    acc = Float32(0)
    for j in 1:K
        e = exp(scores[h, j] - row_max)
        scores[h, j] = e
        acc += e
    end
    inv = acc == Float32(0) ? Float32(0) : (Float32(1) / acc)
    for j in 1:K
        p = scores[h, j] * inv
        scores[h, j] = p
        dst[h, j] = p
    end
end

@kernel cpu=false function _lava_attn_values_batched_kernel!(attn, probs, v, H::Int, K::Int, Dh::Int)
    idx = @index(Global, Cartesian)
    h = idx[1]
    d = idx[2]
    (h > H || d > Dh) && return
    acc = Float32(0)
    for j in 1:K
        acc += probs[h, j] * v[j, h, d]
    end
    attn[1, h, d] = acc
end

function _attention_heads!(
    ::LavaBackend,
    attn::AbstractArray,
    scores,
    scores_out,
    q::AbstractArray,
    k::AbstractArray,
    v::AbstractArray,
    n_heads::Int,
    d_head::Int,
    L::Int,
    K::Int,
    workload,
)
    sc, probs = scores.storage, scores_out.storage
    if L == 1 && ndims(sc) == 2 && size(sc, 1) == n_heads && size(sc, 2) == K
        a = _LAVA_DECODE_AUDIT
        t0 = a.enabled ? time_ns() : UInt64(0)
        scale = Float32(1 / sqrt(d_head))
        backend = KA.get_backend(q)
        kern_s = _lava_attn_scores_batched_kernel!(backend)
        kern_s(sc, q, k, scale, n_heads, Int(K), Int(d_head); ndrange=(n_heads, K), workgroupsize=_LAVA_WG2)
        kern_m = _lava_softmax_heads_kernel!(backend)
        kern_m(probs, sc, n_heads, Int(K); ndrange=n_heads, workgroupsize=_LAVA_WG1)
        kern_v = _lava_attn_values_batched_kernel!(backend)
        kern_v(attn, probs, v, n_heads, Int(K), Int(d_head); ndrange=(n_heads, d_head), workgroupsize=_LAVA_WG2)
        if a.enabled
            _lava_note_op!(:softmax, time_ns() - t0)
        end
        return attn
    end
    for hh in 1:n_heads
        _attention_scores_device!(LavaBackend(), sc, q, k, hh, d_head, K)
        softmax!(LavaBackend(), scores_out, scores, workload)
        _attention_values_device!(LavaBackend(), attn, probs, v, hh, K)
    end
    return attn
end

# quantize! / dequantize!: NO Lava method — the generic decline still fires
# (their math is a Representation-phase concern, §LXXXI).

# --- explicit transfer (§LXXXI: "the interpreter does not copy") ---------------

"""
    to_device(::LavaBackend, tensors) -> tensors'

Copy `Array` storage to `LavaArray{Float32}` (F64→F32 conversion is part of
the lowering) and return a new named tuple of the same shape — `embedding`,
`blocks`, `lm_head`, and `final_rms` when present. NEVER mutates the CPU
tensors. Tied heads stay tied on device: `lm_head === embedding`.
"""
function _lava_to_device(tensors)
    # GGUF quant matrices dequantize to F32 here (embedding rows are gathered
    # from F32 storage). Projection weights stay as GGUF bytes when they are
    # eligible and fit the dense budget (the CUDA rule, GESSO_GPU_BUDGET_MIB).
    budget = Ref(Gesso.Inference._gpu_budget_bytes())
    _to_f32(s) = s isa Lava.LavaArray ? s :
        s isa Gesso.Inference.QuantMatrix ? Lava.LavaArray{Float32}(Gesso.Inference._quant_host_f32(s)) :
        Lava.LavaArray{Float32}(s)
    _to_proj(s) = begin
        if s isa Gesso.Inference.QuantMatrix && _lq_eligible(s) && _lq_device_bytes(s) <= budget[]
            budget[] -= _lq_device_bytes(s)
            return _lq_upload(s)
        end
        return _to_f32(s)
    end
    # tensors in the map carry (shape, storage); blocks are NAMED TUPLES of
    # tensors, so their fields convert field-wise
    _conv_t(t; proj::Bool = false) = begin
        hasproperty(t, :storage) || return t         # mixer tags and other metadata
        t.storage === nothing && return t            # unset storage stays unset
        st = proj && t isa Gesso.ProjectionWeight ? _to_proj(t.storage) : _to_f32(t.storage)
        typeof(t)(; shape = t.shape, storage = st)
    end
    # Routed expert banks stay on the host (the CUDA GESSO_MOE_HOST_EXPERTS
    # split): a GGUF slab is never dequantized to device F32 here. The host
    # MoE route in lava_hybrid.jl reads these banks in place.
    _host_expert_key(k) = k in (:wexp_gate, :wexp_up, :wexp_down)
    _host_expert_t(t) = hasproperty(t, :storage) && t.storage isa Gesso.Inference.GgufQuantSlice
    _conv_blk(b) = NamedTuple{keys(b)}(map(keys(b)) do k
        t = getproperty(b, k)
        _host_expert_key(k) && _host_expert_t(t) ? t : _conv_t(t; proj = true)
    end)
    embedding = _conv_t(tensors.embedding)
    blocks = [_conv_blk(b) for b in tensors.blocks]
    lm_head = tensors.lm_head === tensors.embedding ? embedding : _conv_t(tensors.lm_head; proj = true)
    fr = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    final_rms = fr === nothing ? nothing : _conv_t(fr)
    _lava_sync!()
    result = (; embedding, blocks, lm_head, final_rms)
    return merge(result, _keep_tensor_meta(tensors))
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

# ext-local dispatch wrapper (bound as Gesso.to_device when this extension is
# the first backend extension to load; otherwise __init__ attaches the worker
# to the already-installed function object)
function to_device(::LavaBackend, tensors)
    return _lava_to_device(tensors)
end

# --- decode activation arena ---------------------------------------------------
# A decode step asks for the same sequence of activation shapes every token, so
# the n-th request is served from the n-th view the previous step laid out.
# The views are slices of one slab, so the step's zeroing is one fill! instead
# of one allocation and one fill per buffer. Without this, ~185 fresh device
# buffers per 35B token leave the pool under pressure and maybe_collect runs
# GC.gc on the decode path. Requests within a step never share a view, and a
# step whose requests differ from the layout gets fresh buffers for the ones
# that do not fit; the next step lays the slab out again.
#
# A request the slab cannot hold (rank above four, or empty) keeps its place in
# the layout with no slab space and a `nothing` view. It is served loose, and
# because its slot is kept, one such request does not drop the layout for the
# other requests in the step.
mutable struct LavaArenaSlab
    slab::Union{Nothing, Lava.LavaArray{Float32, 1}}
    views::Vector{Any}
    layout::Vector{NTuple{5, Int}}
    requested::Vector{NTuple{5, Int}}
    loose::Vector{Any}
end

struct LavaDecodeArena
    s::LavaArenaSlab
    next::Base.RefValue{Int}
end

# (ndims, dims padded to four). Rank above four or an empty buffer gets key 0:
# a request the slab cannot hold.
function _lava_arena_key(dims::NTuple{N, Int}) where {N}
    (N <= 4 && prod(dims) > 0) || return (0, 0, 0, 0, 0)
    return (N, ntuple(k -> k <= N ? dims[k] : 0, 4)...)
end

# 64 Float32s: views start on 256-byte boundaries, as separate allocations would.
_lava_arena_round(n::Int) = cld(n, 64) * 64

function _lava_arena_relayout!(s::LavaArenaSlab)
    for v in s.views
        v === nothing || Lava.unsafe_free!(v)
    end
    s.slab === nothing || Lava.unsafe_free!(s.slab)
    empty!(s.views)
    copy!(s.layout, s.requested)
    total = 0
    for key in s.layout
        key[1] == 0 || (total += _lava_arena_round(prod(key[2:(key[1] + 1)])))
    end
    s.slab = Lava.LavaArray{Float32}(undef, max(total, 1))
    off = 0
    for key in s.layout
        if key[1] == 0
            push!(s.views, nothing)
            continue
        end
        dims = key[2:(key[1] + 1)]
        n = prod(dims)
        push!(s.views, reshape(view(s.slab, (off + 1):(off + n)), dims))
        off += _lava_arena_round(n)
    end
    return s
end

function (a::LavaDecodeArena)(dims::NTuple{N, Int}) where {N}
    s = a.s
    i = (a.next[] += 1)
    key = _lava_arena_key(dims)
    push!(s.requested, key)
    if key[1] != 0 && i <= length(s.layout) && s.layout[i] == key
        return s.views[i]::Lava.LavaArray{Float32, N}
    end
    buf = fill!(Lava.LavaArray{Float32}(undef, dims...), 0.0f0)
    push!(s.loose, buf)
    return buf
end

const _LAVA_DECODE_ARENA = LavaArenaSlab(
    nothing, Any[], NTuple{5, Int}[], NTuple{5, Int}[], Any[])

# Mirrors the CUDA arena's step protocol (ext/cuda_ops.jl). Freeing the previous
# step's views and slab is safe only because nothing taken from this factory
# outlives its step (the decode contract in Inference.jl).
function Gesso.Inference._decode_zeros_like(be::LavaBackend, tensors)
    proto = Gesso.Inference._first_array_storage(tensors)
    proto isa Lava.LavaArray{Float32} ||
        return Gesso.Inference._zeros_like_factory(be, tensors)
    s = _LAVA_DECODE_ARENA
    # `loose` holds only the previous step's overflow buffers, so it is always
    # released here; the layout is rebuilt only when the requests changed.
    for b in s.loose
        Lava.unsafe_free!(b)
    end
    empty!(s.loose)
    if !isempty(s.requested) && s.requested != s.layout
        _lava_arena_relayout!(s)
    end
    empty!(s.requested)
    s.slab === nothing || fill!(s.slab, 0.0f0)
    return LavaDecodeArena(s, Ref(0))
end
