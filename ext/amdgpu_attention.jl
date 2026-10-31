# Attention on KADeviceBackend: the per-head path that prefill and windowed
# decode take, and the fused single-query decode that reads the cache rows in
# place with the GQA mapping (query head h reads KV head div(h-1, group)+1).
#
# Math follows the CPU reference: Float32 storage, a query row sees the causal
# key prefix (key index <= K-L+t) and, with a window, only the last `window` of
# those keys. Masked keys are left out of the max, the sum and the values sum,
# so they contribute exact zeros and their storage is never read.
#
# The softmax is fused into the values kernel. The per-head loop therefore
# runs two launches per head (scores, then softmax and values), and there is
# no separate softmax op for this backend.
#
# Only f32 caches are read. A quantized cache (GESSO_KV_FORMAT=q8) has no KA
# reader and is refused with ERR_INVALID_PLAN rather than dequantized on the
# host.

import Gesso.Inference: _attention_heads!, _attention_decode_device!

# Decode splits the keys into chunks of this many rows, one workgroup per
# (query head, chunk). A chunk is a single pass: its max and sum fold through
# local memory and the chunk partials are merged by a combine kernel. Local
# memory takes static sizes, so the head dimension is capped at _KA_ATTN_DMAX
# and wider heads decline to the per-head path.
const _KA_ATTN_TILE = 512
const _KA_ATTN_DMAX = 512

function _ka_require_dense_kv(::Any)
    return nothing
end
function _ka_require_dense_kv(::Gesso.Inference.QuantKV)
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "a quantized KV cache has no KernelAbstractions reader; the AMDGPU " *
            "backend stores f32 caches, so set GESSO_KV_FORMAT=f32";
            requested_backend=:amdgpu,
            kv_format=:quantized,
        ),
    )
end

# Scores of one query head: sc[t, j] = q[t,hh,:] . k[j,hh,:] / sqrt(d).
# Masked entries are computed too and ignored by the softmax, which keeps this
# kernel branch-free.
@kernel function _ka_attn_scores_kernel!(
    sc,
    @Const(q),
    @Const(k),
    hh::Int,
    scale::Float32,
    L::Int,
    K::Int,
    d::Int,
)
    i = @index(Global, Linear)
    if i <= L * K
        t = (i - 1) % L + 1
        j = (i - 1) ÷ L + 1
        acc = 0.0f0
        for c in 1:d
            acc += q[t, hh, c] * k[j, hh, c]
        end
        sc[t, j] = acc * scale
    end
end

# One workgroup per query row. Max and sum fold through local memory as trees
# with uniform trip counts. Values are read only in the visible range, so a
# masked value never reaches the sum even when it is not finite.
@kernel function _ka_attn_softmax_values_kernel!(
    attn,
    probs,
    @Const(sc),
    @Const(v),
    hh::Int,
    L::Int,
    K::Int,
    d::Int,
    window::Int,
)
    wg = @uniform @groupsize()[1]
    red = @localmem Float32 (_KA_MAX_WG,)
    tot = @localmem Float32 (_KA_MAX_WG,)
    t = @index(Group, Linear)
    lane = @index(Local, Linear)
    m = -Inf32
    j = (window > 0 ? max(1, K - L + t - window + 1) : 1) + lane - 1
    while j <= K - L + t
        m = max(m, sc[t, j])
        j += wg
    end
    red[lane] = m
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for st in 1:steps
        lane = @index(Local, Linear)
        half = wg >> st
        if lane <= half
            red[lane] = max(red[lane], red[lane+half])
        end
        @synchronize
    end

    t = @index(Group, Linear)
    lane = @index(Local, Linear)
    mx = red[1]
    s = 0.0f0
    j = (window > 0 ? max(1, K - L + t - window + 1) : 1) + lane - 1
    while j <= K - L + t
        p = exp(sc[t, j] - mx)
        probs[t, j] = p
        s += p
        j += wg
    end
    tot[lane] = s
    @synchronize
    for st in 1:steps
        lane = @index(Local, Linear)
        half = wg >> st
        if lane <= half
            tot[lane] += tot[lane+half]
        end
        @synchronize
    end

    t = @index(Group, Linear)
    lane = @index(Local, Linear)
    inv = 1.0f0 / tot[1]
    jd = lane
    while jd <= d
        acc = 0.0f0
        for jj in (window > 0 ? max(1, K - L + t - window + 1) : 1):(K - L + t)
            acc += probs[t, jj] * v[jj, hh, jd]
        end
        attn[t, hh, jd] = acc * inv
        jd += wg
    end
end

# Decode split indices. One workgroup per (query head, chunk); group index g
# runs over heads then chunks, so the chunk is (g-1)%nsplit and the head is
# (g-1)÷nsplit. The arithmetic is written out in each kernel: a call into a
# Gesso helper from a kernel does not compile for AMDGPU (dynamic getproperty
# on the device), while the same expression inline does.
# Single pass per chunk. Scores land in `sp`, the chunk max folds through `red`,
# exp(score - max) and the chunk sum fold through `tot`, then the values are
# accumulated into `acc`. The chunk partials (max, sum, weighted sum) are left
# in `part` for the combine kernel.
#
# Group and lane indices are read again after each barrier rather than carried
# across it, and every loop around a barrier has a uniform trip count.
@kernel function _ka_attn_split_kernel!(
    part,
    @Const(q),
    @Const(kc),
    @Const(vc),
    group::Int,
    K::Int,
    scale::Float32,
    d::Int,
    nsplit::Int,
)
    wg = @uniform @groupsize()[1]
    sq = @localmem Float32 (_KA_ATTN_DMAX,)
    acc = @localmem Float32 (_KA_ATTN_DMAX,)
    sp = @localmem Float32 (_KA_ATTN_TILE,)
    red = @localmem Float32 (_KA_MAX_WG,)
    tot = @localmem Float32 (_KA_MAX_WG,)
    steps = @uniform trailing_zeros(wg)

    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    j = lane
    while j <= d
        sq[j] = q[1, ((g - 1) ÷ nsplit + 1), j]
        acc[j] = 0.0f0
        j += wg
    end
    @synchronize

    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    lmax = -Inf32
    i = lane
    while i <= max(0, min(_KA_ATTN_TILE, K - ((g - 1) % nsplit) * _KA_ATTN_TILE))
        x = 0.0f0
        for c in 1:d
            x += sq[c] * kc[(((g - 1) % nsplit) * _KA_ATTN_TILE)+i, (div((g - 1) ÷ nsplit, group) + 1), c]
        end
        x *= scale
        sp[i] = x
        lmax = max(lmax, x)
        i += wg
    end
    red[lane] = lmax
    @synchronize
    for st in 1:steps
        lane = @index(Local, Linear)
        half = wg >> st
        if lane <= half
            red[lane] = max(red[lane], red[lane+half])
        end
        @synchronize
    end

    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    m = red[1]
    lsum = 0.0f0
    i = lane
    while i <= max(0, min(_KA_ATTN_TILE, K - ((g - 1) % nsplit) * _KA_ATTN_TILE))
        p = exp(sp[i] - m)
        sp[i] = p
        lsum += p
        i += wg
    end
    tot[lane] = lsum
    @synchronize
    for st in 1:steps
        lane = @index(Local, Linear)
        half = wg >> st
        if lane <= half
            tot[lane] += tot[lane+half]
        end
        @synchronize
    end

    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    j = lane
    while j <= d
        piece = 0.0f0
        for i in 1:max(0, min(_KA_ATTN_TILE, K - ((g - 1) % nsplit) * _KA_ATTN_TILE))
            piece += sp[i] * vc[(((g - 1) % nsplit) * _KA_ATTN_TILE)+i, (div((g - 1) ÷ nsplit, group) + 1), j]
        end
        acc[j] = piece
        j += wg
    end
    @synchronize

    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    j = lane
    while j <= d
        part[((g - 1) % nsplit + 1), ((g - 1) ÷ nsplit + 1), 2+j] = acc[j]
        j += wg
    end
    if lane == 1
        part[((g - 1) % nsplit + 1), ((g - 1) ÷ nsplit + 1), 1] = red[1]
        part[((g - 1) % nsplit + 1), ((g - 1) ÷ nsplit + 1), 2] = tot[1]
    end
end

# Merges the chunk partials of one head with the rescale exp(m_s - M).
@kernel function _ka_attn_combine_kernel!(attn, @Const(part), nsplit::Int, d::Int)
    wg = @uniform @groupsize()[1]
    h = @index(Group, Linear)
    lane = @index(Local, Linear)
    M = -Inf32
    for s in 1:nsplit
        M = max(M, part[s, h, 1])
    end
    Lsum = 0.0f0
    for s in 1:nsplit
        Lsum += exp(part[s, h, 1] - M) * part[s, h, 2]
    end
    inv = 1.0f0 / Lsum
    j = lane
    while j <= d
        num = 0.0f0
        for s in 1:nsplit
            num += exp(part[s, h, 1] - M) * part[s, h, 2+j]
        end
        attn[1, h, j] = num * inv
        j += wg
    end
end

function _ka_attention_scores!(ka, sc, q, k, hh::Int, d_head::Int, L::Int, K::Int; wg::Int=64)
    _ka_check_wg(wg)
    scale = Float32(1 / sqrt(d_head))
    _ka_attn_scores_kernel!(ka, wg)(
        sc,
        q,
        k,
        hh,
        scale,
        L,
        K,
        d_head;
        ndrange=L * K,
    )
    return sc
end

function _ka_attention_heads!(
    ka,
    attn,
    sc,
    probs,
    q,
    k,
    v,
    n_heads::Int,
    d_head::Int,
    L::Int,
    K::Int;
    window::Int=0,
    wg::Int=64,
)
    _ka_check_wg(wg)
    _ka_require_dense_kv(k)
    _ka_require_dense_kv(v)
    L == 0 && return attn
    1 <= L <= K || throw(
        ArgumentError("attention needs 1 <= L <= K, got L=$L K=$K"),
    )
    kern_m = _ka_attn_softmax_values_kernel!(ka, wg)
    for hh in 1:n_heads
        _ka_attention_scores!(ka, sc, q, k, hh, d_head, L, K; wg)
        kern_m(attn, probs, sc, v, hh, L, K, d_head, window; ndrange=L * wg)
    end
    return attn
end

function _ka_attention_decode!(
    ka,
    attn,
    q,
    kc,
    vc,
    n_heads::Int,
    d_head::Int,
    group::Int,
    K::Int;
    wg::Int=64,
)
    _ka_check_wg(wg)
    _ka_require_dense_kv(kc)
    _ka_require_dense_kv(vc)
    K >= 1 ||
        throw(ArgumentError("decode attention needs at least one cache row, got K=$K"))
    d_head <= _KA_ATTN_DMAX || return false
    nsplit = cld(K, _KA_ATTN_TILE)
    part = KA.zeros(ka, Float32, nsplit, n_heads, d_head + 2)
    scale = Float32(1 / sqrt(d_head))
    _ka_attn_split_kernel!(ka, wg)(
        part,
        q,
        kc,
        vc,
        group,
        K,
        scale,
        d_head,
        nsplit;
        ndrange=n_heads * nsplit * wg,
    )
    _ka_attn_combine_kernel!(ka, wg)(attn, part, nsplit, d_head; ndrange=n_heads * wg)
    return true
end

function _attention_heads!(
    backend::KADeviceBackend,
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
    workload;
    window::Int=0,
)
    return _ka_attention_heads!(
        backend.ka,
        attn,
        scores.storage,
        scores_out.storage,
        q,
        k,
        v,
        n_heads,
        d_head,
        L,
        K;
        window,
    )
end

_attention_decode_device!(
    backend::KADeviceBackend,
    attn::AbstractArray,
    q::AbstractArray,
    kc::AbstractArray,
    vc::AbstractArray,
    n_heads::Int,
    d_head::Int,
    group::Int,
    K::Int,
) = _ka_attention_decode!(backend.ka, attn, q, kc, vc, n_heads, d_head, group, K)
