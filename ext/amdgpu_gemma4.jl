# Gemma4 device blocks: the attention and FFN layers of one transformer block,
# plus the per-model hooks the core calls around them.
#
# The block runs on the device except the parallel MoE FFN, whose router and
# expert bank are host-resident GGUF slices (see amdgpu_host.jl). Head and row
# RMS norms share one segment kernel: a segment is a contiguous run of `hd`
# values in one row, and a full-width row norm is the single-segment case.

const _G4A_SCORE_BYTES = Ref(256 * 2^20)

const _G4A_HOST_BT_FIELDS = (
    :ffn_rms,
    :wgate,
    :wup,
    :wdown,
    :ffn_post_rms_1,
    :wrouter,
    :wrouter_scale,
    :ffn_pre_rms_2,
    :wexp_gate_up,
    :wexp_down,
    :wexp_down_scale,
    :ffn_post_rms_2,
    :ffn_post_rms,
)

_ka_g4_launch(ka, kernel, wg::Int, total::Int, args...) = begin
    _ka_check_wg(wg)
    total > 0 && kernel(ka, wg)(args..., total; ndrange=cld(total, wg) * wg)
    nothing
end

# --- kernels ------------------------------------------------------------------

# Segment RMS: segment g is row t = (g-1) % N + 1, head hh = (g-1) ÷ N + 1,
# columns o+1:o+hd with o = (hh-1)*hd. The sum of squares folds through local
# memory; the scaled write re-reads its index after the last barrier.
@kernel function _ka_g4_seg_norm_kernel!(dst, src, @Const(w), eps::Float32, N::Int, hd::Int, use_w::Bool)
    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    wg = @uniform @groupsize()[1]
    red = @localmem Float32 (_KA_MAX_WG,)
    t = (g - 1) % N + 1
    o = ((g - 1) ÷ N) * hd
    ss = 0.0f0
    j = lane
    while j <= hd
        ss += src[t, o+j] * src[t, o+j]
        j += wg
    end
    red[lane] = ss
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for k in 1:steps
        lane = @index(Local, Linear)
        s = wg >> k
        if lane <= s
            red[lane] += red[lane+s]
        end
        @synchronize
    end
    g = @index(Group, Linear)
    wg = @uniform @groupsize()[1]
    t = (g - 1) % N + 1
    o = ((g - 1) ÷ N) * hd
    r = 1.0f0 / sqrt(red[1] / Float32(hd) + eps)
    j = @index(Local, Linear)
    while j <= hd
        wj = use_w ? w[j] : 1.0f0
        dst[t, o+j] = src[t, o+j] * r * wj
        j += wg
    end
end

function _ka_g4_seg_norm!(ka, dst, src, w, eps::Real, hd::Int, N::Int, nheads::Int, use_w::Bool; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    (N > 0 && nheads > 0 && hd > 0) || return dst
    _ka_g4_seg_norm_kernel!(ka, wg)(
        dst, src, w, Float32(eps), N, hd, use_w;
        ndrange=wg * N * nheads,
    )
    return dst
end

# Rotate pairs (p, p+half) of every head in place. The cos/sin tables come from
# the host in Float64 then Float32, the same rounding the CPU rope uses.
@kernel function _ka_g4_rope_kernel!(A, @Const(cs), @Const(sn), N::Int, nheads::Int, hd::Int, half::Int, total::Int)
    i = @index(Global, Linear)
    if i <= total
        t = (i - 1) % N + 1
        p = (i - 1) ÷ N + 1
        c = cs[t, p]
        s = sn[t, p]
        for h in 1:nheads
            o = (h - 1) * hd
            x0 = A[t, o+p]
            x1 = A[t, o+p+half]
            A[t, o+p] = x0 * c - x1 * s
            A[t, o+p+half] = x0 * s + x1 * c
        end
    end
end

# dst[row0+t-1, h, d] = src[t, (h-1)*hd + d]: heads split into the cache layout.
@kernel function _ka_g4_kv_put_kernel!(dst, @Const(src), row0::Int, N::Int, nheads::Int, hd::Int, total::Int)
    i = @index(Global, Linear)
    if i <= total
        t = (i - 1) % N + 1
        rest = (i - 1) ÷ N
        h = rest % nheads + 1
        d = rest ÷ nheads + 1
        dst[row0+t-1, h, d] = src[t, (h-1)*hd+d]
    end
end

# Scores S[kk, c] for key k0+kk against query column c = (hh-1)*Nt + tl, with
# causal and sliding-window masking. Scale is 1 (Gemma4 folds it into the Q norm).
@kernel function _ka_g4_scores_kernel!(
    S,
    @Const(Q),
    @Const(Kd),
    nh::Int,
    nkv::Int,
    hd::Int,
    qoff::Int,
    win::Int,
    t0::Int,
    Nt::Int,
    k0::Int,
    Kn::Int,
    total::Int,
)
    i = @index(Global, Linear)
    if i <= total
        kk = (i - 1) % Kn + 1
        c = (i - 1) ÷ Kn + 1
        tl = (c - 1) % Nt + 1
        hh = (c - 1) ÷ Nt + 1
        t = t0 + tl - 1
        k = k0 + kk
        r = qoff + t - 1
        lo = win > 0 ? max(1, r - win + 1) : 1
        if k < lo || k > r
            S[kk, c] = -Inf32
        else
            kvh = (hh - 1) ÷ (nh ÷ nkv) + 1
            qo = (hh - 1) * hd
            s = 0.0f0
            for d in 1:hd
                s += Q[t, qo+d] * Kd[k, kvh, d]
            end
            S[kk, c] = s
        end
    end
end

# Column softmax, one workgroup per column. The max and the sum fold through
# separate local arrays; every value that crosses a barrier is re-read.
@kernel function _ka_g4_softmax_cols_kernel!(S, Kn::Int)
    lane = @index(Local, Linear)
    c = @index(Group, Linear)
    wg = @uniform @groupsize()[1]
    redm = @localmem Float32 (_KA_MAX_WG,)
    reds = @localmem Float32 (_KA_MAX_WG,)
    mx = -Inf32
    kk = lane
    while kk <= Kn
        mx = max(mx, S[kk, c])
        kk += wg
    end
    redm[lane] = mx
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for k in 1:steps
        lane = @index(Local, Linear)
        s = wg >> k
        if lane <= s
            redm[lane] = max(redm[lane], redm[lane+s])
        end
        @synchronize
    end
    lane = @index(Local, Linear)
    c = @index(Group, Linear)
    wg = @uniform @groupsize()[1]
    mxv = redm[1]
    sm = 0.0f0
    kk = lane
    while kk <= Kn
        e = exp(S[kk, c] - mxv)
        S[kk, c] = e
        sm += e
        kk += wg
    end
    reds[lane] = sm
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for k in 1:steps
        lane = @index(Local, Linear)
        s = wg >> k
        if lane <= s
            reds[lane] += reds[lane+s]
        end
        @synchronize
    end
    lane = @index(Local, Linear)
    c = @index(Group, Linear)
    wg = @uniform @groupsize()[1]
    inv = 1.0f0 / reds[1]
    kk = lane
    while kk <= Kn
        S[kk, c] = S[kk, c] * inv
        kk += wg
    end
end

# O[t0+tl-1, (hh-1)*hd + d] = sum over the tile's keys of S * V.
@kernel function _ka_g4_pv_kernel!(O, @Const(S), @Const(Vd), nh::Int, nkv::Int, hd::Int, t0::Int, Nt::Int, k0::Int, Kn::Int, total::Int)
    i = @index(Global, Linear)
    if i <= total
        tl = (i - 1) % Nt + 1
        rest = (i - 1) ÷ Nt
        hh = rest % nh + 1
        d = rest ÷ nh + 1
        c = (hh - 1) * Nt + tl
        kvh = (hh - 1) ÷ (nh ÷ nkv) + 1
        acc = 0.0f0
        for kk in 1:Kn
            acc += S[kk, c] * Vd[k0+kk, kvh, d]
        end
        O[t0+tl-1, (hh-1)*hd+d] = acc
    end
end

@kernel function _ka_g4_add_kernel!(dst, @Const(a), @Const(b), total::Int)
    i = @index(Global, Linear)
    if i <= total
        dst[i] = a[i] + b[i]
    end
end

@kernel function _ka_g4_copy_kernel!(dst, @Const(src), total::Int)
    i = @index(Global, Linear)
    if i <= total
        dst[i] = src[i]
    end
end

@kernel function _ka_g4_scale_kernel!(X, s::Float32, total::Int)
    i = @index(Global, Linear)
    if i <= total
        X[i] = X[i] * s
    end
end

@kernel function _ka_g4_softcap_kernel!(X, cap::Float32, total::Int)
    i = @index(Global, Linear)
    if i <= total
        X[i] = cap * tanh(X[i] / cap)
    end
end

# --- launch helpers and host tables -------------------------------------------

_ka_g4_rope_tables(positions, inv_freq, half::Int) = begin
    N = length(positions)
    cs = Matrix{Float32}(undef, N, half)
    sn = Matrix{Float32}(undef, N, half)
    for t in 1:N
        pos = Float64(positions[t])
        for i in 1:half
            theta = pos * inv_freq[i]
            cs[t, i] = Float32(cos(theta))
            sn[t, i] = Float32(sin(theta))
        end
    end
    return cs, sn
end

function _ka_g4_rope!(ka, A, positions, nheads::Int, hd::Int, n_rot::Int, inv_freq; wg::Int=_KA_OP_WG)
    half = n_rot ÷ 2
    N = size(A, 1)
    (half == 0 || N == 0) && return A
    cs, sn = _ka_g4_rope_tables(positions, inv_freq, half)
    total = N * half
    _ka_g4_launch(
        ka, _ka_g4_rope_kernel!, wg, total,
        A, _ka_upload_host(ka, Float32, cs), _ka_upload_host(ka, Float32, sn), N, nheads, hd, half,
    )
    return A
end

function _ka_g4_kv_put!(ka, dst, src, row0::Int, nheads::Int, hd::Int; wg::Int=_KA_OP_WG)
    N = size(src, 1)
    _ka_g4_launch(ka, _ka_g4_kv_put_kernel!, wg, N * nheads * hd, dst, src, row0, N, nheads, hd)
    return dst
end

_ka_g4_tile_keys(qoff::Int, win::Int, t0::Int, t1::Int) = begin
    lo = win > 0 ? max(1, qoff + t0 - 1 - win + 1) : 1
    return lo - 1, qoff + t1 - 1 - lo + 1
end

# Attention over query rows t = 1:Nq at absolute positions qoff..qoff+Nq-1, against
# keys 1:(qoff+Nq-1). The score tile is capped by _G4A_SCORE_BYTES, so a long
# prefill walks the queries in tiles.
function _ka_g4_attention(ka, Q, Kd, Vd, nh::Int, nkv::Int, hd::Int, qoff::Int, win::Int; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    Nq = size(Q, 1)
    O = KA.allocate(ka, Float32, Nq, nh * hd)
    Nq == 0 && return O
    budget = max(1, _G4A_SCORE_BYTES[] ÷ (4 * nh))
    t0 = 1
    while t0 <= Nq
        t1 = Nq
        k0, Kn = _ka_g4_tile_keys(qoff, win, t0, t1)
        if Kn * (t1 - t0 + 1) > budget
            t1 = min(Nq, t0 + max(1, budget ÷ Kn) - 1)
            k0, Kn = _ka_g4_tile_keys(qoff, win, t0, t1)
        end
        Nt = t1 - t0 + 1
        S = KA.allocate(ka, Float32, Kn, Nt * nh)
        _ka_g4_launch(
            ka, _ka_g4_scores_kernel!, wg, Kn * Nt * nh,
            S, Q, Kd, nh, nkv, hd, qoff, win, t0, Nt, k0, Kn,
        )
        _ka_check_wg(wg)
        _ka_g4_softmax_cols_kernel!(ka, wg)(S, Kn; ndrange=Nt * nh * wg)
        _ka_g4_launch(ka, _ka_g4_pv_kernel!, wg, Nt * nh * hd, O, S, Vd, nh, nkv, hd, t0, Nt, k0, Kn)
        t0 = t1 + 1
    end
    return O
end

# --- device-side helpers for the block ----------------------------------------

function _ka_g4_weight(b::KADeviceBackend, p)
    p === nothing && return nothing
    p.storage === nothing && return nothing
    return _ka_device_storage!(:gemma4, b, p)
end

function _ka_g4_proj(w)
    w isa ProjectionWeight && return w
    return ProjectionWeight(w.shape, w.storage)
end

function _ka_g4_gemm(b::KADeviceBackend, x, w, wl)
    wp = _ka_g4_proj(w)
    N = size(x, 1)
    nout = wp.storage isa AmdDenseQuant ? wp.storage.nrows : size(wp.storage, 1)
    dst = KA.allocate(b.ka, Float32, N, nout)
    matmul!(b, Activation((N, nout), dst), Activation((N, size(x, 2)), x), wp, wl)
    return dst
end

function _ka_g4_seg_norm(b::KADeviceBackend, src, nheads::Int, hd::Int, w, eps::Real)
    dst = KA.allocate(b.ka, Float32, size(src))
    N = size(src, 1)
    if w === nothing
        ones_w = _ka_upload_host(b.ka, Float32, ones(Float32, hd))
        return _ka_g4_seg_norm!(b.ka, dst, src, ones_w, eps, hd, N, nheads, false)
    end
    return _ka_g4_seg_norm!(b.ka, dst, src, w, eps, hd, N, nheads, true)
end

function _ka_g4_geom(bt, tensors, eps)
    hd = Int(bt.d_head_layer)
    nkv = Int(bt.n_kv_heads_layer)
    nh = Int(tensors.spec.n_heads)
    win = Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[] && bt.attn_window > 0 ? Int(bt.attn_window) : 0
    return hd, nkv, nh, win, Float32(eps)
end

function _ka_g4_qkv(b::KADeviceBackend, X, bt, positions, wl, eps_f::Float32, nh::Int, nkv::Int, hd::Int)
    d = size(X, 2)
    xn = _ka_g4_seg_norm(b, X, 1, d, _ka_g4_weight(b, bt.attn_rms), eps_f)
    Q = _ka_g4_gemm(b, xn, bt.wq, wl)
    K = _ka_g4_gemm(b, xn, bt.wk, wl)
    V = if haskey(bt, :wv) && bt.wv !== nothing
        _ka_g4_gemm(b, xn, bt.wv, wl)
    else
        Vk = KA.allocate(b.ka, Float32, size(K))
        copyto!(Vk, K)
        Vk
    end
    Q = _ka_g4_seg_norm(b, Q, nh, hd, _ka_g4_weight(b, bt.wq_norm), eps_f)
    K = _ka_g4_seg_norm(b, K, nkv, hd, _ka_g4_weight(b, bt.wk_norm), eps_f)
    V = _ka_g4_seg_norm(b, V, nkv, hd, nothing, eps_f)
    rl = bt.rope_layer
    n_rot = rl.rotary_dim == 0 ? hd : rl.rotary_dim
    Gesso.Inference._apply_rope_neox!(b, Q, positions, nh, hd, n_rot, rl.inv_freq)
    Gesso.Inference._apply_rope_neox!(b, K, positions, nkv, hd, n_rot, rl.inv_freq)
    return Q, K, V
end

# The parallel FFN runs on the host (router and expert bank are GGUF slices).
function _ka_g4_moe(b::KADeviceBackend, attn, bt, eps_f::Float32)
    xh = Array(attn)
    hb = _ka_host_bt(bt, _G4A_HOST_BT_FIELDS)
    mlp = Gesso.Inference._gemma4_parallel_moe_ffn!(
        Gesso.CPUBackend(), Matrix{Float32}(xh), hb, eps_f,
    )
    return _ka_upload_host(b.ka, Float32, Matrix{Float32}(mlp))
end

function _ka_g4_finish!(b::KADeviceBackend, X, O, bt, wl, eps_f::Float32)
    A = _ka_g4_gemm(b, O, bt.wo, wl)
    A = _ka_g4_seg_norm(b, A, 1, size(A, 2), _ka_g4_weight(b, bt.attn_post_rms), eps_f)
    attn_out = KA.allocate(b.ka, Float32, size(X))
    _ka_g4_launch(b.ka, _ka_g4_add_kernel!, _KA_OP_WG, length(X), attn_out, A, X)
    mlp = _ka_g4_moe(b, attn_out, bt, eps_f)
    cur = KA.allocate(b.ka, Float32, size(X))
    _ka_g4_launch(b.ka, _ka_g4_add_kernel!, _KA_OP_WG, length(X), cur, mlp, attn_out)
    if haskey(bt, :layer_output_scale) && bt.layer_output_scale !== nothing
        sv = _ka_host_storage(bt.layer_output_scale.storage)
        s = Float32(sv isa AbstractArray ? sv[1] : sv)
        _ka_g4_launch(b.ka, _ka_g4_scale_kernel!, _KA_OP_WG, length(cur), cur, s)
    end
    _ka_g4_launch(b.ka, _ka_g4_copy_kernel!, _KA_OP_WG, length(X), X, cur)
    return X
end

# --- core hooks ---------------------------------------------------------------

function Gesso.Inference._apply_rope_neox!(
    b::KADeviceBackend,
    A::AbstractMatrix{Float32},
    positions,
    nheads::Int,
    hd::Int,
    n_rot::Int,
    inv_freq::Vector{Float64},
)
    _ka_device_storage!(:rope_neox, b, (; storage=A))
    return _ka_g4_rope!(b.ka, A, positions, nheads, hd, n_rot, inv_freq)
end

function Gesso.Inference._apply_embedding_scale!(b::KADeviceBackend, h, tensors)
    hs = _ka_device_storage!(:embedding_scale, b, h)
    s = eltype(hs)(tensors.spec.embedding_scale)
    _ka_g4_launch(b.ka, _ka_g4_scale_kernel!, _KA_OP_WG, length(hs), hs, s)
    return h
end

function Gesso.Inference._apply_logit_softcap!(b::KADeviceBackend, logits, tensors)
    if Gesso.Inference._GEMMA4_SOFTCAP_ENABLED[] && tensors.spec.final_logit_softcap > 0
        ls = _ka_device_storage!(:logit_softcap, b, logits)
        cap = Float32(tensors.spec.final_logit_softcap)
        _ka_g4_launch(b.ka, _ka_g4_softcap_kernel!, _KA_OP_WG, length(ls), ls, cap)
    end
    return logits
end

function Gesso.Inference._gemma4_prefill_block!(
    b::KADeviceBackend,
    h,
    bt,
    tensors,
    positions,
    wl,
    eps,
    zeros_like;
    scratch=nothing,
)
    X = _ka_device_storage!(:gemma4_block, b, (; storage=_ka_act_storage(h)))
    hd, nkv, nh, win, eps_f = _ka_g4_geom(bt, tensors, eps)
    Q, K, V = _ka_g4_qkv(b, X, bt, positions, wl, eps_f, nh, nkv, hd)
    Kh = KA.allocate(b.ka, Float32, size(K, 1), nkv, hd)
    Vh = KA.allocate(b.ka, Float32, size(V, 1), nkv, hd)
    _ka_g4_kv_put!(b.ka, Kh, K, 1, nkv, hd)
    _ka_g4_kv_put!(b.ka, Vh, V, 1, nkv, hd)
    O = _ka_g4_attention(b.ka, Q, Kh, Vh, nh, nkv, hd, 1, win)
    _ka_g4_finish!(b, X, O, bt, wl, eps_f)
    return h
end

function Gesso.Inference._gemma4_generate_block!(
    b::KADeviceBackend,
    hp,
    kc_layer,
    vc_layer,
    bt,
    tensors,
    s0,
    s1,
    wl,
    eps,
    zeros_like;
    scratch=nothing,
)
    X = _ka_device_storage!(:gemma4_block, b, (; storage=_ka_act_storage(hp)))
    hd, nkv, nh, win, eps_f = _ka_g4_geom(bt, tensors, eps)
    _ka_require_dense_kv(kc_layer.storage)
    _ka_require_dense_kv(vc_layer.storage)
    kc = _ka_device_storage!(:gemma4_block, b, kc_layer)
    vc = _ka_device_storage!(:gemma4_block, b, vc_layer)
    positions = collect((s0 - 1):(s1 - 1))
    Q, K, V = _ka_g4_qkv(b, X, bt, positions, wl, eps_f, nh, nkv, hd)
    _ka_g4_kv_put!(b.ka, kc, K, s0, nkv, hd)
    _ka_g4_kv_put!(b.ka, vc, V, s0, nkv, hd)
    O = _ka_g4_attention(b.ka, Q, kc, vc, nh, nkv, hd, s0, win)
    _ka_g4_finish!(b, X, O, bt, wl, eps_f)
    return hp
end
