# Gemma 4 on Lava. Projections, norms, NeoX RoPE, KV writes and attention run on
# device. The dense FFN and the routed experts run on host through the CPU
# routine, as the LFM2 MoE route does, so their numerics are the CPU oracle's.

const _G4L_WG = (64,)

# Row-wise RMSNorm, one thread per row. use_w = 0 is the weightless norm.
@kernel cpu=false function _g4l_rms_rows_kernel!(Y, X, w, eps::Float32, N::Int, E::Int, use_w::Int)
    t = @index(Global)
    if t <= N
        ss = 0.0f0
        for i in 1:E
            ss += X[t, i] * X[t, i]
        end
        r = 1.0f0 / sqrt(ss / E + eps)
        for i in 1:E
            wi = use_w == 1 ? w[i] : 1.0f0
            Y[t, i] = X[t, i] * r * wi
        end
    end
end

# Per-head RMSNorm in place, one thread per (row, head).
@kernel cpu=false function _g4l_head_norm_kernel!(A, w, eps::Float32, N::Int, nheads::Int, hd::Int, use_w::Int)
    i = @index(Global)
    if i <= N * nheads
        t = (i - 1) % N + 1
        h = (i - 1) ÷ N + 1
        o = (h - 1) * hd
        ss = 0.0f0
        for d in 1:hd
            ss += A[t, o + d] * A[t, o + d]
        end
        r = 1.0f0 / sqrt(ss / hd + eps)
        for d in 1:hd
            wd = use_w == 1 ? w[d] : 1.0f0
            A[t, o + d] = A[t, o + d] * r * wd
        end
    end
end

# NeoX pairs (p, p + half) per head; cos and sin come from host tables so the
# angle is the CPU's Float64 product rounded once.
@kernel cpu=false function _g4l_rope_kernel!(A, cs, sn, N::Int, nheads::Int, hd::Int, half::Int)
    i = @index(Global)
    if i <= N * half
        t = (i - 1) % N + 1
        p = (i - 1) ÷ N + 1
        c = cs[t, p]
        s = sn[t, p]
        for h in 1:nheads
            o = (h - 1) * hd
            x0 = A[t, o + p]
            x1 = A[t, o + p + half]
            A[t, o + p] = x0 * c - x1 * s
            A[t, o + p + half] = x0 * s + x1 * c
        end
    end
end

# (N, nheads*hd) to (N, nheads, hd) so the KV writers see a 3D source.
@kernel cpu=false function _g4l_heads_kernel!(Kh, K, N::Int, nheads::Int, hd::Int)
    i = @index(Global)
    if i <= N * nheads * hd
        t = (i - 1) % N + 1
        r = (i - 1) ÷ N
        h = r % nheads + 1
        d = r ÷ nheads + 1
        Kh[t, h, d] = K[t, (h - 1) * hd + d]
    end
end

# Plain f32 cache write: dst[row0 + t - 1, h, d] = src[t, h, d].
@kernel cpu=false function _g4l_kv_put_kernel!(dst, src, row0::Int, N::Int, nheads::Int, hd::Int)
    i = @index(Global)
    if i <= N * nheads * hd
        t = (i - 1) % N + 1
        r = (i - 1) ÷ N
        h = r % nheads + 1
        d = r ÷ nheads + 1
        dst[row0 + t - 1, h, d] = src[t, h, d]
    end
end

# Causal attention with an optional sliding window and scale 1.0. Query row t
# sits at 1-based cache row qoff + t - 1 and sees rows lo:r, with
# lo = max(1, r - win + 1) when win > 0. Three passes (max, sum, weighted value)
# keep the score row out of per-thread storage.
@kernel cpu=false function _g4l_attn_kernel!(O, Q, Kd, Vd, nh::Int, nkv::Int, hd::Int, qoff::Int, win::Int, Nq::Int)
    i = @index(Global)
    if i <= Nq * nh
        t = (i - 1) % Nq + 1
        hh = (i - 1) ÷ Nq + 1
        rep = nh ÷ nkv
        kvh = (hh - 1) ÷ rep + 1
        qo = (hh - 1) * hd
        r = qoff + t - 1
        lo = win > 0 ? max(1, r - win + 1) : 1
        mx = -3.0f38
        for k in lo:r
            s = 0.0f0
            for d in 1:hd
                s += Q[t, qo + d] * Kd[k, kvh, d]
            end
            mx = max(mx, s)
        end
        den = 0.0f0
        for k in lo:r
            s = 0.0f0
            for d in 1:hd
                s += Q[t, qo + d] * Kd[k, kvh, d]
            end
            den += exp(s - mx)
        end
        inv = 1.0f0 / den
        for d in 1:hd
            acc = 0.0f0
            for k in lo:r
                s = 0.0f0
                for dd in 1:hd
                    s += Q[t, qo + dd] * Kd[k, kvh, dd]
                end
                acc += (exp(s - mx) * inv) * Vd[k, kvh, d]
            end
            O[t, qo + d] = acc
        end
    end
end

@kernel cpu=false function _g4l_add2_kernel!(dst, a, b)
    i = @index(Global)
    if i <= length(dst)
        dst[i] = a[i] + b[i]
    end
end

@kernel cpu=false function _g4l_copy_kernel!(dst, src)
    i = @index(Global)
    if i <= length(dst)
        dst[i] = src[i]
    end
end

@kernel cpu=false function _g4l_scale_kernel!(X, s::Float32)
    i = @index(Global)
    if i <= length(X)
        X[i] = X[i] * s
    end
end

@kernel cpu=false function _g4l_softcap_kernel!(X, cap::Float32)
    i = @index(Global)
    if i <= length(X)
        X[i] = cap * tanh(X[i] / cap)
    end
end

# --- launch helpers -----------------------------------------------------------

_g4l_new(dims...) = Lava.LavaArray{Float32}(undef, dims...)

function _g4l_up(h::AbstractArray{Float32})
    d = _g4l_new(size(h)...)
    copyto!(d, h)
    return d
end

function _g4l_launch!(kern, arr, n::Int, args...)
    n == 0 && return nothing
    kern(KA.get_backend(arr))(args...; ndrange=n, workgroupsize=_G4L_WG)
    return nothing
end

function _g4l_copy!(dst, src)
    _g4l_launch!(_g4l_copy_kernel!, dst, length(dst), dst, src)
    return dst
end

_g4l_wstore(w) = w isa FrozenParameter ? w.storage : w

function _g4l_rms_rows!(Y, X, w, eps_f::Float32)
    N, E = size(X)
    ws = _g4l_wstore(w)
    use_w = ws === nothing ? 0 : 1
    wv = ws === nothing ? X : ws
    _g4l_launch!(_g4l_rms_rows_kernel!, X, N, Y, X, wv, eps_f, N, E, use_w)
    return Y
end

function _g4l_head_norm!(A, nheads::Int, hd::Int, w, eps_f::Float32)
    N = size(A, 1)
    ws = _g4l_wstore(w)
    use_w = ws === nothing ? 0 : 1
    wv = ws === nothing ? A : ws
    _g4l_launch!(_g4l_head_norm_kernel!, A, N * nheads, A, wv, eps_f, N, nheads, hd, use_w)
    return A
end

function _g4l_rope!(A, positions, nheads::Int, hd::Int, n_rot::Int, inv_freq)
    N = size(A, 1)
    half = n_rot ÷ 2
    cs = Matrix{Float32}(undef, N, half)
    sn = Matrix{Float32}(undef, N, half)
    @inbounds for t in 1:N, i in 1:half
        theta = Float64(positions[t]) * inv_freq[i]
        cs[t, i] = Float32(cos(theta))
        sn[t, i] = Float32(sin(theta))
    end
    _g4l_launch!(_g4l_rope_kernel!, A, N * half, A, _g4l_up(cs), _g4l_up(sn), N, nheads, hd, half)
    return A
end

function _g4l_heads(K, N::Int, nheads::Int, hd::Int)
    Kh = _g4l_new(N, nheads, hd)
    _g4l_launch!(_g4l_heads_kernel!, Kh, N * nheads * hd, Kh, K, N, nheads, hd)
    return Kh
end

function _g4l_kv_write!(dst::AbstractArray, src, row0::Int)
    N, nheads, hd = size(src)
    _g4l_launch!(_g4l_kv_put_kernel!, src, N * nheads * hd, dst, src, row0, N, nheads, hd)
    return nothing
end

_g4l_kv_write!(dst::_LavaQ8, src, row0::Int) = Gesso.Inference._kv_write_rows!(dst, src, row0)

function _g4l_attention(Q, Kd, Vd, nh::Int, nkv::Int, hd::Int, qoff::Int, win::Int)
    Nq = size(Q, 1)
    O = _g4l_new(Nq, nh * hd)
    _g4l_launch!(_g4l_attn_kernel!, O, Nq * nh, O, Q, Kd, Vd, nh, nkv, hd, qoff, win, Nq)
    return O
end

function _g4l_mm!(dst, x, w)
    _lava_matmul!((; storage=dst), (; storage=x), w)
    return dst
end

# --- host route for the parallel dense + routed FFN ----------------------------

# Device quant bytes back to the GGUF layout the CPU reader expects. Q8_0 is
# inverted from the repack in lava_quant.jl: the f16 scale was widened to f32
# exactly, so narrowing it again is lossless.
function _g4l_quant_host_uncached(q::LavaBlockQuant)
    blck = Int64(Gesso.Inference._GGML_KNOWN_BLOCKS[Int32(q.ggml_type)][1])
    return Gesso.Inference.QuantMatrix(
        Array(q.data), 0, q.ggml_type, q.ne0, q.nrows, q.row_bytes, blck,
    )
end

function _g4l_quant_host_uncached(q::LavaQ8Quant)
    nblk = q.nrows * (q.ne0 ÷ 32)
    qs = Array(q.qs)
    dv = Array(q.d)
    bytes = Vector{UInt8}(undef, 34 * nblk)
    @inbounds for b in 0:(nblk - 1)
        o = 34 * b
        bits = reinterpret(UInt16, Float16(dv[b + 1]))
        bytes[o + 1] = UInt8(bits & 0xff)
        bytes[o + 2] = UInt8(bits >> 8)
        for w in 0:7
            v = qs[8 * b + w + 1]
            p = o + 2 + 4 * w
            bytes[p + 1] = UInt8(v & 0xff)
            bytes[p + 2] = UInt8((v >> 8) & 0xff)
            bytes[p + 3] = UInt8((v >> 16) & 0xff)
            bytes[p + 4] = UInt8((v >> 24) & 0xff)
        end
    end
    return Gesso.Inference.QuantMatrix(
        bytes, 0, Int32(Gesso.Inference.GGML_TYPE_Q8_0), q.ne0, q.nrows, 34 * (q.ne0 ÷ 32), 32,
    )
end

const _G4L_HOST_QUANT = Dict{UInt, Tuple{WeakRef, Any}}()
const _G4L_HOST_QUANT_LOCK = ReentrantLock()

# The quant structs are immutable, so a WeakRef to one holds a fresh box that
# the next GC frees, and every call after it pulls the whole tensor down
# again. Anchor the entry on the mutable device buffer the quant owns instead.
_g4l_quant_anchor(q::LavaBlockQuant) = q.data
_g4l_quant_anchor(q::LavaQ8Quant) = q.qs

function _g4l_quant_host(q)
    a = _g4l_quant_anchor(q)
    k = objectid(a)
    lock(_G4L_HOST_QUANT_LOCK) do
        e = get(_G4L_HOST_QUANT, k, nothing)
        e !== nothing && e[1].value === a && return e[2]
        filter!(p -> p.second[1].value !== nothing, _G4L_HOST_QUANT)
        h = _g4l_quant_host_uncached(q)
        _G4L_HOST_QUANT[k] = (WeakRef(a), h)
        return h
    end
end

_g4l_host_storage(x) = x
_g4l_host_storage(x::Lava.LavaArray) = _lh_host_weight(x)
_g4l_host_storage(q::Union{LavaBlockQuant, LavaQ8Quant}) = _g4l_quant_host(q)

_g4l_host_wrap(::Nothing) = nothing
_g4l_host_wrap(w::ProjectionWeight) = ProjectionWeight(w.shape, _g4l_host_storage(w.storage))
_g4l_host_wrap(w::FrozenParameter) = FrozenParameter(w.shape, _g4l_host_storage(w.storage))
_g4l_host_wrap(w::Gesso.ExpertWeight) = Gesso.ExpertWeight(w.shape, _g4l_host_storage(w.storage))
_g4l_host_wrap(w) = w

const _G4L_HOST_BT_FIELDS = (
    :ffn_rms, :wgate, :wup, :wdown, :ffn_post_rms_1,
    :wrouter, :wrouter_scale, :ffn_pre_rms_2,
    :wexp_gate_up, :wexp_down, :wexp_down_scale,
    :ffn_post_rms_2, :ffn_post_rms,
)

function _g4l_host_bt(bt)
    pairs_ = (f => _g4l_host_wrap(getproperty(bt, f)) for f in _G4L_HOST_BT_FIELDS if haskey(bt, f))
    return merge(bt, NamedTuple(pairs_))
end

# Attention-free FFN block: the device attention output comes down, the CPU
# routine runs on host copies, and the result goes back up.
function _g4l_moe(attn::Lava.LavaArray, bt, eps_f::Float32)
    xh = _lh_host(attn)
    mlp = Gesso.Inference._gemma4_parallel_moe_ffn!(
        Gesso.CPUBackend(), xh, _g4l_host_bt(bt), eps_f;
        scratch=Gesso.Inference._Gemma4Scratch(),
    )
    out = _g4l_new(size(xh)...)
    copyto!(out, Matrix{Float32}(mlp))
    return out
end

# --- hooks --------------------------------------------------------------------

function Gesso.Inference._apply_rope_neox!(
    ::LavaBackend, A, positions, nheads::Int, hd::Int, n_rot::Int, inv_freq::Vector{Float64},
)
    _g4l_rope!(A, positions, nheads, hd, n_rot, inv_freq)
    return A
end

function Gesso.Inference._apply_embedding_scale!(::LavaBackend, h, tensors)
    s = Float32(tensors.spec.embedding_scale)
    _g4l_launch!(_g4l_scale_kernel!, h.storage, length(h.storage), h.storage, s)
    return h
end

function Gesso.Inference._apply_logit_softcap!(::LavaBackend, logits, tensors)
    if Gesso.Inference._GEMMA4_SOFTCAP_ENABLED[] && tensors.spec.final_logit_softcap > 0
        cap = Float32(tensors.spec.final_logit_softcap)
        _g4l_launch!(_g4l_softcap_kernel!, logits.storage, length(logits.storage), logits.storage, cap)
    end
    return logits
end

function Gesso.Inference._gemma4_v_norm!(::LavaBackend, vh, eps; nheads::Int=1, hd::Int=size(vh, 2))
    _g4l_head_norm!(vh, nheads, hd, nothing, Float32(eps))
    return vh
end

function Gesso.Inference._gemma4_k_only_v!(::LavaBackend, k)
    return _g4l_copy!(_g4l_new(size(k)...), k)
end

function Gesso.Inference._gemma4_layer_output_scale!(::LavaBackend, out, scale_vec)
    s = scale_vec isa AbstractArray ? _lh_host(scale_vec)[1] : scale_vec
    _g4l_launch!(_g4l_scale_kernel!, out, length(out), out, Float32(s))
    return out
end

function Gesso.Inference._gemma4_parallel_moe_ffn!(::LavaBackend, attn, bt, eps::Real; scratch=nothing)
    X = attn isa Activation ? attn.storage : attn
    return _g4l_moe(X, bt, Float32(eps))
end

# --- block prefill and generate ------------------------------------------------

# Norms, projections, RoPE and the new K and V rows for N tokens.
function _g4l_qkv(X, bt, nh::Int, nkv::Int, hd::Int, positions, eps_f::Float32)
    N, E = size(X)
    qdim = nh * hd
    kvdim = nkv * hd
    xn = _g4l_new(N, E)
    _g4l_rms_rows!(xn, X, bt.attn_rms, eps_f)
    Q = _g4l_mm!(_g4l_new(N, qdim), xn, bt.wq)
    K = _g4l_mm!(_g4l_new(N, kvdim), xn, bt.wk)
    if haskey(bt, :wv) && bt.wv !== nothing
        V = _g4l_mm!(_g4l_new(N, kvdim), xn, bt.wv)
    else
        V = _g4l_copy!(_g4l_new(N, kvdim), K)
    end
    _g4l_head_norm!(Q, nh, hd, bt.wq_norm, eps_f)
    _g4l_head_norm!(K, nkv, hd, bt.wk_norm, eps_f)
    _g4l_head_norm!(V, nkv, hd, nothing, eps_f)
    n_rot = bt.rope_layer.rotary_dim == 0 ? hd : bt.rope_layer.rotary_dim
    _g4l_rope!(Q, positions, nh, hd, n_rot, bt.rope_layer.inv_freq)
    _g4l_rope!(K, positions, nkv, hd, n_rot, bt.rope_layer.inv_freq)
    return Q, _g4l_heads(K, N, nkv, hd), _g4l_heads(V, N, nkv, hd)
end

function _g4l_window(bt)
    window = bt.attn_window
    return (Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[] && window > 0) ? window : 0
end

# Output projection, post norm, the parallel FFN, layer scale, and the write
# back into X.
function _g4l_finish!(backend, X, O, bt, eps_f::Float32)
    N, E = size(X)
    A = _g4l_mm!(_g4l_new(N, E), O, bt.wo)
    An = _g4l_rms_rows!(_g4l_new(N, E), A, bt.attn_post_rms, eps_f)
    attn_out = _g4l_new(N, E)
    _g4l_launch!(_g4l_add2_kernel!, attn_out, length(attn_out), attn_out, An, X)
    cur = _g4l_moe(attn_out, bt, eps_f)
    out = _g4l_new(N, E)
    _g4l_launch!(_g4l_add2_kernel!, out, length(out), out, cur, attn_out)
    if haskey(bt, :layer_output_scale) && bt.layer_output_scale !== nothing
        Gesso.Inference._gemma4_layer_output_scale!(backend, out, bt.layer_output_scale.storage)
    end
    _g4l_copy!(X, out)
    return X
end

function Gesso.Inference._gemma4_prefill_block!(
    backend::LavaBackend, h, bt, tensors, positions, wl, eps, zeros_like; scratch=nothing,
)
    X = h isa Activation ? h.storage : h
    N, _ = size(X)
    hd = bt.d_head_layer
    nkv = bt.n_kv_heads_layer
    nh = tensors.spec.n_heads
    eps_f = Float32(eps)
    Q, Kh, Vh = _g4l_qkv(X, bt, nh, nkv, hd, positions, eps_f)
    O = _g4l_attention(Q, Kh, Vh, nh, nkv, hd, 1, _g4l_window(bt))
    _g4l_finish!(backend, X, O, bt, eps_f)
    return h
end

function Gesso.Inference._gemma4_generate_block!(
    backend::LavaBackend, hp, kc_layer, vc_layer, bt, tensors, s0, s1, wl, eps, zeros_like; scratch=nothing,
)
    X = hp isa Activation ? hp.storage : hp
    hd = bt.d_head_layer
    nkv = bt.n_kv_heads_layer
    nh = tensors.spec.n_heads
    eps_f = Float32(eps)
    positions = collect((s0 - 1):(s1 - 1))
    Q, Kh, Vh = _g4l_qkv(X, bt, nh, nkv, hd, positions, eps_f)
    _g4l_kv_write!(kc_layer.storage, Kh, s0)
    _g4l_kv_write!(vc_layer.storage, Vh, s0)
    Kd = Gesso.Inference._kv_dense(kc_layer.storage, s1)
    Vd = Gesso.Inference._kv_dense(vc_layer.storage, s1)
    O = _g4l_attention(Q, Kd, Vd, nh, nkv, hd, s0, _g4l_window(bt))
    _g4l_finish!(backend, X, O, bt, eps_f)
    return hp
end
