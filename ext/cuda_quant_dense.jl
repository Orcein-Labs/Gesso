# Dense GGUF quant weights kept in their native bytes on the device, and the
# decode GEMV that reads them.
#
# Design follows llama.cpp's MMVQ path (MIT; ggml/src/ggml-cuda/mmvq.cu,
# vecdotq.cuh, quantize.cu): the activation row is quantized once per matmul
# to 32-value blocks (d = amax/127, int8 values, the block sum kept for the
# K-quant minimums), each output row is handled by one warp, and each 32-value
# block is one dp4a dot product with the weight values unpacked into int8
# words. The block byte layouts are the ggml ones. The decode math here is
# written from those layouts and is checked against this repo's CPU dequant
# (`_quant_host_f32`) in test/test_cuda_quant_dense.jl.
#
# Activation layout (per token t, block gb = (t-1)*nb + b, 0-based b):
#   qx[gb*8 + w + 1]  Int32, four int8 values packed little-endian (w = 0..7)
#   dx[gb + 1]        Float32 block scale d = amax/127
#   sx[gb + 1]        Int32 sum of the block's int8 values

const _DQ_KINDS = (:q4_k, :q5_k, :q6_k, :iq4_xs, :q8_0)

# Device-resident dense weight: the file's bytes for rows 1..nrows, row r at
# (r-1)*row_bytes, in ggml block order. Not an AbstractArray on purpose, as on
# the host side (`QuantMatrix`).
struct GpuDenseQuant
    data::CuVector{UInt8}
    ggml_type::Int32
    ne0::Int
    nrows::Int
    row_bytes::Int
end

Base.size(q::GpuDenseQuant) = (q.nrows, q.ne0)
Base.size(q::GpuDenseQuant, d::Integer) = size(q)[d]
Base.ndims(::GpuDenseQuant) = 2
Base.length(q::GpuDenseQuant) = q.nrows * q.ne0
Base.sizeof(q::GpuDenseQuant) = sizeof(q.data)

# GESSO_GPU_DENSE_QUANT=0 restores the dequant-to-F16 path for A/B runs.
_gpu_dense_quant_on() = get(ENV, "GESSO_GPU_DENSE_QUANT", "1") != "0"

function _dq_kind(t::Int32)
    I = Gesso.Inference
    t == I.GGML_TYPE_Q4_K && return :q4_k
    t == I.GGML_TYPE_Q5_K && return :q5_k
    t == I.GGML_TYPE_Q6_K && return :q6_k
    t == I.GGML_TYPE_IQ4_XS && return :iq4_xs
    t == I.GGML_TYPE_Q8_0 && return :q8_0
    return nothing
end

# Whether a host QuantMatrix can be placed natively on the device. The K-quant
# superblock is 256 values, Q8_0 blocks are 32.
function _dq_eligible(s)
    s isa Gesso.Inference.QuantMatrix || return false
    _gpu_dense_quant_on() || return false
    kind = _dq_kind(s.ggml_type)
    kind === nothing && return false
    s.ne0 % 32 == 0 || return false
    kind === :q8_0 || s.ne0 % 256 == 0 || return false
    return true
end

_dq_device_bytes(s::Gesso.Inference.QuantMatrix) = Int64(s.nrows) * Int64(s.row_bytes)

function _to_gpu_dense_quant(s::Gesso.Inference.QuantMatrix)
    bytes = CuArray{UInt8}(Gesso.Inference._quant_bytes(s))
    return GpuDenseQuant(bytes, s.ggml_type, s.ne0, s.nrows, s.row_bytes)
end

# --- activation quantize: Q8_1-style, one thread per (token, 32-value block) --

function _cuda_q8_1_quant_kernel!(qx, dx, sx, X, nb::Int, total::Int)
    g = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    g > total && return
    t = (g - 1) ÷ nb + 1
    b = (g - 1) % nb
    base = b * 32
    @inbounds begin
        amax = 0.0f0
        for l in 1:32
            amax = max(amax, abs(X[t, base + l]))
        end
        d = amax / 127.0f0
        inv = amax == 0.0f0 ? 0.0f0 : 127.0f0 / amax
        ssum = Int32(0)
        wbase = (t - 1) * nb * 8 + b * 8
        for w in 0:7
            u = UInt32(0)
            for tt in 0:3
                v = X[t, base + 4 * w + tt + 1] * inv
                qi = unsafe_trunc(Int32, v + copysign(0.5f0, v))
                qi = clamp(qi, Int32(-127), Int32(127))
                ssum += qi
                u |= UInt32(qi & 0xff) << (8 * tt)
            end
            qx[wbase + w + 1] = reinterpret(Int32, u)
        end
        dx[g] = d
        sx[g] = ssum
    end
    return
end

# --- per-block contributions ------------------------------------------------
# Each returns the float dot product of one 32-value weight block with the
# matching activation block. `boff` is the byte offset of the superblock (or
# of the Q8_0 block), `gb` the activation block index.

# Q4_K / Q5_K: value = d*sc*q - dmin*mn, q is 4 bits (Q5_K adds one high bit
# from qh). Sub-block s of a superblock takes nibble (s&1) of qs chunk s>>1.
# Superblocks are 144 and 176 bytes, so every word read here is 4-aligned and
# four values unpack at once with byte-lane masks.
@inline function _dq_kq45(W, boff::Int, has_qh::Bool, b::Int, gb::Int, qx, dx, sx)
    @inbounds begin
        s = b & 7
        dd = _cu_u32a(W, boff)
        d = Float32(reinterpret(Float16, UInt16(dd & 0xffff)))
        dmin = Float32(reinterpret(Float16, UInt16(dd >> 16)))
        sbase = boff + 4
        if s < 4
            sc = Float32(W[sbase + s + 1] & 0x3f)
            mn = Float32(W[sbase + s + 5] & 0x3f)
        else
            sc = Float32((W[sbase + s + 5] & 0x0f) | ((W[sbase + s - 3] >> 6) << 4))
            mn = Float32((W[sbase + s + 5] >> 4) | ((W[sbase + s + 1] >> 6) << 4))
        end
        qs0 = boff + (has_qh ? 48 : 16) + 32 * (s >> 1)
        qh0 = boff + 16
        shift = 4 * (s & 1)
        isum = Int32(0)
        for w in 0:7
            word = (_cu_u32a(W, qs0 + 4 * w) >> shift) & 0x0f0f0f0f
            if has_qh
                word |= ((_cu_u32a(W, qh0 + 4 * w) >> s) & 0x01010101) << 4
            end
            isum = CUDA.dp4a(reinterpret(Int32, word), qx[gb * 8 + w + 1], isum)
        end
        return dx[gb + 1] * (d * sc * Float32(isum) - dmin * mn * Float32(sx[gb + 1]))
    end
end

# Q6_K: value = d*sc*(q - 32), 6-bit q split into ql (low 4) and qh (high 2).
# Sub-block s = 4h + g covers values 32s..32s+31; its first 16 values use scale
# 2s and the last 16 use scale 2s+1. The 210-byte superblock is only 2-aligned.
@inline function _dq_q6k(W, boff::Int, b::Int, gb::Int, qx, dx)
    @inbounds begin
        s = b & 7
        h = s >> 2
        g = s & 3
        ql0 = boff + 64 * h + 32 * (g & 1)
        hi = g >> 1
        qh0 = boff + 128 + 32 * h
        sh = 2 * g
        ia = Int32(0)
        ib = Int32(0)
        for w in 0:7
            lw = (_cu_u32h(W, ql0 + 4 * w) >> (4 * hi)) & 0x0f0f0f0f
            hw = (_cu_u32h(W, qh0 + 4 * w) >> sh) & 0x03030303
            # q - 32 in each byte lane. q is 0..63, so with bit 7 set first the
            # subtraction never borrows from the next lane.
            word = ((lw | (hw << 4) | 0x80808080) - 0x20202020) ⊻ 0x80808080
            xw = qx[gb * 8 + w + 1]
            if w < 4
                ia = CUDA.dp4a(reinterpret(Int32, word), xw, ia)
            else
                ib = CUDA.dp4a(reinterpret(Int32, word), xw, ib)
            end
        end
        d = _cu_f16a(W, boff + 208)
        sc_a = Float32(reinterpret(Int8, W[boff + 192 + 2 * s + 1]))
        sc_b = Float32(reinterpret(Int8, W[boff + 192 + 2 * s + 2]))
        return dx[gb + 1] * d * (sc_a * Float32(ia) + sc_b * Float32(ib))
    end
end

# Four IQ4_NL table values for the four nibbles in the low bits of each byte lane.
@inline function _kv4_word(kv, nibs::UInt32)
    @inbounds begin
        word = UInt32(0)
        for tt in 0:3
            word |= UInt32(reinterpret(UInt8, kv[Int((nibs >> (8 * tt)) & 0x0f) + 1])) << (8 * tt)
        end
        return reinterpret(Int32, word)
    end
end

# IQ4_XS: 6-bit sub-block scale ls (value ls-32), values are the IQ4_NL table
# entries for the nibbles; the low nibbles of a 16-byte run are values 0..15,
# the high nibbles are values 16..31.
@inline function _dq_iq4xs(W, boff::Int, b::Int, gb::Int, qx, dx, kv)
    @inbounds begin
        ib = b & 7
        d = _cu_f16a(W, boff)
        scales_h = _cu_u16a(W, boff + 2)
        sl = W[boff + 4 + (ib >> 1) + 1]
        ls = Int((sl >> (4 * (ib & 1))) & 0x0f) | (Int((scales_h >> (2 * ib)) & 0x03) << 4)
        dl = d * Float32(ls - 32)
        qs0 = boff + 8 + 16 * ib
        isum = Int32(0)
        for w in 0:3
            qw = _cu_u32a(W, qs0 + 4 * w)
            isum = CUDA.dp4a(_kv4_word(kv, qw), qx[gb * 8 + w + 1], isum)
            isum = CUDA.dp4a(_kv4_word(kv, qw >> 4), qx[gb * 8 + w + 5], isum)
        end
        return dx[gb + 1] * dl * Float32(isum)
    end
end

@inline function _cuda_dq_block(::Val{:q4_k}, W, rowoff::Int, b::Int, qx, dx, sx, tok::Int, nb::Int, kv)
    return _dq_kq45(W, rowoff + (b >> 3) * 144, false, b, (tok - 1) * nb + b, qx, dx, sx)
end
@inline function _cuda_dq_block(::Val{:q5_k}, W, rowoff::Int, b::Int, qx, dx, sx, tok::Int, nb::Int, kv)
    return _dq_kq45(W, rowoff + (b >> 3) * 176, true, b, (tok - 1) * nb + b, qx, dx, sx)
end
@inline function _cuda_dq_block(::Val{:q6_k}, W, rowoff::Int, b::Int, qx, dx, sx, tok::Int, nb::Int, kv)
    return _dq_q6k(W, rowoff + (b >> 3) * 210, b, (tok - 1) * nb + b, qx, dx)
end
@inline function _cuda_dq_block(::Val{:iq4_xs}, W, rowoff::Int, b::Int, qx, dx, sx, tok::Int, nb::Int, kv)
    return _dq_iq4xs(W, rowoff + (b >> 3) * 136, b, (tok - 1) * nb + b, qx, dx, kv)
end
@inline function _cuda_dq_block(::Val{:q8_0}, W, rowoff::Int, b::Int, qx, dx, sx, tok::Int, nb::Int, kv)
    @inbounds begin
        off = rowoff + b * 34
        d = _cu_f16a(W, off)
        gb = (tok - 1) * nb + b
        isum = Int32(0)
        for w in 0:7
            isum = CUDA.dp4a(reinterpret(Int32, _cu_u32h(W, off + 2 + 4 * w)), qx[gb * 8 + w + 1], isum)
        end
        return d * dx[gb + 1] * Float32(isum)
    end
end

# --- decode GEMV: one warp per output row, four rows per block --------------

function _cuda_dq_gemv_kernel!(y, W, qx, dx, sx, kv, ::Val{K}, nrows::Int, nb::Int, row_bytes::Int) where {K}
    lane = Int((threadIdx().x - 1) % 32)
    row = Int((blockIdx().x - 1) * 4 + (threadIdx().x - 1) ÷ 32 + 1)
    tok = Int(blockIdx().y)
    row > nrows && return
    rowoff = (row - 1) * row_bytes
    acc = 0.0f0
    b = lane
    while b < nb
        acc += _cuda_dq_block(Val{K}(), W, rowoff, b, qx, dx, sx, tok, nb, kv)
        b += 32
    end
    off = 16
    while off >= 1
        acc += CUDA.shfl_down_sync(0xffffffff, acc, off)
        off >>= 1
    end
    if lane == 0
        @inbounds y[tok, row] = acc
    end
    return
end

# dst[t, o] = sum_i x[t, i] * W[o, i] for a native GpuDenseQuant W.
function _cuda_dq_matmul!(dst, x, w)
    q = w.storage
    xs = x.storage
    ds = dst.storage
    n, k = size(xs, 1), size(xs, 2)
    k == q.ne0 || throw(DimensionMismatch("dense quant matmul: x is $(size(xs)), W is $(size(q))"))
    size(ds) == (n, q.nrows) || throw(DimensionMismatch("dense quant matmul: dst is $(size(ds)), want ($n, $(q.nrows))"))
    n == 0 && return dst
    _cu_aligned(q.data, 4) || error("dense quant matmul: weight bytes not 4-byte aligned")
    nb = k ÷ 32
    total = n * nb
    qx, dx, sx = _cuda_q8_buffers!(_CUDA_Q8_DENSE, total)
    @cuda threads = 256 blocks = cld(total, 256) _cuda_q8_1_quant_kernel!(qx, dx, sx, xs, nb, total)
    kv = _cuda_q_tables().kv4
    @cuda threads = 128 blocks = (cld(q.nrows, 4), n) _cuda_dq_gemv_kernel!(
        ds, q.data, qx, dx, sx, kv, Val{_dq_kind(q.ggml_type)}(), q.nrows, nb, q.row_bytes,
    )
    return dst
end
