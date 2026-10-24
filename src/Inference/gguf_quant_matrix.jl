# Quantized 2-D projection storage for GGUF weights.
#
# A `QuantMatrix` is a dense weight that stays as the file's quant bytes
# (mmap'd for whole tensors; a fresh packed buffer when rows are gathered).
# It is Gesso's `(out, in)` = `(nrows, ne0)` matrix: row `r` is `row_bytes`
# contiguous bytes at `offset0 + (r-1)*row_bytes`, and `ne0` is the GGUF
# innermost axis (the input features).
#
# Matmul dispatch (CPU, correctness first):
#   * native formats in `_qgemv_bytes!` (IQ2_XS, IQ3_XXS, IQ4_XS): the fused
#     quant GEMV, no dequant buffer;
#   * K-quants Q2_K..Q6_K, Q8_0, and IQ2_XXS / IQ2_S / IQ3_S / IQ1_S / IQ1_M:
#     native integer kernels against Q8_K/Q8_0 activations (`_qgemv_q8k_f32!`,
#     quant_cpu_kquants.jl and quant_cpu_iq_s.jl);
#   * F16 and BF16: threaded row dots that widen each weight row once and
#     dot it with every token, up to 256 tokens; longer prefills widen row
#     chunks into F32 and run one BLAS GEMM (`_half_matmul!`);
#   * every other format: FALLBACK. Rows are dequantized per call through the
#     existing ggml `to_float` path (`_dequant_f32`), in bounded row chunks,
#     then multiplied in F32. This is a temporary correctness bridge. It is
#     logged once per ggml type through `@gfallback`. Native kernels for those
#     formats are separate work.

using ..Gesso: @gfallback
import ..Gesso: _is_quant_storage, _quant_matmul!, _quant_embedding_rows!

"""
    QuantMatrix(data, offset0, ggml_type, ne0, nrows, row_bytes, to_float, blck_size)

Quantized `(nrows, ne0)` weight. Not an `AbstractArray` on purpose: code that
takes an array as an activation prototype must not pick it up.
"""
struct QuantMatrix
    data::Vector{UInt8}
    offset0::Int                 # 0-based byte offset of row 1 in `data`
    ggml_type::Int32
    ne0::Int                     # input features (GGUF innermost)
    nrows::Int                   # output rows
    row_bytes::Int
    to_float::Ptr{Cvoid}
    blck_size::Int64
end

Base.size(q::QuantMatrix) = (q.nrows, q.ne0)
Base.size(q::QuantMatrix, d::Integer) = size(q)[d]
Base.ndims(::QuantMatrix) = 2
Base.length(q::QuantMatrix) = q.nrows * q.ne0
Base.eltype(::QuantMatrix) = Float32
_is_host_float_storage(::QuantMatrix) = true
_is_quant_storage(::QuantMatrix) = true

# Types whose matmul runs the integer Q8_K / Q8_0 kernels: the i-quants in
# quant_cpu_simd.jl and the K-quants in quant_cpu_kquants.jl.
_quant_q8k_native(t::Int32) = _q8k_supported_type(t)

_quant_native(t::Int32) =
    t == GGML_TYPE_F32 || t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ3_XXS || t == GGML_TYPE_IQ4_XS ||
    _quant_q8k_native(t)

const GGML_TYPE_F16 = Int32(1)
const GGML_TYPE_BF16 = Int32(30)

_quant_half(t::Int32) = t == GGML_TYPE_F16 || t == GGML_TYPE_BF16

_contiguous_rows(rows) = all(i -> rows[i + 1] == rows[i] + 1, 1:(length(rows) - 1))

"""
    q[rows, :]

Row selection. A consecutive run returns a view over the same bytes. Any other
row list gathers the selected row bytes into a new packed buffer (the fused
Q|gate split of `attn_q` takes this path; it is a one-time load copy).
"""
function Base.getindex(q::QuantMatrix, rows::AbstractVector{<:Integer}, ::Colon)
    isempty(rows) && error("QuantMatrix: empty row selection")
    all(r -> 1 <= r <= q.nrows, rows) || error("QuantMatrix: row selection out of 1:$(q.nrows)")
    rb = q.row_bytes
    rs = collect(Int, rows)
    if _contiguous_rows(rs)
        return QuantMatrix(
            q.data, q.offset0 + (rs[1] - 1) * rb, q.ggml_type, q.ne0, length(rs), rb,
            q.to_float, q.blck_size,
        )
    end
    packed = Vector{UInt8}(undef, length(rs) * rb)
    for (k, r) in enumerate(rs)
        copyto!(packed, (k - 1) * rb + 1, q.data, q.offset0 + (r - 1) * rb + 1, rb)
    end
    return QuantMatrix(
        packed, 0, q.ggml_type, q.ne0, length(rs), rb, q.to_float, q.blck_size,
    )
end

_quant_bytes(q::QuantMatrix) = view(q.data, (q.offset0 + 1):(q.offset0 + q.nrows * q.row_bytes))

"""
    _quant_host_f32(q) -> Matrix{Float32} (nrows, ne0)

Full dequant to host F32 in Gesso layout. Used by the CUDA upload and the
tests; the CPU matmul never builds this whole matrix on the native path.
"""
function _quant_host_f32(q::QuantMatrix)
    y = _dequant_f32(q.data, q.offset0, q.ggml_type, q.nrows * q.ne0)
    return permutedims(reshape(y, q.ne0, q.nrows))
end

const _QUANT_FALLBACK_TYPES = Set{Int32}()

function _note_quant_fallback(t::Int32)
    t in _QUANT_FALLBACK_TYPES && return nothing
    push!(_QUANT_FALLBACK_TYPES, t)
    @gfallback(
        "native quant GEMV for ggml type $t",
        "ggml to_float dequant per call (correctness bridge, not a native kernel)",
    )
    return nothing
end

# Rows dequantized per fallback chunk: bounds the F32 buffer to about 64 MiB.
const _QUANT_FALLBACK_CHUNK_ELEMS = 1 << 24

# Activation and output rows for the single-column loop in `_quant_matmul!`.
# No scratch reaches that call, so the buffers live here. Pool workers run
# only the chunks and never get here, but two generations on two threads can
# both be drivers: the pool itself allows concurrent callers. A caller that
# finds the buffers claimed allocates its own, as every call did before. The
# buffers grow to the largest shape seen and are not shrunk, so steady decode
# allocates nothing on this path.
mutable struct _QuantGemvScratch
    xf::Vector{Float32}
    y::Vector{Float32}
end
const _QUANT_GEMV_SCRATCH = _QuantGemvScratch(Float32[], Float32[])
const _QUANT_GEMV_BUSY = Threads.Atomic{Int}(0)

function _quant_gemv_scratch(ne0::Int, nrows::Int)
    s = _QUANT_GEMV_SCRATCH
    length(s.xf) < ne0 && resize!(s.xf, ne0)
    length(s.y) < nrows && resize!(s.y, nrows)
    return s
end

function _quant_matmul!(dst::AbstractMatrix, x::AbstractMatrix, q::QuantMatrix)
    seq, ne0 = size(x)
    ne0 == q.ne0 || error("quant matmul: x has $ne0 features, weight has $(q.ne0)")
    size(dst) == (seq, q.nrows) ||
        error("quant matmul: dst $(size(dst)) ≠ ($seq, $(q.nrows))")
    if _quant_native(q.ggml_type)
        if seq > 1 && _quant_q8k_native(q.ggml_type) && _prefill_multi_enabled()
            return _quant_matmul_multi!(dst, x, q)
        end
        owned = Threads.atomic_cas!(_QUANT_GEMV_BUSY, 0, 1) == 0
        try
            if owned
                s = _quant_gemv_scratch(q.ne0, q.nrows)
                xf = view(s.xf, 1:q.ne0)
                y = view(s.y, 1:q.nrows)
            else
                xf = view(Vector{Float32}(undef, q.ne0), 1:q.ne0)
                y = view(Vector{Float32}(undef, q.nrows), 1:q.nrows)
            end
            bytes = _quant_bytes(q)
            for t in 1:seq
                @inbounds for i in 1:q.ne0
                    xf[i] = Float32(x[t, i])
                end
                if _quant_q8k_native(q.ggml_type)
                    _qgemv_q8k_f32!(y, xf, bytes, q.ggml_type, q.ne0, q.nrows; threaded=true)
                else
                    _qgemv_bytes_threaded!(y, xf, bytes, q.ggml_type, q.ne0, q.nrows, 1.0f0, 0.0f0)
                end
                @inbounds for r in 1:q.nrows
                    dst[t, r] = y[r]
                end
            end
        finally
            owned && Threads.atomic_xchg!(_QUANT_GEMV_BUSY, 0)
        end
        return dst
    end
    _quant_half(q.ggml_type) && return _half_matmul!(dst, x, q)
    _note_quant_fallback(q.ggml_type)
    xf = Matrix{Float32}(x)
    nc = max(1, _QUANT_FALLBACK_CHUNK_ELEMS ÷ q.ne0)
    r0 = 1
    while r0 <= q.nrows
        r1 = min(q.nrows, r0 + nc - 1)
        nr = r1 - r0 + 1
        buf = _dequant_f32(q.data, q.offset0 + (r0 - 1) * q.row_bytes, q.ggml_type, nr * q.ne0)
        Wc = reshape(buf, q.ne0, nr)
        out = Matrix{Float32}(undef, seq, nr)
        mul!(out, xf, Wc)
        @inbounds for c in 1:nr, t in 1:seq
            dst[t, r0 + c - 1] = out[t, c]
        end
        r0 = r1 + 1
    end
    return dst
end

# ---------------------------------------------------------------- F16 / BF16

# Both widen exactly to Float32: BF16 is the top half of an F32, F16 converts
# through Float16. Rows are 2-byte elements starting at an even byte offset
# in the GGUF data section, so a UInt16 pointer reads them in place.
@inline _half_f32(::Val{GGML_TYPE_BF16}, u::UInt16) = reinterpret(Float32, UInt32(u) << 16)
@inline _half_f32(::Val{GGML_TYPE_F16}, u::UInt16) = Float32(reinterpret(Float16, u))

# Up to this many tokens run as row dots that read each weight row once for
# every token (decode, verify windows, short prefills). Longer prefills widen
# row chunks into F32 and run one BLAS GEMM per chunk.
const _HALF_ROWDOT_MAX_SEQ = 256

function _half_matmul!(dst::AbstractMatrix, x::AbstractMatrix, q::QuantMatrix)
    q.row_bytes == 2 * q.ne0 || error("half matmul: row_bytes $(q.row_bytes) ≠ 2·$(q.ne0)")
    v = Val(q.ggml_type)
    seq = size(x, 1)
    if seq <= _HALF_ROWDOT_MAX_SEQ
        # tokens padded with zero columns to whole tiles of _HALF_TILE
        xt = zeros(Float32, q.ne0, cld(seq, _HALF_TILE) * _HALF_TILE)
        @inbounds for t in 1:seq, i in 1:q.ne0
            xt[i, t] = Float32(x[t, i])
        end
        if dst isa Matrix{Float32}
            _half_rowdot!(dst, xt, q, v)
        else
            out = Matrix{Float32}(undef, seq, q.nrows)
            _half_rowdot!(out, xt, q, v)
            copyto!(dst, out)
        end
        return dst
    end
    xf = Matrix{Float32}(x)
    nc = max(1, _QUANT_FALLBACK_CHUNK_ELEMS ÷ q.ne0)
    W = Matrix{Float32}(undef, q.ne0, min(nc, q.nrows))
    out = Matrix{Float32}(undef, seq, min(nc, q.nrows))
    r0 = 1
    while r0 <= q.nrows
        r1 = min(q.nrows, r0 + nc - 1)
        nr = r1 - r0 + 1
        _half_widen_rows!(W, q, r0, nr, v)
        oc = view(out, :, 1:nr)
        mul!(oc, xf, view(W, :, 1:nr))
        @inbounds for c in 1:nr, t in 1:seq
            dst[t, r0 + c - 1] = oc[t, c]
        end
        r0 = r1 + 1
    end
    return dst
end

# Tokens per pass over a widened row: four independent FMA chains share each
# load of the row.
const _HALF_TILE = 4

# dst[t, j] = Σ_i w[j, i] · xt[i, t] for every row j and token t ≤ size(dst, 1).
# `xt` has a whole number of _HALF_TILE columns (zero padded). Each row is
# widened once into a per-task buffer, then dotted with the tokens a tile at a
# time. Every token runs the same tile code in one fixed order, so a row's
# result is the same for any chunk count and any number of tokens in the call.
function _half_rowdot!(dst::Matrix{Float32}, xt::Matrix{Float32}, q::QuantMatrix, v::Val)
    nc = _qgemv_nchunks(q.nrows)
    seq = size(dst, 1)
    size(xt, 2) % _HALF_TILE == 0 && size(xt, 2) >= seq ||
        error("half rowdot: xt has $(size(xt, 2)) columns for $seq tokens")
    GC.@preserve q xt dst begin
        _run_row_chunks!(nc, q.nrows * q.ne0 * seq) do c
            j0 = div((c - 1) * q.nrows, nc) + 1
            j1 = div(c * q.nrows, nc)
            _half_rows!(dst, xt, q, v, j0, j1)
        end
    end
    return dst
end

# Explicit 8-wide vectors for the row dot. `llvm.fma` is a correctly rounded
# fused multiply-add on every target (a libm call where the CPU lacks FMA),
# so the bits do not depend on what the vectorizer would have chosen.
const _V8 = NTuple{8,VecElement{Float32}}

@inline _v8load(p::Ptr{Float32}) = Base.llvmcall("""
    %v = load <8 x float>, ptr %0, align 4
    ret <8 x float> %v""", _V8, Tuple{Ptr{Float32}}, p)

@inline _v8fma(a::_V8, b::_V8, c::_V8) = Base.llvmcall(("""
    declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)
    define <8 x float> @entry(<8 x float> %a, <8 x float> %b, <8 x float> %c) #0 {
        %r = call <8 x float> @llvm.fma.v8f32(<8 x float> %a, <8 x float> %b, <8 x float> %c)
        ret <8 x float> %r
    }
    attributes #0 = { alwaysinline }""", "entry"), _V8, Tuple{_V8,_V8,_V8}, a, b, c)

@inline _v8add(a::_V8, b::_V8) = Base.llvmcall("""
    %r = fadd <8 x float> %0, %1
    ret <8 x float> %r""", _V8, Tuple{_V8,_V8}, a, b)

# 16 lanes (two vectors) summed in a fixed tree.
@inline function _v8sum(a::_V8, b::_V8)
    c = _v8add(a, b)
    return ((c[1].value + c[5].value) + (c[3].value + c[7].value)) +
           ((c[2].value + c[6].value) + (c[4].value + c[8].value))
end

# Dot one widened row with four token columns. Lane k of 16 accumulates the
# elements i ≡ k (mod 16) in index order; the n % 16 tail is a scalar fma
# chain added last. The order is fixed in the source, so a token gets the
# same bits in any tile slot and in any call.
@inline function _half_dot4(pb::Ptr{Float32}, p1::Ptr{Float32}, p2::Ptr{Float32},
                            p3::Ptr{Float32}, p4::Ptr{Float32}, n::Int)
    z = ntuple(_ -> VecElement(0.0f0), Val(8))
    a1 = b1 = a2 = b2 = a3 = b3 = a4 = b4 = z
    i = 0
    while i + 16 <= n
        wa = _v8load(pb + 4i)
        wb = _v8load(pb + 4i + 32)
        a1 = _v8fma(wa, _v8load(p1 + 4i), a1); b1 = _v8fma(wb, _v8load(p1 + 4i + 32), b1)
        a2 = _v8fma(wa, _v8load(p2 + 4i), a2); b2 = _v8fma(wb, _v8load(p2 + 4i + 32), b2)
        a3 = _v8fma(wa, _v8load(p3 + 4i), a3); b3 = _v8fma(wb, _v8load(p3 + 4i + 32), b3)
        a4 = _v8fma(wa, _v8load(p4 + 4i), a4); b4 = _v8fma(wb, _v8load(p4 + 4i + 32), b4)
        i += 16
    end
    t1 = t2 = t3 = t4 = 0.0f0
    for k in i+1:n
        w = unsafe_load(pb, k)
        t1 = fma(w, unsafe_load(p1, k), t1)
        t2 = fma(w, unsafe_load(p2, k), t2)
        t3 = fma(w, unsafe_load(p3, k), t3)
        t4 = fma(w, unsafe_load(p4, k), t4)
    end
    return _v8sum(a1, b1) + t1, _v8sum(a2, b2) + t2, _v8sum(a3, b3) + t3, _v8sum(a4, b4) + t4
end

# Rows widened per pass over the token tiles once a call has more than 16 tokens.
const _HALF_ROW_BLOCK = 2

function _half_rows!(dst::Matrix{Float32}, xt::Matrix{Float32}, q::QuantMatrix, v::Val, j0::Int, j1::Int)
    n = q.ne0
    seq = size(dst, 1)
    # Short calls stream row by row so the widened row stays in L1.
    rb = seq <= 16 ? 1 : _HALF_ROW_BLOCK
    buf = Vector{Float32}(undef, n * rb)
    GC.@preserve buf begin
        @inbounds for b0 in j0:rb:j1
            b1 = min(b0 + rb - 1, j1)
            for j in b0:b1
                pb = pointer(buf, (j - b0) * n + 1)
                p = Ptr{UInt16}(pointer(q.data, q.offset0 + (j - 1) * q.row_bytes + 1))
                @simd for i in 1:n
                    unsafe_store!(pb, _half_f32(v, unsafe_load(p, i)), i)
                end
            end
            for t0 in 1:_HALF_TILE:seq
                p1 = pointer(xt, (t0 - 1) * n + 1)
                p2 = p1 + 4n
                p3 = p2 + 4n
                p4 = p3 + 4n
                for j in b0:b1
                    a1, a2, a3, a4 = _half_dot4(pointer(buf, (j - b0) * n + 1), p1, p2, p3, p4, n)
                    dst[t0, j] = a1
                    t0 + 1 <= seq && (dst[t0 + 1, j] = a2)
                    t0 + 2 <= seq && (dst[t0 + 2, j] = a3)
                    t0 + 3 <= seq && (dst[t0 + 3, j] = a4)
                end
            end
        end
    end
    return nothing
end

# Columns 1:nr of `W` (ne0, ≥ nr) ← rows r0 .. r0+nr-1 of `q`, widened to F32.
function _half_widen_rows!(W::Matrix{Float32}, q::QuantMatrix, r0::Int, nr::Int, v::Val)
    nc = _qgemv_nchunks(nr)
    n = q.ne0
    GC.@preserve q W begin
        _run_row_chunks!(nc, nr * n) do c
            c0 = div((c - 1) * nr, nc) + 1
            c1 = div(c * nr, nc)
            @inbounds for k in c0:c1
                p = Ptr{UInt16}(pointer(q.data, q.offset0 + (r0 + k - 2) * q.row_bytes + 1))
                pw = pointer(W, (k - 1) * n + 1)
                @simd for i in 1:n
                    unsafe_store!(pw, _half_f32(v, unsafe_load(p, i)), i)
                end
            end
        end
    end
    return W
end

# Prefill (seq > 1) of a Q8_K-kind weight: each token row is quantized once, and
# the weight rows are split across threads. A chunk walks its rows in blocks of
# _MULTI_ROW_BLOCK; for each tile of _MULTI_TILE tokens it runs every row of the
# block, so a row's sub-block unpacking is shared across the tile and the tile's
# activations stay in cache across the block.
function _quant_matmul_multi!(dst::AbstractMatrix, x::AbstractMatrix, q::QuantMatrix)
    seq = size(x, 1)
    simd = q8k_simd_enabled()
    # Q6_K: shuffle each activation once here rather than once per weight row.
    xqs = _prefill_quantize(q.ggml_type, x, seq, q.ne0, simd && q.ggml_type == GGML_TYPE_Q6_K)
    # Ticketed chunks as in the decode GEMV: one chunk per thread waits on the
    # slowest thread (2-3x on the 27B's verify window). Rows are independent,
    # so the bits do not depend on the chunk count.
    nc = _qgemv_nchunks(q.nrows)
    # The Q6_K, Q8_0 and Q4_0 tiles read their columns' d values from one table.
    dt = simd && (q.ggml_type == GGML_TYPE_Q6_K || _q80_act_type(q.ggml_type)) ? _multi_d_table(xqs) : nothing
    chunk = _PrefillChunk(dst, _quant_bytes(q), xqs, dt, q.ggml_type, q.ne0, q.nrows, nc, simd)
    _run_row_chunks!(chunk, nc, q.nrows * q.ne0 * seq)
    return dst
end

# Quantizes each token row (and preps it, for Q6_K), with the tokens split
# across threads.
function _prefill_quantize(ggml_type::Int32, x::AbstractMatrix, seq::Int, ne0::Int, prep::Bool=false)
    x1 = _prefill_quantize_row(ggml_type, x, 1, Vector{Float32}(undef, ne0), prep)
    return _prefill_quantize_rest!(fill(x1, seq), ggml_type, x, prep)
end

function _prefill_quantize_row(ggml_type::Int32, x::AbstractMatrix, t::Int, xf::Vector{Float32}, prep::Bool)
    @inbounds for i in eachindex(xf)
        xf[i] = Float32(x[t, i])
    end
    xq = _q8k_quantize_act(ggml_type, xf)
    return prep ? _q6k_prep(xq) : xq
end

function _prefill_quantize_rest!(xqs::Vector, ggml_type::Int32, x::AbstractMatrix, prep::Bool)
    seq = length(xqs)
    ne0 = size(x, 2)
    nc = max(1, min(Threads.nthreads(), seq - 1))
    _run_row_chunks!(nc, (seq - 1) * ne0 * 16) do c
        xf = Vector{Float32}(undef, ne0)
        for t in (div((c - 1) * (seq - 1), nc) + 2):(div(c * (seq - 1), nc) + 1)
            xqs[t] = _prefill_quantize_row(ggml_type, x, t, xf, prep)
        end
    end
    return xqs
end

# Weight rows per block of a prefill chunk. On Spark-sized Q6_K FFN matrices (Xeon
# E5-2699 v3, 8 threads) 32 and 64 tie; row by row is 1.2x slower, 1.6x with the
# Q6_K prep, whose activations are twice the bytes.
const _MULTI_ROW_BLOCK = 32

# One row chunk of _quant_matmul_multi!, as a callable struct (see _GroupChunk).
struct _PrefillChunk{D<:AbstractMatrix,B<:AbstractVector{UInt8},X<:AbstractVector,T}
    dst::D
    bytes::B
    xqs::X
    dt::T
    ggml_type::Int32
    ne0::Int
    nrows::Int
    nc::Int
    simd::Bool
end

function (p::_PrefillChunk)(c::Int)
    j0 = div((c - 1) * p.nrows, p.nc) + 1
    j1 = div(c * p.nrows, p.nc)
    rb = _qrow_bytes(p.ggml_type, p.ne0)
    seq = length(p.xqs)
    buf = Vector{Float32}(undef, seq)
    @inbounds for b0 in j0:_MULTI_ROW_BLOCK:j1
        b1 = min(b0 + _MULTI_ROW_BLOCK - 1, j1)
        t0 = 1
        while t0 <= seq
            K = min(_MULTI_TILE, seq - t0 + 1)
            for j in b0:b1
                _dot_q8k_multi!(buf, p.ggml_type, p.bytes, (j - 1) * rb, p.xqs, t0, K, p.simd, p.dt)
                for t in t0:(t0 + K - 1)
                    # Same expression as the per-token path, so -0.0 becomes +0.0 identically.
                    p.dst[t, j] = 1.0f0 * buf[t] + 0.0f0
                end
            end
            t0 += K
        end
    end
    return nothing
end

function _quant_embedding_rows!(dst::AbstractMatrix, q::QuantMatrix, tokens)
    rb = q.row_bytes
    for (t, tok) in enumerate(tokens)
        buf = _dequant_f32(q.data, q.offset0 + Int(tok) * rb, q.ggml_type, q.ne0)
        @inbounds for i in 1:q.ne0
            dst[t, i] = buf[i]
        end
    end
    return dst
end

# ---------------------------------------------------------------- shared input

# True when `q` multiplies through the Q8_K integer kernel, so one Q8_K
# quantization of its input serves every weight that reads that input. Q8_0 and
# Q4_0 weights use a Q8_0 activation and fall back to the per-weight path.
_quant_q8k_kind(q::QuantMatrix) = _quant_q8k_native(q.ggml_type) && !_q80_act_type(q.ggml_type)

"""
    _quant_matmul_group!(dsts, xq, ws; simd) -> dsts

Seq = 1 projections of one shared input. `xq` is the Q8_K quantization of the
input row (`Q8KVec`, length `ws[m].ne0`), `dsts[m]` a `(1, ws[m].nrows)` Float32
matrix, and every `ws[m]` a `QuantMatrix` of Q8_K kind (see `_quant_q8k_kind`).

The rows of all weights form one range, split into row chunks that run as a
single threaded pass. Each output row is the same `_dot_q8k` on the same
`xq` as `_quant_matmul!` uses for that weight alone, so the outputs are
bit-identical to separate matmuls for any chunk count.
"""
function _quant_matmul_group!(dsts::Tuple, xq::Q8KVec, ws::Tuple; simd::Bool=q8k_simd_enabled())
    for w in ws
        _quant_q8k_kind(w) || error("quant group: weight type $(w.ggml_type) has no Q8_K kernel")
        w.ne0 == length(xq) || error("quant group: x has $(length(xq)) features, weight has $(w.ne0)")
    end
    for (d, w) in zip(dsts, ws)
        size(d) == (1, w.nrows) || error("quant group: dst $(size(d)) ≠ (1, $(w.nrows))")
    end
    total = _group_rows(ws)
    nc = _qgemv_nchunks(total)
    _run_row_chunks!(_GroupChunk(dsts, ws, xq, total, nc, simd), nc, total * length(xq))
    return dsts
end

# One chunk of a group pass, as a callable struct: the pool keeps a reference to
# it for the call, so it is built once per call and its fields never change.
struct _GroupChunk{D<:Tuple,W<:Tuple}
    dsts::D
    ws::W
    xq::Q8KVec
    total::Int
    nc::Int
    simd::Bool
end

function (g::_GroupChunk)(c::Int)
    g0 = div((c - 1) * g.total, g.nc) + 1
    g1 = div(c * g.total, g.nc)
    _group_segments!(g.dsts, g.ws, 0, g0, g1, g.xq, g.simd)
    return nothing
end

_group_rows(ws::Tuple) = sum(w -> w.nrows, ws; init=0)

# Walks the weights in order. `base` is the global row offset of the first
# weight in the tuple, so the chunk's global rows [g0, g1] intersect each
# weight's own rows [base+1, base+nrows] at most once.
@inline function _group_segments!(dsts::Tuple, ws::Tuple, base::Int, g0::Int, g1::Int, xq::Q8KVec, simd::Bool)
    w = first(ws)
    lo = max(g0, base + 1)
    hi = min(g1, base + w.nrows)
    lo <= hi && _gemv_q8k_rows_into!(first(dsts), xq, w, lo - base, hi - base, simd)
    return _group_segments!(Base.tail(dsts), Base.tail(ws), base + w.nrows, g0, g1, xq, simd)
end
@inline _group_segments!(::Tuple{}, ::Tuple{}, ::Int, ::Int, ::Int, ::Q8KVec, ::Bool) = nothing

# Rows j0:j1 of `w` into row 1 of `dst`, written as `alpha * acc + beta * y`
# with alpha = 1 and beta = 0, the expression `_qgemv_q8k_rows!` evaluates.
# The weight bytes are read in place at the file offset: no view is built.
function _gemv_q8k_rows_into!(dst::AbstractMatrix{Float32}, xq::Q8KVec, w::QuantMatrix,
    j0::Int, j1::Int, simd::Bool)
    t = w.ggml_type
    rb = _qrow_bytes(t, w.ne0)
    data = w.data
    base = w.offset0
    @inbounds for j in j0:j1
        acc = _dot_q8k(t, data, base + (j - 1) * rb, xq, simd)
        dst[1, j] = 1.0f0 * acc + 0.0f0
    end
    return dst
end
