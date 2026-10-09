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
#   * K-quants Q2_K..Q6_K and Q8_0: native integer kernels against Q8_K/Q8_0
#     activations (`_qgemv_q8k_f32!`, quant_cpu_kquants.jl);
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

function _quant_matmul!(dst::AbstractMatrix, x::AbstractMatrix, q::QuantMatrix)
    seq, ne0 = size(x)
    ne0 == q.ne0 || error("quant matmul: x has $ne0 features, weight has $(q.ne0)")
    size(dst) == (seq, q.nrows) ||
        error("quant matmul: dst $(size(dst)) ≠ ($seq, $(q.nrows))")
    if _quant_native(q.ggml_type)
        xf = Vector{Float32}(undef, q.ne0)
        y = Vector{Float32}(undef, q.nrows)
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
        return dst
    end
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
