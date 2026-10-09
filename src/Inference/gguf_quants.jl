# Quantized expert GEMV for the live Ornith GGUF types (IQ2_XS / IQ3_XXS /
# IQ4_XS) plus packed F32. CPU fused dequant-dot matches ggml `to_float`
# then a column dot; the CUDA lowering lives in ext/cuda_quant_gemm.jl.

include("gguf_quant_tables.jl")

const GGML_TYPE_F32 = Int32(0)
const GGML_TYPE_IQ2_XS = Int32(17)
const GGML_TYPE_IQ3_XXS = Int32(18)
const GGML_TYPE_IQ4_XS = Int32(23)

"""
GPU-resident packed quantized expert bank. `data` is `n_exp * bytes_per_expert`
bytes in ggml `(ne0, ne1)` block order (same layout as `GgufQuantSlice`).
`A` is `CuArray{UInt8}` on the CUDA FAST path; a host `Vector{UInt8}` is
legal for tests.
"""
struct GpuQuantSlice{A<:AbstractVector{UInt8}}
    data::A
    ggml_type::Int32
    n_exp::Int
    ne0::Int
    ne1::Int
    bytes_per_expert::Int
    blck_size::Int64
end

Base.size(s::GpuQuantSlice) = (s.n_exp, s.ne1, s.ne0)
Base.size(s::GpuQuantSlice, d::Int) = size(s)[d]
Base.ndims(::GpuQuantSlice) = 3
Base.eltype(::GpuQuantSlice) = Float32
Base.length(s::GpuQuantSlice) = prod(size(s))
Base.sizeof(s::GpuQuantSlice) = sizeof(s.data)
_is_host_float_storage(::GpuQuantSlice) = true

function _iq_block_bytes(ggml_type::Int32)
    ggml_type == GGML_TYPE_IQ2_XS && return 74
    ggml_type == GGML_TYPE_IQ3_XXS && return 98
    ggml_type == GGML_TYPE_IQ4_XS && return 136
    error("gguf_quants: no fused GEMV for ggml type $ggml_type")
end

_quant_nbytes(s::GgufQuantSlice) = Int64(s.n_exp) * Int64(s.bytes_per_expert)
_quant_nbytes(s::GpuQuantSlice) = Int64(length(s.data))

function _quant_expert_off0(s::GgufQuantSlice, e::Int)
    1 <= e <= s.n_exp || error("qgemv: expert $e out of 1:$(s.n_exp)")
    return s.offset0 + (e - 1) * s.bytes_per_expert
end
function _quant_expert_off0(s::GpuQuantSlice, e::Int)
    1 <= e <= s.n_exp || error("qgemv: expert $e out of 1:$(s.n_exp)")
    return (e - 1) * s.bytes_per_expert
end

function _quant_expert_bytes(s::GgufQuantSlice, e::Int)
    off = _quant_expert_off0(s, e)
    return view(s.data, (off + 1):(off + s.bytes_per_expert))
end
function _quant_expert_bytes(s::GpuQuantSlice, e::Int)
    off = _quant_expert_off0(s, e)
    return view(s.data, (off + 1):(off + s.bytes_per_expert))
end

function packed_quant_bytes(s::GgufQuantSlice)
    n = s.n_exp * s.bytes_per_expert
    return view(s.data, (s.offset0 + 1):(s.offset0 + n))
end

function gpu_quant_slice(s::GgufQuantSlice, data::AbstractVector{UInt8})
    length(data) == s.n_exp * s.bytes_per_expert ||
        error("gpu_quant_slice: packed length $(length(data)) ≠ $(s.n_exp * s.bytes_per_expert)")
    return GpuQuantSlice(
        data,
        s.ggml_type,
        s.n_exp,
        s.ne0,
        s.ne1,
        s.bytes_per_expert,
        s.blck_size,
    )
end

@inline function _ld_u8(q, off0::Int)
    return q[off0 + 1]
end
@inline function _ld_u16(q, off0::Int)
    return UInt16(_ld_u8(q, off0)) | (UInt16(_ld_u8(q, off0 + 1)) << 8)
end
@inline function _ld_u32(q, off0::Int)
    return UInt32(_ld_u8(q, off0)) |
           (UInt32(_ld_u8(q, off0 + 1)) << 8) |
           (UInt32(_ld_u8(q, off0 + 2)) << 16) |
           (UInt32(_ld_u8(q, off0 + 3)) << 24)
end
@inline function _ld_f16(q, off0::Int)
    return Float32(reinterpret(Float16, _ld_u16(q, off0)))
end
@inline function _ld_f32(q, off0::Int)
    return reinterpret(Float32, _ld_u32(q, off0))
end

function _qgemv_f32!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    q,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32,
)
    @inbounds for j in 1:ne1
        acc = zero(Float32)
        base = (j - 1) * ne0 * 4
        for i in 1:ne0
            acc += _ld_f32(q, base + (i - 1) * 4) * x[i]
        end
        y[j] = alpha * acc + (beta == 0.0f0 ? 0.0f0 : beta * y[j])
    end
    return y
end

function _qgemv_iq2_xs!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    q,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32,
)
    nb = div(ne0, QK_K)
    bsz = 74
    @inbounds for j in 1:ne1
        acc = zero(Float32)
        col = (j - 1) * nb * bsz
        for ib in 0:(nb - 1)
            boff = col + ib * bsz
            d = _ld_f16(q, boff)
            xoff = ib * QK_K
            for ib32 in 0:7
                sc = _ld_u8(q, boff + 66 + ib32)
                db0 = d * (0.5f0 + Float32(sc & 0x0f)) * 0.25f0
                db1 = d * (0.5f0 + Float32(sc >> 4)) * 0.25f0
                for l in 0:3
                    qv = _ld_u16(q, boff + 2 + (4 * ib32 + l) * 2)
                    grid64 = IQ2XS_GRID[(qv & 0x01ff) + 0x0001]
                    signs = KSIGNS_IQ2XS[(qv >> 9) + 0x0001]
                    db = l < 2 ? db0 : db1
                    base = xoff + ib32 * 32 + l * 8
                    for t in 0:7
                        g = Float32((grid64 >> (8 * t)) & 0xff)
                        sng = (signs & KMASK_IQ2XS[t + 1]) != 0x00 ? -1.0f0 : 1.0f0
                        acc += db * g * sng * x[base + t + 1]
                    end
                end
            end
        end
        y[j] = alpha * acc + (beta == 0.0f0 ? 0.0f0 : beta * y[j])
    end
    return y
end

function _qgemv_iq3_xxs!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    q,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32,
)
    nb = div(ne0, QK_K)
    bsz = 98
    @inbounds for j in 1:ne1
        acc = zero(Float32)
        col = (j - 1) * nb * bsz
        for ib in 0:(nb - 1)
            boff = col + ib * bsz
            d = _ld_f16(q, boff)
            qs0 = boff + 2
            ss0 = qs0 + 64
            xoff = ib * QK_K
            for ib32 in 0:7
                aux32 = _ld_u32(q, ss0 + 4 * ib32)
                db = d * (0.5f0 + Float32(aux32 >> 28)) * 0.5f0
                qbase = qs0 + ib32 * 8
                for l in 0:3
                    signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & UInt32(127)) + UInt32(1)]
                    g1 = IQ3XXS_GRID[Int(_ld_u8(q, qbase + 2 * l)) + 1]
                    g2 = IQ3XXS_GRID[Int(_ld_u8(q, qbase + 2 * l + 1)) + 1]
                    base = xoff + ib32 * 32 + l * 8
                    for t in 0:3
                        s0 = (signs & KMASK_IQ2XS[t + 1]) != 0x00 ? -1.0f0 : 1.0f0
                        s1 = (signs & KMASK_IQ2XS[t + 5]) != 0x00 ? -1.0f0 : 1.0f0
                        acc += db * Float32((g1 >> (8 * t)) & 0xff) * s0 * x[base + t + 1]
                        acc += db * Float32((g2 >> (8 * t)) & 0xff) * s1 * x[base + t + 5]
                    end
                end
            end
        end
        y[j] = alpha * acc + (beta == 0.0f0 ? 0.0f0 : beta * y[j])
    end
    return y
end

function _qgemv_iq4_xs!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    q,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32,
)
    nb = div(ne0, QK_K)
    bsz = 136
    @inbounds for j in 1:ne1
        acc = zero(Float32)
        col = (j - 1) * nb * bsz
        for ibk in 0:(nb - 1)
            boff = col + ibk * bsz
            d = _ld_f16(q, boff)
            scales_h = _ld_u16(q, boff + 2)
            xoff = ibk * QK_K
            qs0 = boff + 8
            for ib in 0:7
                sl = _ld_u8(q, boff + 4 + div(ib, 2))
                ls = Int((sl >> (4 * (ib % 2))) & 0x0f) | Int((scales_h >> (2 * ib)) & 0x0003) << 4
                dl = d * Float32(ls - 32)
                qsb = qs0 + ib * 16
                base = xoff + ib * 32
                for t in 0:15
                    qs = _ld_u8(q, qsb + t)
                    acc += dl * Float32(KVALUES_IQ4NL[(qs & 0x0f) + 0x01]) * x[base + t + 1]
                    acc += dl * Float32(KVALUES_IQ4NL[(qs >> 4) + 0x01]) * x[base + t + 17]
                end
            end
        end
        y[j] = alpha * acc + (beta == 0.0f0 ? 0.0f0 : beta * y[j])
    end
    return y
end

function _qgemv_bytes!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    q,
    ggml_type::Int32,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32,
)
    length(y) == ne1 || error("qgemv: y length $(length(y)) ≠ ne1 $ne1")
    length(x) == ne0 || error("qgemv: x length $(length(x)) ≠ ne0 $ne0")
    if ggml_type == GGML_TYPE_F32
        return _qgemv_f32!(y, x, q, ne0, ne1, alpha, beta)
    elseif ggml_type == GGML_TYPE_IQ2_XS
        return _qgemv_iq2_xs!(y, x, q, ne0, ne1, alpha, beta)
    elseif ggml_type == GGML_TYPE_IQ3_XXS
        return _qgemv_iq3_xxs!(y, x, q, ne0, ne1, alpha, beta)
    elseif ggml_type == GGML_TYPE_IQ4_XS
        return _qgemv_iq4_xs!(y, x, q, ne0, ne1, alpha, beta)
    end
    error("qgemv: unsupported ggml type $ggml_type")
end

"""
    _qgemv_expert!(y, x, slice, e; alpha=1, beta=0)

`y = alpha * W_e * x + beta * y` with `W_e` the `(ne1, ne0)` expert matrix
stored as ggml `(ne0, ne1)` blocks. `slice` is `GgufQuantSlice` or host
`GpuQuantSlice`.
"""
function _qgemv_expert!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    s::Union{GgufQuantSlice, GpuQuantSlice},
    e::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
)
    return _qgemv_bytes!(
        y,
        x,
        _quant_expert_bytes(s, e),
        s.ggml_type,
        s.ne0,
        s.ne1,
        alpha,
        beta,
    )
end

_is_quant_expert_bank(s) = s isa GgufQuantSlice || s isa GpuQuantSlice

# Row-chunked host GEMV. Each chunk is the same kernel on a view of the rows
# [j0, j1]: q is shifted by whole rows and y by the same rows, so every output
# element is computed by exactly one task with the single-threaded summation
# order. Results are bit-identical for any chunk count.
const _QGEMV_MIN_ROWS = 8

function _qrow_bytes(ggml_type::Int32, ne0::Int)
    ggml_type == GGML_TYPE_F32 && return ne0 * 4
    return div(ne0, QK_K) * _iq_block_bytes(ggml_type)
end

function _qgemv_bytes_threaded!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    q,
    ggml_type::Int32,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32;
    nchunks::Int=min(Threads.nthreads(), max(1, cld(ne1, _QGEMV_MIN_ROWS))),
)
    length(y) == ne1 || error("qgemv: y length $(length(y)) ≠ ne1 $ne1")
    length(x) == ne0 || error("qgemv: x length $(length(x)) ≠ ne0 $ne0")
    nc = clamp(nchunks, 1, ne1)
    rb = _qrow_bytes(ggml_type, ne0)
    nq = length(q)
    p = _PROFILE[]
    if p !== nothing && p.window
        return _qgemv_bytes_threaded_timed!(p, y, x, q, ggml_type, ne0, ne1, alpha, beta, nc, rb, nq)
    end
    Threads.@threads for c in 1:nc
        _qgemv_chunk!(y, x, q, ggml_type, ne0, ne1, alpha, beta, nc, c, rb, nq)
    end
    return y
end

@inline function _qgemv_chunk!(y, x, q, ggml_type, ne0, ne1, alpha, beta, nc, c, rb, nq)
    j0 = div((c - 1) * ne1, nc) + 1
    j1 = div(c * ne1, nc)
    qc = view(q, ((j0 - 1) * rb + 1):nq)
    _qgemv_bytes!(view(y, j0:j1), x, qc, ggml_type, ne0, j1 - j0 + 1, alpha, beta)
    return nothing
end

# Profiled twin of the threaded GEMV: times each chunk on its own thread so
# the fork/join cost (wall minus the slowest chunk) is separated from the
# kernel work. Same chunks, same arithmetic.
function _qgemv_bytes_threaded_timed!(p, y, x, q, ggml_type, ne0, ne1, alpha, beta, nc, rb, nq)
    length(p.chunk_ns) >= nc || resize!(p.chunk_ns, nc)
    cns = p.chunk_ns
    t_wall = time_ns()
    Threads.@threads for c in 1:nc
        t0 = time_ns()
        _qgemv_chunk!(y, x, q, ggml_type, ne0, ne1, alpha, beta, nc, c, rb, nq)
        @inbounds cns[c] = Int64(time_ns() - t0)
    end
    wall = Int64(time_ns() - t_wall)
    crit = maximum(view(cns, 1:nc))
    p.sub_ns[:host_fork_join] += wall - crit
    p.sub_ns[:host_critical_path] += crit
    p.sub_ns[:host_cpu_work] += sum(view(cns, 1:nc))
    p.counters[:host_threaded_calls] += 1
    return y
end

"""
    _moe_host_experts!(yrow, xf, order, weights, n_used, gate_s, up_s, down_s, gacc, uacc, hact, ybuf)

Top-k SwiGLU for one token on the host, reading the mmap'd expert bytes
directly. `order[1:n_used]` are the selected expert ids and `weights[k]` the
already-normalized routing weight of `order[k]`. Experts are accumulated in
k order, so the result does not depend on the thread count.
"""
function _moe_host_experts!(
    yrow::AbstractVector{Float32},
    xf::AbstractVector{Float32},
    order,
    weights,
    n_used::Int,
    gate_s::GgufQuantSlice,
    up_s::GgufQuantSlice,
    down_s::GgufQuantSlice,
    gacc::AbstractVector{Float32},
    uacc::AbstractVector{Float32},
    hact::AbstractVector{Float32},
    ybuf::AbstractVector{Float32},
)
    fill!(yrow, 0.0f0)
    for k in 1:n_used
        e = order[k]
        w = weights[k]
        _qgemv_bytes_threaded!(
            gacc, xf, _quant_expert_bytes(gate_s, e), gate_s.ggml_type,
            gate_s.ne0, gate_s.ne1, 1.0f0, 0.0f0,
        )
        _qgemv_bytes_threaded!(
            uacc, xf, _quant_expert_bytes(up_s, e), up_s.ggml_type,
            up_s.ne0, up_s.ne1, 1.0f0, 0.0f0,
        )
        @inbounds for j in eachindex(hact)
            hact[j] = _cpu_silu(gacc[j]) * uacc[j]
        end
        _qgemv_bytes_threaded!(
            ybuf, hact, _quant_expert_bytes(down_s, e), down_s.ggml_type,
            down_s.ne0, down_s.ne1, w, 0.0f0,
        )
        @inbounds for i in eachindex(yrow)
            yrow[i] += ybuf[i]
        end
    end
    return yrow
end
