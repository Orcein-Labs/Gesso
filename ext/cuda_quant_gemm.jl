# CUDA fused quantized expert GEMV. One thread per output row, ggml
# (ne0, ne1) block layout. Tables are device copies of ggml-common.h.

const _CUDA_Q_TABLES = Ref{Any}(nothing)
const _CUDA_Q_SCRATCH = Ref{Any}(nothing)

function _cuda_q_tables()
    t = _CUDA_Q_TABLES[]
    t !== nothing && return t
    Inf_ = Gesso.Inference
    t = (
        iq2grid=CuArray(Inf_.IQ2XS_GRID),
        ksigns=CuArray(Inf_.KSIGNS_IQ2XS),
        kmask=CuArray(Inf_.KMASK_IQ2XS),
        iq3grid=CuArray(Inf_.IQ3XXS_GRID),
        kv4=CuArray(Inf_.KVALUES_IQ4NL),
    )
    _CUDA_Q_TABLES[] = t
    return t
end

function _cuda_q_scratch(nbytes::Int)
    s = _CUDA_Q_SCRATCH[]
    if s === nothing || length(s) < nbytes
        s = CUDA.zeros(UInt8, max(nbytes, 1))
        _CUDA_Q_SCRATCH[] = s
    end
    return s
end

@inline function _cu_u16(q, off0)
    return UInt16(q[off0 + 1]) | (UInt16(q[off0 + 2]) << 8)
end
@inline function _cu_u32(q, off0)
    return UInt32(q[off0 + 1]) |
           (UInt32(q[off0 + 2]) << 8) |
           (UInt32(q[off0 + 3]) << 16) |
           (UInt32(q[off0 + 4]) << 24)
end
@inline function _cu_f16(q, off0)
    return Float32(reinterpret(Float16, _cu_u16(q, off0)))
end
@inline function _cu_x(x, i, k)
    return size(x, 2) == 1 ? x[i, 1] : x[i, k]
end

function _cuda_qgemv_iq2_xs_kernel!(
    y, x, q, offsets, cols, ne0::Int32, ne1::Int32, alpha::Float32, beta::Float32, grid, ksigns, kmask,
)
    j0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x) - Int32(1)
    i = Int(blockIdx().y)
    k = Int(cols[i])
    j0 >= ne1 && return
    expert_off = offsets[i]
    nb = div(ne0, Int32(256))
    bsz = Int32(74)
    acc = 0.0f0
    col = expert_off + j0 * nb * bsz
    ib = Int32(0)
    while ib < nb
        boff = col + ib * bsz
        d = _cu_f16(q, Int(boff))
        xoff = ib * Int32(256)
        ib32 = Int32(0)
        while ib32 < Int32(8)
            sc = q[Int(boff + Int32(66) + ib32) + 1]
            db0 = d * (0.5f0 + Float32(sc & 0x0f)) * 0.25f0
            db1 = d * (0.5f0 + Float32(sc >> 4)) * 0.25f0
            l = Int32(0)
            while l < Int32(4)
                qs_off = boff + Int32(2) + (Int32(4) * ib32 + l) * Int32(2)
                qv = _cu_u16(q, Int(qs_off))
                grid64 = grid[Int(qv & 0x01ff) + 1]
                signs = ksigns[Int(qv >> 9) + 1]
                db = l < Int32(2) ? db0 : db1
                base = xoff + ib32 * Int32(32) + l * Int32(8)
                t = Int32(0)
                while t < Int32(8)
                    g = Float32((grid64 >> (8 * Int(t))) & 0xff)
                    sng = (signs & kmask[Int(t) + 1]) != 0x00 ? -1.0f0 : 1.0f0
                    acc += db * g * sng * _cu_x(x, Int(base + t) + 1, k)
                    t += Int32(1)
                end
                l += Int32(1)
            end
            ib32 += Int32(1)
        end
        ib += Int32(1)
    end
    dest = Int(j0) + 1
    yj = beta == 0.0f0 ? 0.0f0 : beta * y[dest, k]
    y[dest, k] = alpha * acc + yj
    return
end

function _cuda_qgemv_iq3_xxs_kernel!(
    y, x, q, offsets, cols, ne0::Int32, ne1::Int32, alpha::Float32, beta::Float32, grid, ksigns, kmask,
)
    j0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x) - Int32(1)
    i = Int(blockIdx().y)
    k = Int(cols[i])
    j0 >= ne1 && return
    expert_off = offsets[i]
    nb = div(ne0, Int32(256))
    bsz = Int32(98)
    acc = 0.0f0
    col = expert_off + j0 * nb * bsz
    ib = Int32(0)
    while ib < nb
        boff = col + ib * bsz
        d = _cu_f16(q, Int(boff))
        qs0 = boff + Int32(2)
        ss0 = qs0 + Int32(64)
        xoff = ib * Int32(256)
        ib32 = Int32(0)
        while ib32 < Int32(8)
            aux32 = _cu_u32(q, Int(ss0 + Int32(4) * ib32))
            db = d * (0.5f0 + Float32(aux32 >> 28)) * 0.5f0
            qbase = qs0 + ib32 * Int32(8)
            l = Int32(0)
            while l < Int32(4)
                signs = ksigns[Int((aux32 >> (7 * Int(l))) & UInt32(127)) + 1]
                g1 = grid[Int(q[Int(qbase + Int32(2) * l) + 1]) + 1]
                g2 = grid[Int(q[Int(qbase + Int32(2) * l + Int32(1)) + 1]) + 1]
                base = xoff + ib32 * Int32(32) + l * Int32(8)
                t = Int32(0)
                while t < Int32(4)
                    s0 = (signs & kmask[Int(t) + 1]) != 0x00 ? -1.0f0 : 1.0f0
                    s1 = (signs & kmask[Int(t) + 5]) != 0x00 ? -1.0f0 : 1.0f0
                    acc += db * Float32((g1 >> (8 * Int(t))) & 0xff) * s0 * _cu_x(x, Int(base + t) + 1, k)
                    acc += db * Float32((g2 >> (8 * Int(t))) & 0xff) * s1 * _cu_x(x, Int(base + t) + 5, k)
                    t += Int32(1)
                end
                l += Int32(1)
            end
            ib32 += Int32(1)
        end
        ib += Int32(1)
    end
    dest = Int(j0) + 1
    yj = beta == 0.0f0 ? 0.0f0 : beta * y[dest, k]
    y[dest, k] = alpha * acc + yj
    return
end

function _cuda_qgemv_iq4_xs_kernel!(
    y, x, q, offsets, cols, ne0::Int32, ne1::Int32, alpha::Float32, beta::Float32, kv4,
)
    j0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x) - Int32(1)
    i = Int(blockIdx().y)
    k = Int(cols[i])
    j0 >= ne1 && return
    expert_off = offsets[i]
    nb = div(ne0, Int32(256))
    bsz = Int32(136)
    acc = 0.0f0
    col = expert_off + j0 * nb * bsz
    ibk = Int32(0)
    while ibk < nb
        boff = col + ibk * bsz
        d = _cu_f16(q, Int(boff))
        scales_h = _cu_u16(q, Int(boff + Int32(2)))
        xoff = ibk * Int32(256)
        qs0 = boff + Int32(8)
        ib = Int32(0)
        while ib < Int32(8)
            sl = q[Int(boff + Int32(4) + div(ib, Int32(2))) + 1]
            ls = Int((sl >> (4 * Int(ib % Int32(2)))) & 0x0f) | (Int((scales_h >> (2 * Int(ib))) & 0x0003) << 4)
            dl = d * Float32(ls - 32)
            qsb = qs0 + ib * Int32(16)
            base = xoff + ib * Int32(32)
            t = Int32(0)
            while t < Int32(16)
                qs = q[Int(qsb + t) + 1]
                acc += dl * Float32(kv4[Int(qs & 0x0f) + 1]) * _cu_x(x, Int(base + t) + 1, k)
                acc += dl * Float32(kv4[Int(qs >> 4) + 1]) * _cu_x(x, Int(base + t) + 17, k)
                t += Int32(1)
            end
            ib += Int32(1)
        end
        ibk += Int32(1)
    end
    dest = Int(j0) + 1
    yj = beta == 0.0f0 ? 0.0f0 : beta * y[dest, k]
    y[dest, k] = alpha * acc + yj
    return
end

function _cuda_qgemv_f32_kernel!(y, x, q, offsets, cols, ne0::Int32, ne1::Int32, alpha::Float32, beta::Float32)
    j0 = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x) - Int32(1)
    i = Int(blockIdx().y)
    k = Int(cols[i])
    j0 >= ne1 && return
    expert_off = offsets[i]
    acc = 0.0f0
    base = expert_off + j0 * ne0 * Int32(4)
    i = Int32(0)
    while i < ne0
        off = base + i * Int32(4)
        u = _cu_u32(q, Int(off))
        acc += reinterpret(Float32, u) * _cu_x(x, Int(i) + 1, k)
        i += Int32(1)
    end
    dest = Int(j0) + 1
    yj = beta == 0.0f0 ? 0.0f0 : beta * y[dest, k]
    y[dest, k] = alpha * acc + yj
    return
end

const _CUDA_Q_OFFSETS = Ref{Any}(nothing)
const _CUDA_Q_HOSTPACK = Ref{Vector{UInt8}}(UInt8[])
const _CUDA_Q_HOSTOFF = Ref{Vector{Int32}}(Int32[])

function _cuda_q_offsets(n::Int)
    s = _CUDA_Q_OFFSETS[]
    if s === nothing || length(s) < n
        s = CUDA.zeros(Int32, max(n, 8))
        _CUDA_Q_OFFSETS[] = s
    end
    return s
end

# Identity column map 1:n (launch i writes rank i). Read-only: a prefix of it
# is valid for any n up to its length. Never written after it is built.
const _CUDA_Q_IDENT = Ref{Any}(nothing)
# Per-launch column map for the cached path (launch i writes rank cols[i]).
const _CUDA_Q_MAPCOLS = Ref{Any}(nothing)

function _cuda_q_ident(n::Int)
    s = _CUDA_Q_IDENT[]
    if s === nothing || length(s) < n
        s = CUDA.CuArray(Int32.(1:max(n, 8)))
        _CUDA_Q_IDENT[] = s
    end
    return s
end

function _cuda_q_mapcols(cols::AbstractVector{Int32})
    n = length(cols)
    s = _CUDA_Q_MAPCOLS[]
    if s === nothing || length(s) < n
        s = CUDA.zeros(Int32, max(n, 8))
        _CUDA_Q_MAPCOLS[] = s
    end
    copyto!(s, 1, cols, 1, n)
    return s
end

function _cuda_launch_qgemv!(
    y::CuMatrix{Float32},
    x::CuArray{Float32},
    q::CuVector{UInt8},
    ggml_type::Int32,
    offsets::CuVector{Int32},
    cols::CuVector{Int32},
    n_used::Int,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32,
)
    tables = _cuda_q_tables()
    nthreads = min(256, ne1)
    nblocks = cld(ne1, nthreads)
    n0 = Int32(ne0)
    n1 = Int32(ne1)
    if ggml_type == Gesso.Inference.GGML_TYPE_IQ2_XS
        @cuda threads = nthreads blocks = (nblocks, n_used) _cuda_qgemv_iq2_xs_kernel!(
            y, x, q, offsets, cols, n0, n1, alpha, beta, tables.iq2grid, tables.ksigns, tables.kmask,
        )
    elseif ggml_type == Gesso.Inference.GGML_TYPE_IQ3_XXS
        @cuda threads = nthreads blocks = (nblocks, n_used) _cuda_qgemv_iq3_xxs_kernel!(
            y, x, q, offsets, cols, n0, n1, alpha, beta, tables.iq3grid, tables.ksigns, tables.kmask,
        )
    elseif ggml_type == Gesso.Inference.GGML_TYPE_IQ4_XS
        @cuda threads = nthreads blocks = (nblocks, n_used) _cuda_qgemv_iq4_xs_kernel!(
            y, x, q, offsets, cols, n0, n1, alpha, beta, tables.kv4,
        )
    elseif ggml_type == Gesso.Inference.GGML_TYPE_F32
        @cuda threads = nthreads blocks = (nblocks, n_used) _cuda_qgemv_f32_kernel!(
            y, x, q, offsets, cols, n0, n1, alpha, beta,
        )
    else
        error("cuda qgemv: unsupported ggml type $ggml_type")
    end
    return y
end

function _cuda_launch_qgemv!(
    y::CuVector{Float32},
    x::CuVector{Float32},
    q::CuVector{UInt8},
    ggml_type::Int32,
    expert_off::Int,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32,
)
    offv = _cuda_q_offsets(1)
    copyto!(offv, 1, Int32[Int32(expert_off)], 1, 1)
    return _cuda_launch_qgemv!(
        reshape(y, ne1, 1),
        reshape(x, ne0, 1),
        q,
        ggml_type,
        offv,
        _cuda_q_ident(1),
        1,
        ne0,
        ne1,
        alpha,
        beta,
    )
end

function _cuda_bind_experts!(bank, experts::AbstractVector{Int})
    Inf_ = Gesso.Inference
    n_used = length(experts)
    nbytes = bank.bytes_per_expert
    hostoff = _CUDA_Q_HOSTOFF[]
    length(hostoff) >= n_used || (hostoff = Vector{Int32}(undef, n_used); _CUDA_Q_HOSTOFF[] = hostoff)
    if bank isa Inf_.GpuQuantSlice
        @inbounds for k in 1:n_used
            hostoff[k] = Int32(Inf_._quant_expert_off0(bank, experts[k]))
        end
        offv = _cuda_q_offsets(n_used)
        copyto!(offv, 1, hostoff, 1, n_used)
        return bank.data, offv
    elseif bank isa Inf_.GgufQuantSlice
        total = nbytes * n_used
        hostpack = _CUDA_Q_HOSTPACK[]
        length(hostpack) >= total || (hostpack = Vector{UInt8}(undef, total); _CUDA_Q_HOSTPACK[] = hostpack)
        timed = Inf_._profile_window_open()
        t_gather = timed ? time_ns() : UInt64(0)
        @inbounds for k in 1:n_used
            off = Inf_._quant_expert_off0(bank, experts[k])
            copyto!(hostpack, (k - 1) * nbytes + 1, bank.data, off + 1, nbytes)
            hostoff[k] = Int32((k - 1) * nbytes)
        end
        timed && Inf_._profile_sub!(:upload_gather, time_ns() - t_gather)
        scratch = _cuda_q_scratch(total)
        Inf_._profile_copy!(false)
        Inf_._profile_count!(:expert_h2d_bytes, total)
        CUDA.unsafe_copyto!(pointer(scratch), pointer(hostpack), total)
        offv = _cuda_q_offsets(n_used)
        Inf_._profile_copy!(false)
        copyto!(offv, 1, hostoff, 1, n_used)
        return scratch, offv
    else
        error("cuda qgemv: not a quantized expert bank $(typeof(bank))")
    end
end

function _cuda_qgemv_experts!(
    y::CuMatrix{Float32},
    x::CuArray{Float32},
    bank,
    experts::Vector{Int};
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
)
    n_used = length(experts)
    q, offv = Gesso.Inference.@decode_stage :moe_qupload _CUDASync() _cuda_bind_experts!(bank, experts)
    xx = ndims(x) == 1 ? reshape(x, length(x), 1) : x
    return Gesso.Inference.@decode_stage :moe_qgemv _CUDASync() _cuda_launch_qgemv!(
        y, xx, q, bank.ggml_type, offv, _cuda_q_ident(n_used), n_used, bank.ne0, bank.ne1, alpha, beta,
    )
end

# Device-side bind for one projection of a cached layer.
#
# Hits: resident experts already sit in the layer's slot pool, so their launch
# only receives slot byte offsets. Misses: the existing gather-and-upload of
# the missed experts (`_cuda_bind_experts!`) runs for those experts alone.
# This is the one miss-path entry point; a later CPU-side miss compute replaces
# this function, not the caller.
function _cuda_cached_project!(
    out::CuMatrix{Float32},
    xin::CuMatrix{Float32},
    layer::Gesso.Inference.ExpertLayerCache,
    role::Int,
    hit_rank::Vector{Int32},
    hit_slot::Vector{Int32},
    nh::Int,
    miss_rank::Vector{Int32},
    miss_exp::Vector{Int},
    nm::Int,
)
    Inf_ = Gesso.Inference
    bank = layer.banks[role]
    bpe = Int32(bank.bytes_per_expert)
    if nh > 0
        offs = Int32[(hit_slot[i] - Int32(1)) * bpe for i in 1:nh]
        offv = _cuda_q_offsets(nh)
        copyto!(offv, 1, offs, 1, nh)
        colv = _cuda_q_mapcols(hit_rank[1:nh])
        Inf_._profile_copy!(false)
        Gesso.Inference.@decode_stage :moe_qgemv _CUDASync() _cuda_launch_qgemv!(
            out, xin, layer.pools[role], bank.ggml_type, offv, colv, nh, bank.ne0, bank.ne1, 1.0f0, 0.0f0,
        )
    end
    if nm > 0
        q, offv = Gesso.Inference.@decode_stage :moe_qupload _CUDASync() _cuda_bind_experts!(bank, miss_exp[1:nm])
        colv = _cuda_q_mapcols(miss_rank[1:nm])
        Gesso.Inference.@decode_stage :moe_qgemv _CUDASync() _cuda_launch_qgemv!(
            out, xin, q, bank.ggml_type, offv, colv, nm, bank.ne0, bank.ne1, 1.0f0, 0.0f0,
        )
    end
    return out
end

function Gesso.Inference._qgemv_expert!(
    y::CuVector{Float32},
    x::CuVector{Float32},
    bank::Union{Gesso.Inference.GgufQuantSlice, Gesso.Inference.GpuQuantSlice},
    e::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
)
    return _cuda_qgemv_expert!(y, x, bank, e; alpha=alpha, beta=beta)
end

function _cuda_qgemv_expert!(
    y::CuVector{Float32},
    x::CuVector{Float32},
    bank,
    e::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
)
    Inf_ = Gesso.Inference
    if bank isa Inf_.GpuQuantSlice
        off = Inf_._quant_expert_off0(bank, e)
        return _cuda_launch_qgemv!(y, x, bank.data, bank.ggml_type, off, bank.ne0, bank.ne1, alpha, beta)
    elseif bank isa Inf_.GgufQuantSlice
        nbytes = bank.bytes_per_expert
        scratch = _cuda_q_scratch(nbytes)
        off = Inf_._quant_expert_off0(bank, e)
        GC.@preserve bank begin
            CUDA.unsafe_copyto!(pointer(scratch), pointer(bank.data, off + 1), nbytes)
        end
        return _cuda_launch_qgemv!(y, x, scratch, bank.ggml_type, 0, bank.ne0, bank.ne1, alpha, beta)
    else
        error("cuda qgemv: not a quantized expert bank $(typeof(bank))")
    end
end
