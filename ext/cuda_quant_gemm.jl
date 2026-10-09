# CUDA fused quantized expert GEMV. Tables are device copies of ggml-common.h.
#
# The IQ quant types use the same integer scheme as the dense path
# (cuda_quant_dense.jl): the activation column is quantized once per launch to
# 32-value Q8_1-style blocks, and each 32-value weight block is one dp4a dot
# product with its ggml grid and sign lanes unpacked into int8 words. The
# F32 expert kernel stays naive: it has no quantized weight to feed dp4a.
#
# Expert launches read one metadata vector: [byte offset of each expert slab
# (1:n); activation column of each launch (n+1:2n)]. It is uploaded with one
# copy per launch, so the offsets and the column map cost one sync, not four.

# The caches are concretely typed: a `Ref{Any}` here made every launch site
# that reads them dispatch at run time, a few microseconds per op.
const _QTables = NamedTuple{(:iq2grid, :ksigns, :kmask, :iq3grid, :kv4),Tuple{
    _CuVec{eltype(Gesso.Inference.IQ2XS_GRID)},_CuVec{eltype(Gesso.Inference.KSIGNS_IQ2XS)},
    _CuVec{eltype(Gesso.Inference.KMASK_IQ2XS)},_CuVec{eltype(Gesso.Inference.IQ3XXS_GRID)},
    _CuVec{eltype(Gesso.Inference.KVALUES_IQ4NL)}}}
const _Q8Bufs = Tuple{_CuVec{Int32},_CuVec{Float32},_CuVec{Int32}}

const _CUDA_Q_TABLES = Ref{Union{Nothing,_QTables}}(nothing)
const _CUDA_Q_SCRATCH = Ref{Union{Nothing,_CuVec{UInt8}}}(nothing)
# Activation Q8 buffers for expert launches. Kept apart from the dense slot so
# that a dense matmul never reallocates the expert buffers.
const _CUDA_Q8_EXPERT = Ref{Union{Nothing,_Q8Bufs}}(nothing)
const _CUDA_Q8_DENSE = Ref{Union{Nothing,_Q8Bufs}}(nothing)
const _CUDA_Q_META = Ref{Union{Nothing,_CuVec{Int32}}}(nothing)
const _CUDA_Q_META_HOST = Ref{Vector{Int32}}(Int32[])
const _CUDA_Q_HOSTPACK = Ref{Vector{UInt8}}(UInt8[])
const _CUDA_Q_HOSTOFF = Ref{Vector{Int32}}(Int32[])

function _cuda_q_tables()::_QTables
    t = _CUDA_Q_TABLES[]
    t !== nothing && return t
    Inf_ = Gesso.Inference
    # Grid bytes become signed int8 lanes of a dp4a operand. A byte above 127
    # would read back negative, so the table is checked before it is uploaded.
    for g in Inf_.IQ2XS_GRID, tt in 0:7
        ((g >> (8 * tt)) & 0xff) <= 0x7f || error("IQ2_XS grid byte above 127 breaks the int8 lane")
    end
    for g in Inf_.IQ3XXS_GRID, tt in 0:3
        ((g >> (8 * tt)) & 0xff) <= 0x7f || error("IQ3_XXS grid byte above 127 breaks the int8 lane")
    end
    for i in 0:127
        Inf_.KSIGNS_IQ2XS[i + 1] == UInt8(i | ((count_ones(i) & 1) << 7)) ||
            error("ksigns no longer the parity expansion the IQ decoders compute")
    end
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

# Activation buffers for `nblocks` 32-value blocks: (qx, dx, sx). Grown on
# demand and reused, so a steady decode allocates nothing per call.
function _cuda_q8_buffers!(slot::Ref{Union{Nothing,_Q8Bufs}}, nblocks::Int)::_Q8Bufs
    s = slot[]
    if s === nothing || length(s[2]) < nblocks
        n = max(nblocks, 1)
        s = (CUDA.zeros(Int32, 8 * n), CUDA.zeros(Float32, n), CUDA.zeros(Int32, n))
        slot[] = s
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

# Aligned loads straight from the device pointer. Assembling a word from four
# checked byte loads costs four load instructions plus the checks, and that,
# not DRAM, bounded the GEMV kernels. The caller guarantees `off0` is a
# multiple of the width; slab bases are allocation-aligned, which the launch
# sites check.
@inline _cu_ptr_as(::Type{T}, p::Core.LLVMPtr{UInt8,A}) where {T,A} = reinterpret(Core.LLVMPtr{T,A}, p)
@inline _cu_u32a(q, off0) = unsafe_load(_cu_ptr_as(UInt32, pointer(q) + off0), 1, Val(4))
@inline _cu_u16a(q, off0) = unsafe_load(_cu_ptr_as(UInt16, pointer(q) + off0), 1, Val(2))
# A 32-bit word at a 2-aligned offset: block sizes 210, 98, 74 and 34 leave
# every other block on a half-word boundary.
@inline _cu_u32h(q, off0) = UInt32(_cu_u16a(q, off0)) | (UInt32(_cu_u16a(q, off0 + 2)) << 16)
@inline _cu_f16a(q, off0) = Float32(reinterpret(Float16, _cu_u16a(q, off0)))

_cu_aligned(a, n::Int) = UInt(pointer(a)) % n == 0

@inline function _cu_x(x, i, k)
    return size(x, 2) == 1 ? x[i, 1] : x[i, k]
end

# --- activation quantize, one thread per (column, 32-value block) -------------

function _cuda_q8_1_quant_cols_kernel!(qx, dx, X, nbk::Int, total::Int)
    g = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    g > total && return
    gb = g - 1
    k = gb ÷ nbk + 1
    base = (gb % nbk) * 32
    @inbounds begin
        amax = 0.0f0
        for l in 1:32
            amax = max(amax, abs(X[base + l, k]))
        end
        inv = amax == 0.0f0 ? 0.0f0 : 127.0f0 / amax
        for w in 0:7
            u = UInt32(0)
            for tt in 0:3
                v = X[base + 4 * w + tt + 1, k] * inv
                qi = unsafe_trunc(Int32, v + copysign(0.5f0, v))
                qi = clamp(qi, Int32(-127), Int32(127))
                u |= UInt32(qi & 0xff) << (8 * tt)
            end
            qx[gb * 8 + w + 1] = reinterpret(Int32, u)
        end
        dx[g] = amax / 127.0f0
    end
    return
end

# --- per-block contributions for the IQ types --------------------------------

# Sign handling without the ksigns and kmask tables. A 7-bit sign index
# expands to 8 sign bits by adding an even-parity bit 7 (the table check at
# upload holds ksigns to this). Four sign bits become 0x00/0xff byte lanes: the
# multiply copies the nibble to shifts 0, 7, 14 and 21, which never overlap, so
# bit i lands on bit 8i with no carry.
@inline _iq_signs8(i::UInt32) = (i & 0x7f) | ((UInt32(count_ones(i & 0x7f)) & 0x1) << 7)
@inline _iq_mask4(s::UInt32) = (((s & 0x0f) * 0x00204081) & 0x01010101) * 0x000000ff
# dp4a of x with the grid lanes of g negated where the sign bit is set. For a
# grid byte g in 0..127, int8(g xor 0xff) is -g - 1; the first dp4a adds the 1
# back on the negated lanes. Integer sums, so the result is exact.
@inline function _iq_signed_dp4a(g::UInt32, s4::UInt32, x::Int32, acc::Int32)
    m = _iq_mask4(s4)
    acc = CUDA.dp4a(reinterpret(Int32, m & 0x01010101), x, acc)
    return CUDA.dp4a(reinterpret(Int32, g ⊻ m), x, acc)
end

# `boff` is the byte offset of the superblock, `b` the 32-value block inside the
# row (its position in the superblock is b & 7), `gb` the activation block.

# IQ2_XS (74 B): f16 d, 32 x u16 (9-bit grid index, 7-bit sign index), 8 scale
# bytes. Two 4-bit scales per 32-value block: the low nibble covers elements
# l = 0,1 of the block, the high nibble l = 2,3. Each l covers 8 values taken
# from one 8-byte grid entry, signed by bits of the expanded sign index.
@inline function _eq_iq2xs(Q, boff::Int, b::Int, gb::Int, qx, dx, grid, ksigns, kmask)
    @inbounds begin
        ib32 = b & 7
        d = _cu_f16a(Q, boff)
        sc = Q[boff + 66 + ib32 + 1]
        db0 = d * (0.5f0 + Float32(sc & 0x0f)) * 0.25f0
        db1 = d * (0.5f0 + Float32(sc >> 4)) * 0.25f0
        s_lo = Int32(0)
        s_hi = Int32(0)
        for l in 0:3
            qv = UInt32(_cu_u16a(Q, boff + 2 + (4 * ib32 + l) * 2))
            g64 = grid[Int(qv & 0x01ff) + 1]
            sg = _iq_signs8(qv >> 9)
            p = _iq_signed_dp4a(g64 % UInt32, sg, qx[gb * 8 + 2 * l + 1], Int32(0))
            p = _iq_signed_dp4a(UInt32(g64 >> 32), sg >> 4, qx[gb * 8 + 2 * l + 2], p)
            if l < 2
                s_lo += p
            else
                s_hi += p
            end
        end
        return dx[gb + 1] * (db0 * Float32(s_lo) + db1 * Float32(s_hi))
    end
end

# IQ3_XXS (98 B): f16 d, 64 grid bytes (qs), then 8 x u32 aux (4-bit scale in
# bits 28..31, four 7-bit sign indices). Each l takes two 4-byte grid entries:
# the first covers values 0..3 of the l-th group, the second 4..7.
@inline function _eq_iq3xxs(Q, boff::Int, b::Int, gb::Int, qx, dx, grid, ksigns, kmask)
    @inbounds begin
        ib32 = b & 7
        d = _cu_f16a(Q, boff)
        aux32 = _cu_u32h(Q, boff + 2 + 64 + 4 * ib32)
        db = d * (0.5f0 + Float32(aux32 >> 28)) * 0.5f0
        qbase = boff + 2 + 8 * ib32
        isum = Int32(0)
        for l in 0:3
            sg = _iq_signs8(aux32 >> (7 * l))
            qq = _cu_u16a(Q, qbase + 2 * l)
            for h in 0:1
                gv = grid[Int((qq >> (8 * h)) & 0x00ff) + 1]
                isum = _iq_signed_dp4a(gv, sg >> (4 * h), qx[gb * 8 + 2 * l + h + 1], isum)
            end
        end
        return dx[gb + 1] * db * Float32(isum)
    end
end

@inline function _eq_block(::Val{:iq2_xs}, Q, boff::Int, b::Int, gb::Int, qx, dx, grid, kv, ksigns, kmask)
    return _eq_iq2xs(Q, boff, b, gb, qx, dx, grid, ksigns, kmask)
end
@inline function _eq_block(::Val{:iq3_xxs}, Q, boff::Int, b::Int, gb::Int, qx, dx, grid, kv, ksigns, kmask)
    return _eq_iq3xxs(Q, boff, b, gb, qx, dx, grid, ksigns, kmask)
end
@inline function _eq_block(::Val{:iq4_xs}, Q, boff::Int, b::Int, gb::Int, qx, dx, grid, kv, ksigns, kmask)
    return _dq_iq4xs(Q, boff, b, gb, qx, dx, kv)
end

# --- expert MMVQ kernel: one warp per output row, four rows per block --------
# grid = (rows/4, n_used). Launch i reads expert slab meta[i] and writes column
# meta[coloff + i] of y; the activation column is shared when x has one column.

function _cuda_eqgemv_kernel!(
    y, Q, qx, dx, meta, coloff::Int, xshared::Bool, nrows::Int, nbk::Int, bsz::Int, row_bytes::Int,
    alpha::Float32, beta::Float32, ::Val{K}, grid, kv, ksigns, kmask,
) where {K}
    lane = Int((threadIdx().x - 1) % 32)
    row = Int((blockIdx().x - 1) * 4 + (threadIdx().x - 1) ÷ 32 + 1)
    i = Int(blockIdx().y)
    row > nrows && return
    @inbounds begin
        k = Int(meta[coloff + i])
        gbase = (xshared ? 0 : (k - 1) * nbk)
        rowoff = Int(meta[i]) + (row - 1) * row_bytes
        acc = 0.0f0
        b = lane
        while b < nbk
            acc += _eq_block(Val{K}(), Q, rowoff + (b >> 3) * bsz, b, gbase + b, qx, dx, grid, kv, ksigns, kmask)
            b += 32
        end
        off = 16
        while off >= 1
            acc += CUDA.shfl_down_sync(0xffffffff, acc, off)
            off >>= 1
        end
        if lane == 0
            yv = beta == 0.0f0 ? 0.0f0 : beta * y[row, k]
            y[row, k] = alpha * acc + yv
        end
    end
    return
end

# F32 experts stay naive: one thread per output row, reading the slab directly.
function _cuda_qgemv_f32_kernel!(y, x, q, meta, coloff::Int, ne0::Int, ne1::Int, alpha::Float32, beta::Float32)
    j0 = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x) - 1
    i = Int(blockIdx().y)
    j0 >= ne1 && return
    k = Int(meta[coloff + i])
    base = Int(meta[i]) + j0 * ne0 * 4
    acc = 0.0f0
    c = 0
    while c < ne0
        u = _cu_u32(q, base + c * 4)
        acc += reinterpret(Float32, u) * _cu_x(x, c + 1, k)
        c += 1
    end
    dest = j0 + 1
    yj = beta == 0.0f0 ? 0.0f0 : beta * y[dest, k]
    y[dest, k] = alpha * acc + yj
    return
end

# Metadata upload: one synchronous copy of [offsets; columns] per launch. `cols`
# nothing means launch i writes column i.
function _cuda_meta_upload(offs::AbstractVector{Int32}, cols, n::Int)
    host = _CUDA_Q_META_HOST[]
    length(host) >= 2n || (host = Vector{Int32}(undef, 2n); _CUDA_Q_META_HOST[] = host)
    copyto!(host, 1, offs, 1, n)
    if cols === nothing
        @inbounds for i in 1:n
            host[n + i] = Int32(i)
        end
    else
        copyto!(host, n + 1, cols, 1, n)
    end
    dev = _CUDA_Q_META[]
    if dev === nothing || length(dev) < 2n
        dev = CUDA.zeros(Int32, max(2n, 8))
        _CUDA_Q_META[] = dev
    end
    copyto!(dev, 1, host, 1, 2n)
    return dev
end

function _cuda_launch_qgemv!(
    y::CuMatrix{Float32},
    x::CuArray{Float32},
    q::CuVector{UInt8},
    ggml_type::Int32,
    meta::CuVector{Int32},
    coloff::Int,
    n_used::Int,
    ne0::Int,
    ne1::Int,
    alpha::Float32,
    beta::Float32;
    quantize::Bool=true,
)
    I = Gesso.Inference
    if ggml_type == I.GGML_TYPE_F32
        nthreads = min(256, ne1)
        nblocks = cld(ne1, nthreads)
        @fastcuda threads = nthreads blocks = (nblocks, n_used) _cuda_qgemv_f32_kernel!(
            y, x, q, meta, coloff, ne0, ne1, alpha, beta,
        )
        return y
    end
    ggml_type in (I.GGML_TYPE_IQ2_XS, I.GGML_TYPE_IQ3_XXS, I.GGML_TYPE_IQ4_XS) ||
        error("cuda qgemv: unsupported ggml type $ggml_type")
    _cu_aligned(q, 4) || error("cuda qgemv: expert bytes not 4-byte aligned")
    tables = _cuda_q_tables()
    nbk = ne0 ÷ 32
    ncols = size(x, 2)
    total = ncols * nbk
    qx, dx, _ = _cuda_q8_buffers!(_CUDA_Q8_EXPERT, total)
    # quantize=false: the caller just quantized this same x into these buffers
    quantize && @fastcuda threads = 256 blocks = cld(total, 256) _cuda_q8_1_quant_cols_kernel!(qx, dx, x, nbk, total)
    # One launch per type so each `Val` is a compile-time constant.
    args = (y, q, qx, dx, meta, coloff, ncols == 1, ne1, nbk, alpha, beta, n_used, tables)
    if ggml_type == I.GGML_TYPE_IQ2_XS
        _cuda_eqgemv_launch!(Val(:iq2_xs), 74, tables.iq2grid, args...)
    elseif ggml_type == I.GGML_TYPE_IQ3_XXS
        _cuda_eqgemv_launch!(Val(:iq3_xxs), 98, tables.iq3grid, args...)
    else
        _cuda_eqgemv_launch!(Val(:iq4_xs), 136, tables.kv4, args...)
    end
    return y
end

@inline function _cuda_eqgemv_launch!(vk::Val, bsz, grid, y, q, qx, dx, meta, coloff, onecol,
                                      ne1, nbk, alpha, beta, n_used, tables)
    @fastcuda threads = 128 blocks = (cld(ne1, 4), n_used) _cuda_eqgemv_kernel!(
        y, q, qx, dx, meta, coloff, onecol, ne1, nbk, bsz, (nbk ÷ 8) * bsz, alpha, beta,
        vk, grid, tables.kv4, tables.ksigns, tables.kmask,
    )
    return nothing
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
    meta = _cuda_meta_upload(Int32[Int32(expert_off)], Int32[1], 1)
    return _cuda_launch_qgemv!(
        reshape(y, ne1, 1),
        reshape(x, ne0, 1),
        q,
        ggml_type,
        meta,
        1,
        1,
        ne0,
        ne1,
        alpha,
        beta,
    )
end

# Device bytes and metadata for `experts`. `ranks` maps launch i to its output
# column; nothing means column i.
function _cuda_bind_experts!(bank, experts::AbstractVector{Int}, ranks)
    Inf_ = Gesso.Inference
    n_used = length(experts)
    nbytes = bank.bytes_per_expert
    hostoff = _CUDA_Q_HOSTOFF[]
    length(hostoff) >= n_used || (hostoff = Vector{Int32}(undef, n_used); _CUDA_Q_HOSTOFF[] = hostoff)
    if bank isa Inf_.GpuQuantSlice
        @inbounds for k in 1:n_used
            hostoff[k] = Int32(Inf_._quant_expert_off0(bank, experts[k]))
        end
        return bank.data, _cuda_meta_upload(hostoff, ranks, n_used)
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
        Inf_._profile_copy!(false)
        return scratch, _cuda_meta_upload(hostoff, ranks, n_used)
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
    q, meta = Gesso.Inference.@decode_stage :moe_qupload _CUDASync() _cuda_bind_experts!(bank, experts, nothing)
    xx = ndims(x) == 1 ? reshape(x, length(x), 1) : x
    return Gesso.Inference.@decode_stage :moe_qgemv _CUDASync() _cuda_launch_qgemv!(
        y, xx, q, bank.ggml_type, meta, n_used, n_used, bank.ne0, bank.ne1, alpha, beta,
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
        meta = _cuda_meta_upload(offs, hit_rank[1:nh], nh)
        Inf_._profile_copy!(false)
        Gesso.Inference.@decode_stage :moe_qgemv _CUDASync() _cuda_launch_qgemv!(
            out, xin, layer.pools[role], bank.ggml_type, meta, nh, nh, bank.ne0, bank.ne1, 1.0f0, 0.0f0,
        )
    end
    if nm > 0
        q, meta = Gesso.Inference.@decode_stage :moe_qupload _CUDASync() _cuda_bind_experts!(
            bank, miss_exp[1:nm], miss_rank[1:nm],
        )
        Gesso.Inference.@decode_stage :moe_qgemv _CUDASync() _cuda_launch_qgemv!(
            out, xin, q, bank.ggml_type, meta, nm, nm, bank.ne0, bank.ne1, 1.0f0, 0.0f0,
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
