# Lava quant GEMV: the GGUF bytes of a projection or an expert bank stay on the
# device and the matmul dequantizes them inside the kernel. The CUDA path
# (cuda_quant_dense.jl, cuda_quant_gemm.jl) quantizes the activation to Q8_1 and
# runs an integer dot with dp4a; Lava has no dp4a, and this port keeps the
# activation in F32, so every product is exact up to F32 accumulation. The
# weight decode follows the same GGML block layouts as the CUDA kernels.

using Lava.KernelAbstractions: @kernel, @index, @localmem, @synchronize

include("gguf_elem_decode.jl")

# Lanes per output row. Fixed at compile time: Lava bakes the SPIR-V
# LocalSize from workgroupsize, and a shape-dependent size would recompile
# the pipeline for every projection.
const _LQ_LANES = 64

# The IQ4_NL codebook, shared by IQ4_NL and IQ4_XS (ggml kvalues_iq4nl).
const _LQ_KVALUES_IQ4NL =
    Float32[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

# Device-resident dense quant weight, in one of two layouts. LavaBlockQuant keeps
# the GGUF bytes: `data` holds rows 1..nrows in block order, row r starting at byte
# (r-1)*row_bytes. LavaQ8Quant is the repacked Q8_0 layout defined below.
abstract type LavaDenseQuant end

struct LavaBlockQuant <: LavaDenseQuant
    data::Lava.LavaArray{UInt8,1}
    ggml_type::Int32
    kind::Int
    ne0::Int
    nrows::Int
    row_bytes::Int
end

Base.size(q::LavaDenseQuant) = (q.nrows, q.ne0)
Base.size(q::LavaDenseQuant, d::Integer) = size(q)[d]
Base.ndims(::LavaDenseQuant) = 2
Base.length(q::LavaDenseQuant) = q.nrows * q.ne0
Base.sizeof(q::LavaBlockQuant) = sizeof(q.data)

# Expert banks take the K-quants Q2_K..Q6_K as well as the IQ and half types.
# The CUDA expert launcher has no Q5_K or Q6_K kernel, so Lava is wider there;
# both decoders are the ones the dense path already uses.
function _lq_expert_kind(t::Int32)
    I = Gesso.Inference
    t == I.GGML_TYPE_F32 && return _LQ_KIND_F32
    t == I.GGML_TYPE_BF16 && return _LQ_KIND_BF16
    t == I.GGML_TYPE_F16 && return _LQ_KIND_F16
    t == I.GGML_TYPE_Q2_K && return _LQ_KIND_Q2_K
    t == I.GGML_TYPE_Q3_K && return _LQ_KIND_Q3_K
    t == I.GGML_TYPE_Q4_K && return _LQ_KIND_Q4_K
    t == I.GGML_TYPE_Q5_K && return _LQ_KIND_Q5_K
    t == I.GGML_TYPE_Q6_K && return _LQ_KIND_Q6_K
    t == I.GGML_TYPE_IQ2_XS && return _LQ_KIND_IQ2_XS
    t == I.GGML_TYPE_IQ3_XXS && return _LQ_KIND_IQ3_XXS
    t == I.GGML_TYPE_IQ4_XS && return _LQ_KIND_IQ4_XS
    t == I.GGML_TYPE_IQ4_NL && return _LQ_KIND_IQ4_NL
    t == I.GGML_TYPE_Q2_0 && return _LQ_KIND_Q2_0
    t == I.GGML_TYPE_IQ2_XXS && return _LQ_KIND_IQ2_XXS
    t == I.GGML_TYPE_IQ2_S && return _LQ_KIND_IQ2_S
    t == I.GGML_TYPE_IQ3_S && return _LQ_KIND_IQ3_S
    return nothing
end

# Values per block for each kind: K-quant and IQ2/IQ3 superblocks hold 256,
# Q2_0 blocks 64, Q8_0 and IQ4_NL blocks 32. F32, BF16 and F16 are one value
# per element, so a row of any width is a whole number of blocks.
_lq_block_values(kind::Int) =
    kind == _LQ_KIND_F32 || kind == _LQ_KIND_BF16 || kind == _LQ_KIND_F16 ? 1 :
    kind == _LQ_KIND_Q8_0 || kind == _LQ_KIND_IQ4_NL ? 32 :
    kind == _LQ_KIND_Q2_0 ? 64 : 256

# Same rule as the CUDA side: a row is a whole number of blocks.
function _lq_eligible(s)
    s isa Gesso.Inference.QuantMatrix || return false
    kind = _lq_kind(s.ggml_type)
    kind === nothing && return false
    return s.ne0 % _lq_block_values(kind) == 0
end

# Q8_0 repacked is one int8 per value plus one F32 scale per 32 values: ne0 + ne0/8 bytes a row.
function _lq_device_bytes(s::Gesso.Inference.QuantMatrix)
    if s.ggml_type == Gesso.Inference.GGML_TYPE_Q8_0
        return Int64(s.nrows) * Int64(s.ne0 + s.ne0 ÷ 8)
    end
    return Int64(s.nrows) * Int64(s.row_bytes)
end

function _lq_upload(s::Gesso.Inference.QuantMatrix)
    bytes = collect(Gesso.Inference._quant_bytes(s))
    if s.ggml_type == Gesso.Inference.GGML_TYPE_Q8_0
        qs, d = _lq8_repack(bytes, s.nrows, s.ne0)
        return LavaQ8Quant(
            _lava_upload(UInt32, qs),
            _lava_upload(Float32, d),
            s.ne0,
            s.nrows,
        )
    end
    return LavaBlockQuant(
        _lava_upload(UInt8, bytes),
        s.ggml_type,
        _lq_kind(s.ggml_type),
        s.ne0,
        s.nrows,
        s.row_bytes,
    )
end

# --- GEMV ----------------------------------------------------------------------
# One workgroup of _LQ_LANES lanes per output row. Lane l accumulates elements
# l, l+64, ... of the row for each token, the partial sums meet in local
# memory, and lane 1 writes Y[t, row]. X and Y are indexed linearly (column
# major, token fastest): reshaped LavaArrays cannot be indexed inside a kernel.
# The row r starts at byte base + (r-1)*row_bytes, so an expert bank passes its
# expert's byte offset as base.

@kernel cpu=false function _lq_gemv_kernel!(
    Y,
    W,
    X,
    kv,
    g2,
    g3,
    ks,
    kind::Int,
    base::Int,
    ntok::Int,
    ne0::Int,
    row_bytes::Int,
    alpha::Float32,
    beta::Float32,
)
    tl = @index(Local, Linear)
    r = @index(Group, Linear)
    # One slot per subgroup; the subgroup size is only known at run time.
    part = @localmem Float32 (_LQ_LANES,)
    sgs = Lava.subgroup_size()
    sgid = (tl - 1) ÷ sgs
    nsg = _LQ_LANES ÷ sgs
    lane0 = Lava.subgroup_lane() == 0
    rowoff = base + (r - 1) * row_bytes
    for t in 1:ntok
        acc = 0.0f0
        e = tl - 1
        while e < ne0
            acc += _lq_elem(kind, W, kv, g2, g3, ks, rowoff, e) * X[e*ntok+t]
            e += _LQ_LANES
        end
        # subgroup_add must run with every lane active, so it sits outside the branch.
        red = Lava.subgroup_add(acc)
        if lane0
            part[sgid+1] = red
        end
        @synchronize
        if tl == 1
            s = 0.0f0
            for i in 1:nsg
                s += part[i]
            end
            idx = (r - 1) * ntok + t
            # beta == 0 must not read Y: the caller may hand over uninitialized memory.
            Y[idx] = beta == 0.0f0 ? alpha * s : alpha * s + beta * Y[idx]
        end
        @synchronize
    end
end

# --- Q8_0 repacked -------------------------------------------------------------
# A Q8_0 block is a 2-byte f16 scale and 32 int8 values, 34 bytes, so the values
# never sit on a 4-byte boundary and the byte-wise kernel reads them one byte at
# a time (~100 GB/s on the decode shapes). At upload the blocks split into
# 4-byte value words and an F32 scale array, and a lane loads whole words:
# 250-330 GB/s on the same shapes. The split costs 1/16 more device bytes.
struct LavaQ8Quant <: LavaDenseQuant
    qs::Lava.LavaArray{UInt32,1} # row r: words (r-1)*ne0/4+1 ... r*ne0/4
    d::Lava.LavaArray{Float32,1} # row r: scales (r-1)*ne0/32+1 ... r*ne0/32
    ne0::Int
    nrows::Int
end

Base.sizeof(q::LavaQ8Quant) = sizeof(q.qs) + sizeof(q.d)

function _lq8_repack(bytes::AbstractVector{UInt8}, nrows::Int, ne0::Int)
    nblk = nrows * (ne0 >> 5)
    length(bytes) == 34 * nblk ||
        error("Q8_0 repack: $(length(bytes)) bytes for $nblk blocks")
    qs = Vector{UInt32}(undef, 8 * nblk)
    d = Vector{Float32}(undef, nblk)
    @inbounds for b in 0:(nblk-1)
        o = 34 * b
        d[b+1] =
            Float32(reinterpret(Float16, UInt16(bytes[o+1]) | (UInt16(bytes[o+2]) << 8)))
        for w in 0:7
            p = o + 2 + 4 * w
            qs[8*b+w+1] =
                UInt32(bytes[p+1]) | (UInt32(bytes[p+2]) << 8) |
                (UInt32(bytes[p+3]) << 16) | (UInt32(bytes[p+4]) << 24)
        end
    end
    return qs, d
end

# Byte k (0..3) of w as a signed value: shift it to the top, then arithmetic-shift down.
@inline _lq8_i8(w::UInt32, k::Int) = Float32(reinterpret(Int32, w << (24 - 8 * k)) >> 24)

# Word w of the row against token t: elements 4w..4w+3, X linear and token fastest.
@inline function _lq8_dot4(q::UInt32, X, w::Int, ntok::Int, t::Int)
    e = 4 * w * ntok + t
    return _lq8_i8(q, 0) * X[e] +
           _lq8_i8(q, 1) * X[e+ntok] +
           _lq8_i8(q, 2) * X[e+2*ntok] +
           _lq8_i8(q, 3) * X[e+3*ntok]
end

# One subgroup per row and the subgroup sum is the row result: no local memory
# and no barrier. Each lane keeps two words in flight a subgroup apart. The
# subgroup width is only known inside the kernel (32 on NVIDIA, 64 by default
# on AMD), so a workgroup always owns _LQ8_ROWS rows and each subgroup walks
# the ones its width leaves it.
const _LQ8_LANES = 128
const _LQ8_ROWS = 4

@kernel cpu=false function _lq8_gemv_kernel!(Y, QS, D, X, ntok::Int, ne0::Int, nrows::Int)
    tl = @index(Local, Linear)
    g = @index(Group, Linear)
    sgs = Int(Lava.subgroup_size())
    sgid = (tl - 1) ÷ sgs
    nsg = _LQ8_LANES ÷ sgs
    lane = Int(Lava.subgroup_lane())
    nw = ne0 >> 2
    rr = sgid
    while rr < _LQ8_ROWS
        r = (g - 1) * _LQ8_ROWS + rr + 1
        if r <= nrows
            qoff = (r - 1) * nw
            doff = (r - 1) * (ne0 >> 5)
            for t in 1:ntok
                a0 = 0.0f0
                a1 = 0.0f0
                w = lane
                while w + sgs < nw
                    w1 = w + sgs
                    a0 += D[doff+(w>>3)+1] * _lq8_dot4(QS[qoff+w+1], X, w, ntok, t)
                    a1 += D[doff+(w1>>3)+1] * _lq8_dot4(QS[qoff+w1+1], X, w1, ntok, t)
                    w += 2 * sgs
                end
                if w < nw
                    a0 += D[doff+(w>>3)+1] * _lq8_dot4(QS[qoff+w+1], X, w, ntok, t)
                end
                # subgroup_add must run with every lane active, so it sits outside the branches.
                red = Lava.subgroup_add(a0 + a1)
                if lane == 0
                    Y[(r-1)*ntok+t] = red
                end
            end
        end
        rr += nsg
    end
end

@kernel cpu=false function _lq8_dequant_f16_kernel!(
    Wt,
    QS,
    D,
    ne0::Int,
    r0::Int,
    total::Int,
)
    idx = @index(Global, Linear)
    if idx <= total
        q = idx - 1
        r = q ÷ ne0
        e = q - r * ne0
        row = r0 + r
        word = QS[row*(ne0>>2)+(e>>2)+1]
        Wt[idx] = Float16(D[row*(ne0>>5)+(e>>5)+1] * _lq8_i8(word, e & 3))
    end
end

# Device-resident lookup tables, uploaded once. g2 holds the 64-bit grid
# entries of IQ2_XS, IQ2_XXS and IQ2_S back to back, each as two UInt32 words
# (low, then high), at the entry offsets _LQ_G2_*. g3 holds the 32-bit IQ3_XXS
# and IQ3_S grids at _LQ_G3_*. ks is the sign table.
const _LQ_TABLES = Ref{Any}(nothing)
function _lq_tables()
    tb = _LQ_TABLES[]
    if tb === nothing
        I = Gesso.Inference
        @assert length(I.IQ2XS_GRID) == _LQ_G2_XXS - _LQ_G2_XS
        @assert length(I.IQ2XXS_GRID) == _LQ_G2_S - _LQ_G2_XXS
        @assert length(I.IQ2S_GRID) == 1024
        @assert length(I.IQ3XXS_GRID) == _LQ_G3_S - _LQ_G3_XXS
        @assert length(I.IQ3S_GRID) == 512
        grids64 = (I.IQ2XS_GRID, I.IQ2XXS_GRID, I.IQ2S_GRID)
        g2 = Vector{UInt32}(undef, 2 * sum(length, grids64))
        p = 0
        for grid in grids64, v in grid
            g2[p+1] = UInt32(v & 0xffffffff)
            g2[p+2] = UInt32(v >> 32)
            p += 2
        end
        g3 = vcat(collect(UInt32, I.IQ3XXS_GRID), collect(UInt32, I.IQ3S_GRID))
        tb = (
            kv=Lava.LavaArray{Float32}(_LQ_KVALUES_IQ4NL),
            g2=Lava.LavaArray{UInt32}(g2),
            g3=Lava.LavaArray{UInt32}(g3),
            ks=Lava.LavaArray{UInt8}(collect(I.KSIGNS_IQ2XS)),
        )
        _LQ_TABLES[] = tb
    end
    return tb
end

function _lq_launch!(
    Y,
    X,
    W,
    base::Int,
    kind::Int,
    ntok::Int,
    ne0::Int,
    nrows::Int,
    row_bytes::Int,
    alpha::Float32,
    beta::Float32,
)
    tb = _lq_tables()
    kern = _lq_gemv_kernel!(Lava.KernelAbstractions.get_backend(X))
    kern(
        Y,
        W,
        X,
        tb.kv,
        tb.g2,
        tb.g3,
        tb.ks,
        kind,
        base,
        ntok,
        ne0,
        row_bytes,
        alpha,
        beta;
        ndrange=_LQ_LANES * nrows,
        workgroupsize=(_LQ_LANES,),
    )
    return Y
end

function _lq_matmul!(Y, X, q::LavaDenseQuant)
    ntok = size(X, 1)
    size(X, 2) == q.ne0 || throw(
        DimensionMismatch(
            "lava quant GEMV: x has $(size(X, 2)) columns, weight has $(q.ne0)",
        ),
    )
    size(Y) == (ntok, q.nrows) || throw(
        DimensionMismatch("lava quant GEMV: dst is $(size(Y)), want $((ntok, q.nrows))"),
    )
    cpu_order = ntok > 0 && _lq_cpu_order()
    if cpu_order && q isa LavaQ8Quant
        ntok <= _c1_wg_max(8) && return _c1_cpu_q80_matmul_wg!(Y, X, q)
        return _c1_cpu_q80_matmul!(Y, X, q)
    end
    if cpu_order && q isa LavaBlockQuant && q.kind == _LQ_KIND_Q4_K
        ntok <= _c1_wg_max(2) && return _c1_cpu_q4k_matmul_wg!(Y, X, q)
        return _c1_cpu_q4k_matmul!(Y, X, q)
    end
    ntok >= _lq_gemm_min() && return _lq_gemm_prefill!(Y, X, q)
    return _lq_gemv!(Y, X, q)
end

_lq_gemv!(Y, X, q::LavaBlockQuant) = _lq_launch!(
    Y,
    X,
    q.data,
    0,
    q.kind,
    size(X, 1),
    q.ne0,
    q.nrows,
    q.row_bytes,
    1.0f0,
    0.0f0,
)

function _lq_gemv!(Y, X, q::LavaQ8Quant)
    kern = _lq8_gemv_kernel!(Lava.KernelAbstractions.get_backend(X))
    kern(
        Y,
        q.qs,
        q.d,
        X,
        size(X, 1),
        q.ne0,
        q.nrows;
        ndrange=_LQ8_LANES * cld(q.nrows, _LQ8_ROWS),
        workgroupsize=(_LQ8_LANES,),
    )
    return Y
end

# --- prefill GEMM ----------------------------------------------------------------
# The GEMV decodes each weight once per token, so a 256-token prefill decodes
# every weight 256 times. From GESSO_TUNE=dq_gemm_min tokens on (the same default as the CUDA path),
# W is dequantized to F16 in row chunks, capped by GESSO_TUNE=dq_gemm_scratch_mib,
# and Lava's mul! multiplies it with an F16 copy of X into an F32 result. That
# is the cooperative-matrix path when the shape tiles, and the staged GEMM
# otherwise.
_lq_gemm_min() = env_tune(:dq_gemm_min, 32)
# Q8_0 and Q4_K go to the CPU-order kernels (ext/lava_quant_cpu_order.jl:
# thread per output row, CPU summation order) unless this is "0", which uses
# the subgroup GEMV and dequant GEMM every other quant kind already takes.
_lq_cpu_order() = get(ENV, "GESSO_LAVA_CPU_ORDER_QUANT", "1") != "0"
_lq_gemm_scratch_mib() = env_tune(:dq_gemm_scratch_mib, 32)

# Element e of row r0 + r goes to Wt[r * ne0 + e + 1], so Wt is the (ne0 x rc)
# column-major matrix whose columns are the chunk's rows.
@kernel cpu=false function _lq_dequant_f16_kernel!(
    Wt,
    W,
    kv,
    g2,
    g3,
    ks,
    kind::Int,
    ne0::Int,
    row_bytes::Int,
    r0::Int,
    total::Int,
)
    idx = @index(Global, Linear)
    if idx <= total
        q = idx - 1
        r = q ÷ ne0
        e = q - r * ne0
        rowoff = (r0 + r) * row_bytes
        Wt[idx] = Float16(_lq_elem(kind, W, kv, g2, g3, ks, rowoff, e))
    end
end

# Copies a chunk's F32 columns into Y at flat offset `off`.
@kernel cpu=false function _lq_scatter_kernel!(Y, C, off::Int, total::Int)
    idx = @index(Global, Linear)
    if idx <= total
        Y[off+idx] = C[idx]
    end
end

# Scratch outlives the call. A LavaArray allocated per call leaves a GC finalizer
# that runs inside a later call's timed region: the composed prefill measured
# 7.2 ms with the GC inside the timer and 1.2 ms with it outside. The cache is
# emptied once it holds _LQ_SCRATCH_MAX shapes, so a run over many prompt
# lengths cannot grow VRAM without bound.
const _LQ_SCRATCH = Dict{Tuple{DataType, Symbol, Tuple{Vararg{Int}}}, Any}()
const _LQ_SCRATCH_MAX = 24

function _lq_scratch(::Type{T}, tag::Symbol, dims::Tuple{Vararg{Int}}) where {T}
    key = (T, tag, dims)
    a = get(_LQ_SCRATCH, key, nothing)
    if a === nothing
        length(_LQ_SCRATCH) >= _LQ_SCRATCH_MAX && empty!(_LQ_SCRATCH)
        a = Lava.LavaArray{T}(undef, dims...)
        _LQ_SCRATCH[key] = a
    end
    return a
end

_lq_dequant_chunk!(wtf, q::LavaBlockQuant, tb, r0::Int, total::Int) =
    _lq_dequant_f16_kernel!(Lava.KernelAbstractions.get_backend(wtf))(
        wtf,
        q.data,
        tb.kv,
        tb.g2,
        tb.g3,
        tb.ks,
        q.kind,
        q.ne0,
        q.row_bytes,
        r0,
        total;
        ndrange=total,
        workgroupsize=(256,),
    )

_lq_dequant_chunk!(wtf, q::LavaQ8Quant, tb, r0::Int, total::Int) =
    _lq8_dequant_f16_kernel!(Lava.KernelAbstractions.get_backend(wtf))(
        wtf,
        q.qs,
        q.d,
        q.ne0,
        r0,
        total;
        ndrange=total,
        workgroupsize=(256,),
    )

function _lq_gemm_prefill!(Y, X, q::LavaDenseQuant)
    ntok = size(X, 1)
    k = q.ne0
    nrows = q.nrows
    backend = Lava.KernelAbstractions.get_backend(X)
    tb = _lq_tables()
    x16 = _lq_scratch(Float16, :x16, (ntok, k))
    x16 .= Float16.(X)
    rc_max = min(nrows, max(1, (_lq_gemm_scratch_mib() << 20) ÷ (2 * k)))
    scatter = _lq_scatter_kernel!(backend)
    r0 = 0
    while r0 < nrows
        rc = min(rc_max, nrows - r0)
        total = k * rc
        # The dequant writes flat indices, so it takes the flat buffer. mul! takes
        # the reshape, which Lava's gemm reads as a dense strided operand.
        wtf = _lq_scratch(Float16, :wt, (total,))
        _lq_dequant_chunk!(wtf, q, tb, r0, total)
        wt = reshape(wtf, k, rc)
        if rc == nrows
            mul!(Y, x16, wt, one(Float32), zero(Float32))
        else
            yc = _lq_scratch(Float32, :yc, (ntok, rc))
            mul!(yc, x16, wt, one(Float32), zero(Float32))
            scatter(Y, yc, r0 * ntok, ntok * rc; ndrange=ntok * rc, workgroupsize=(256,))
        end
        r0 += rc
    end
    return Y
end

# Single-token expert GEMV: y = alpha * W_e * x + beta * y, with W_e the
# expert's (ne1 x ne0) block of the device bank. Same contract as the CUDA
# method in cuda_quant_gemm.jl.
function Gesso.Inference._qgemv_expert!(
    y::Lava.LavaArray{Float32, 1},
    x::Lava.LavaArray{Float32, 1},
    bank::Gesso.Inference.GpuQuantSlice{<:Lava.LavaArray},
    e::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
)
    I = Gesso.Inference
    kind = _lq_expert_kind(bank.ggml_type)
    kind === nothing &&
        error("lava qgemv: no expert kernel for ggml type $(bank.ggml_type)")
    length(x) == bank.ne0 || throw(
        DimensionMismatch(
            "lava expert GEMV: x has $(length(x)) elements, bank has $(bank.ne0) columns",
        ),
    )
    length(y) == bank.ne1 || throw(
        DimensionMismatch(
            "lava expert GEMV: y has $(length(y)) elements, bank has $(bank.ne1) rows",
        ),
    )
    bn = _lq_block_values(kind)
    bank.ne0 % bn == 0 ||
        error("lava expert GEMV: ne0 = $(bank.ne0) is not a multiple of the $bn-value block")
    row_bytes = bank.bytes_per_expert ÷ bank.ne1
    base = I._quant_expert_off0(bank, e)
    _lq_launch!(y, x, bank.data, base, kind, 1, bank.ne0, bank.ne1, row_bytes, alpha, beta)
    return y
end

include("lava_quant_cpu_order.jl")
