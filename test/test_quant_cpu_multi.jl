# Multi-column Q8_K dot products (src/Inference/quant_cpu_multi.jl):
#   * every tile size 1..8, at several activation offsets, on random rows: each
#     column bit-equal (===) to the single-column _dot_q8k, for simd on and off
#   * the same on real rows when GESSO_SALUKI_GGUF is set (skipped otherwise)
#   * the prefill path through _quant_matmul! equals the per-token path bit for
#     bit, with the GESSO_PREFILL_MULTI switch on and off
#   * Q6_K at the integer extremes (u = 0 and 63, scales and activations at
#     -128 and 127), bit-equal to the single-column kernel
#   * Q4_K and Q5_K at the integer extremes (weights 0 and 15/31, scales and
#     mins at 0 and 63, activations at -128 and 127), plain and prepped
#   * Q3_K at the integer extremes (u = 0 and 7, scales at -32 and 31,
#     activations at -128 and 127), plain and prepped
#   * Q8_0, Q4_0 and IQ4_NL at the integer extremes (weight bytes 0x00 and 0xff,
#     activations at -128 and 127), with and without a d table
#   * IQ2_XS at the integer extremes (grid 511 and 0, every sign set or none,
#     ls = 31 and 1, activations at -128 and 127), with and without a d table
#   * prepped Q3_K, Q4_K, Q5_K and Q6_K activations (_q6k_prep) give the same bits,
#     extremes included
#   * the tile kernels do not allocate after warm-up
#   * the single-column path through _quant_matmul! reuses its scratch buffers
#     across shapes (large, small, large) and stays bit-equal to fresh buffers

using Random: Xoshiro
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _MQ = Gesso.Inference

const _MQ_TYPES = (
    ("IQ2_XXS", _MQ.GGML_TYPE_IQ2_XXS),
    ("IQ3_S", _MQ.GGML_TYPE_IQ3_S),
    ("IQ2_S", _MQ.GGML_TYPE_IQ2_S),
    ("IQ1_S", _MQ.GGML_TYPE_IQ1_S),
    ("IQ1_M", _MQ.GGML_TYPE_IQ1_M),
    ("IQ2_XS", _MQ.GGML_TYPE_IQ2_XS),
    ("IQ3_XXS", _MQ.GGML_TYPE_IQ3_XXS),
    ("IQ4_XS", _MQ.GGML_TYPE_IQ4_XS),
    ("Q2_K", _MQ.GGML_TYPE_Q2_K),
    ("Q3_K", _MQ.GGML_TYPE_Q3_K),
    ("Q4_K", _MQ.GGML_TYPE_Q4_K),
    ("Q5_K", _MQ.GGML_TYPE_Q5_K),
    ("Q6_K", _MQ.GGML_TYPE_Q6_K),
    ("Q8_0", _MQ.GGML_TYPE_Q8_0),
    ("Q4_0", _MQ.GGML_TYPE_Q4_0),
    ("IQ4_NL", _MQ.GGML_TYPE_IQ4_NL),
)

# Random valid rows: arbitrary bytes everywhere, with finite small f16 scales at
# each format's super-scale offsets (random f16 bits are often NaN or Inf).
function _mq_scale_offsets(ggml_type::Int32)
    ggml_type == _MQ.GGML_TYPE_Q2_K && return (80, 82)
    ggml_type in (_MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K) && return (0, 2)
    ggml_type == _MQ.GGML_TYPE_Q3_K && return (108,)
    ggml_type == _MQ.GGML_TYPE_Q6_K && return (208,)
    return (0,)
end

function _mq_random_rows(rng, ggml_type::Int32, nrows::Int, nb::Int)
    # nb counts 256-value super-blocks; Q8_0 and Q4_0 blocks hold 32 values.
    _MQ._q80_act_type(ggml_type) && (nb *= 8)
    bsz = _MQ._iq_block_bytes(ggml_type)
    w = rand(rng, UInt8, nrows * nb * bsz)
    for blk in 0:(nrows*nb-1)
        if ggml_type == _MQ.GGML_TYPE_IQ1_M
            _MQ._iq1m_write_scale!(w, blk * bsz, 0.005f0 + 0.05f0 * rand(rng, Float32))
            continue
        end
        for off in _mq_scale_offsets(ggml_type)
            bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
            w[blk*bsz+off+1] = UInt8(bits & 0xff)
            w[blk*bsz+off+2] = UInt8(bits >> 8)
        end
    end
    return w
end

_mq_acts(rng, ne0::Int, n::Int, ggml_type::Int32=_MQ.GGML_TYPE_Q4_K) =
    [_MQ._q8k_quantize_act(ggml_type, randn(rng, Float32, ne0)) for _ in 1:n]

# Checks every column of a tile of `ntile` activations starting at t0 against
# the single-column kernel, for both simd modes, on the rows of w. Q4_K, Q5_K,
# Q6_K, Q8_0, Q4_0, IQ4_NL and IQ2_XS tiles that start a d-table tile are also
# checked through the table. IQ2_XS tiles also run through the packed table.
_mq_uses_table(ggml_type::Int32, t0::Int, ntile::Int) =
    _MQ._multi_dt_type(ggml_type) &&
    (t0 - 1) % _MQ._MULTI_TILE == 0 &&
    ntile <= _MQ._MULTI_TILE

function _mq_check_tiles(ggml_type::Int32, w, nrows::Int, rb::Int, xqs, t0::Int, ntile::Int)
    nmis = 0
    out = zeros(Float32, length(xqs))
    dt = _mq_uses_table(ggml_type, t0, ntile) ? _MQ._multi_d_table(xqs) : nothing
    modes =
        dt === nothing ? ((true, nothing), (false, nothing)) :
        ((true, nothing), (false, nothing), (true, dt))
    if ggml_type == _MQ.GGML_TYPE_IQ2_XS && dt !== nothing
        modes = (modes..., (true, _MQ._multi_packed(xqs)))
    end
    for (simd, d) in modes, j in 1:nrows
        woff = (j - 1) * rb
        fill!(out, NaN32)
        _MQ._dot_q8k_multi!(out, ggml_type, w, woff, xqs, t0, ntile, simd, d)
        for c in 1:ntile
            ref = _MQ._dot_q8k(ggml_type, w, woff, xqs[t0+c-1], simd)
            nmis += !(out[t0+c-1] === ref)
        end
    end
    return nmis
end

@testset "multi-column dot: every type, every tile size, bit-equal to _dot_q8k" begin
    rng = Xoshiro(0x51ce)
    nb = 3
    ne0 = 256 * nb
    nrows = 4
    for (name, ty) in _MQ_TYPES
        rb = _MQ._qrow_bytes(ty, ne0)
        w = _mq_random_rows(rng, ty, nrows, nb)
        xqs = _mq_acts(rng, ne0, 9, ty)
        @testset "$name" begin
            for ntile in 1:8, t0 in (1, 2)
                nmis = _mq_check_tiles(ty, w, nrows, rb, xqs, t0, ntile)
                @test nmis == 0
            end
        end
    end
end

@testset "multi-column dot: Q6_K at the integer extremes" begin
    # Weights at u = 0 and 63, scales at -128 and 127, activations at -128 and
    # 127 (the quantizer clamps to ±127, so the vectors are built by hand). This
    # drives the 16-bit pair sums and the 32-bit lane totals to their limits.
    rng = Xoshiro(0x6e6e)
    nb = 2
    ty = _MQ.GGML_TYPE_Q6_K
    bsz = _MQ._iq_block_bytes(ty)
    rb = _MQ._qrow_bytes(ty, 256 * nb)
    rows = (
        fill(0xff, rb),                  # u = 63 everywhere
        zeros(UInt8, rb),                # u = 0 everywhere
        rand(rng, (0x00, 0xff), rb),     # a mix of 0 and 63 per byte
    )
    w = UInt8[]
    for (r, row) in enumerate(rows)
        for blk in 0:(nb-1)
            o = blk * bsz
            for k in 1:16
                row[o+192+k] = isodd(k + r) ? 0x80 : 0x7f
            end
            row[o+209] = 0x00
            row[o+210] = 0x3c            # d = 1.0 in f16
        end
        append!(w, row)
    end
    nrows = length(rows)
    pick = (Int8(-128), Int8(127), Int8(-127), Int8(0), Int8(1))
    xqs = map(1:9) do c
        x = _MQ.Q8KVec(nb)
        for i in eachindex(x.qs)
            x.qs[i] = c <= 2 ? (c == 1 ? Int8(-128) : Int8(127)) : rand(rng, pick)
        end
        for i in eachindex(x.bsums)
            x.bsums[i] = sum(Int16, @view x.qs[(16(i-1)+1):(16i)])
        end
        x.d .= 1.0f0
        x
    end
    xps = map(_MQ._q6k_prep, xqs)
    for ntile in 1:8, t0 in (1, 2)
        @test _mq_check_tiles(ty, w, nrows, rb, xqs, t0, ntile) == 0
        @test _mq_check_tiles(ty, w, nrows, rb, xps, t0, ntile) == 0
    end
end

@testset "multi-column dot: Q4_K and Q5_K at the integer extremes" begin
    # All-ones bytes give weight 15 (Q4_K) or 31 (Q5_K) with every scale and min
    # at 63; zero bytes give weight, scale and min 0; the mix sets each byte to
    # one or the other. d = dmin = 1.0, activations at -128 and 127 built by
    # hand, so the 16-bit pair sums reach 2*31*128.
    rng = Xoshiro(0x45c5)
    nb = 2
    for ty in (_MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K)
        bsz = _MQ._iq_block_bytes(ty)
        rb = _MQ._qrow_bytes(ty, 256 * nb)
        rows = (fill(0xff, rb), zeros(UInt8, rb), rand(rng, (0x00, 0xff), rb))
        w = UInt8[]
        for row in rows
            for blk in 0:(nb-1)
                o = blk * bsz
                row[o+1] = 0x00
                row[o+2] = 0x3c
                row[o+3] = 0x00
                row[o+4] = 0x3c
            end
            append!(w, row)
        end
        pick = (Int8(-128), Int8(127), Int8(-127), Int8(0), Int8(1))
        xqs = map(1:9) do c
            x = _MQ.Q8KVec(nb)
            for i in eachindex(x.qs)
                x.qs[i] = c <= 2 ? (c == 1 ? Int8(-128) : Int8(127)) : rand(rng, pick)
            end
            for i in eachindex(x.bsums)
                x.bsums[i] = sum(Int16, @view x.qs[(16(i-1)+1):(16i)])
            end
            x.d .= 1.0f0
            x
        end
        xps = map(_MQ._q6k_prep, xqs)
        for ntile in 1:8, t0 in (1, 2)
            @test _mq_check_tiles(ty, w, length(rows), rb, xqs, t0, ntile) == 0
            @test _mq_check_tiles(ty, w, length(rows), rb, xps, t0, ntile) == 0
        end
    end
end

@testset "multi-column dot: Q3_K at the integer extremes" begin
    # All-ones bytes give u = 7 (weight 3) with every scale at 31; zero bytes
    # give u = 0 (weight -4) with every scale at -32; the mix sets each byte to
    # one or the other. d = 1.0, activations at -128 and 127 built by hand.
    rng = Xoshiro(0x3c3c)
    nb = 2
    ty = _MQ.GGML_TYPE_Q3_K
    bsz = _MQ._iq_block_bytes(ty)
    rb = _MQ._qrow_bytes(ty, 256 * nb)
    rows = (fill(0xff, rb), zeros(UInt8, rb), rand(rng, (0x00, 0xff), rb))
    w = UInt8[]
    for row in rows
        for blk in 0:(nb-1)
            o = blk * bsz
            row[o+109] = 0x00
            row[o+110] = 0x3c            # d = 1.0 in f16
        end
        append!(w, row)
    end
    pick = (Int8(-128), Int8(127), Int8(-127), Int8(0), Int8(1))
    xqs = map(1:9) do c
        x = _MQ.Q8KVec(nb)
        for i in eachindex(x.qs)
            x.qs[i] = c <= 2 ? (c == 1 ? Int8(-128) : Int8(127)) : rand(rng, pick)
        end
        for i in eachindex(x.bsums)
            x.bsums[i] = sum(Int16, @view x.qs[(16(i-1)+1):(16i)])
        end
        x.d .= 1.0f0
        x
    end
    xps = map(_MQ._q6k_prep, xqs)
    for ntile in 1:8, t0 in (1, 2)
        @test _mq_check_tiles(ty, w, length(rows), rb, xqs, t0, ntile) == 0
        @test _mq_check_tiles(ty, w, length(rows), rb, xps, t0, ntile) == 0
    end
end

@testset "multi-column dot: Q8_0, Q4_0 and IQ4_NL at the integer extremes" begin
    # Bytes 0xff and 0x00 give the 4-bit formats their end weights (Q4_0 7 and
    # -8, IQ4_NL 113 and -127), 0x80 and 0x7f give Q8_0 -128 and 127; the mix
    # sets each byte to one of these. d = 1.0, activations at -128 and 127 built
    # by hand.
    rng = Xoshiro(0x40e1)
    nb = 16
    for ty in (_MQ.GGML_TYPE_Q8_0, _MQ.GGML_TYPE_Q4_0, _MQ.GGML_TYPE_IQ4_NL)
        bsz = _MQ._iq_block_bytes(ty)
        rb = _MQ._qrow_bytes(ty, 32 * nb)
        rows = (fill(0xff, rb), zeros(UInt8, rb), rand(rng, (0x00, 0xff, 0x80, 0x7f), rb), rand(rng, UInt8, rb))
        w = UInt8[]
        for row in rows
            for blk in 0:(nb-1)
                row[blk*bsz+1] = 0x00
                row[blk*bsz+2] = 0x3c    # d = 1.0 in f16
            end
            append!(w, row)
        end
        pick = (Int8(-128), Int8(127), Int8(-127), Int8(0), Int8(1))
        xqs = map(1:9) do c
            x = _MQ.Q80Vec(nb)
            for i in eachindex(x.qs)
                x.qs[i] = c <= 2 ? (c == 1 ? Int8(-128) : Int8(127)) : rand(rng, pick)
            end
            x.d .= 1.0f0
            x
        end
        for ntile in 1:8, t0 in (1, 2)
            @test _mq_check_tiles(ty, w, length(rows), rb, xqs, t0, ntile) == 0
        end
    end
end

@testset "multi-column dot: IQ2_XS at the integer extremes" begin
    # 0xff bytes give grid 511, every sign set and scale nibbles 15 (ls = 31);
    # 0x00 bytes give grid 0, no signs and ls = 1. Activations at -128 (which
    # psign cannot negate) and 127, d = 1.0, with and without a d table.
    rng = Xoshiro(0x1a2b)
    nb = 2
    ty = _MQ.GGML_TYPE_IQ2_XS
    bsz = _MQ._iq_block_bytes(ty)
    rb = _MQ._qrow_bytes(ty, 256 * nb)
    rows = (fill(0xff, rb), zeros(UInt8, rb), rand(rng, (0x00, 0xff), rb), rand(rng, UInt8, rb))
    w = UInt8[]
    for row in rows
        for blk in 0:(nb-1)
            row[blk*bsz+1] = 0x00
            row[blk*bsz+2] = 0x3c    # d = 1.0 in f16
        end
        append!(w, row)
    end
    pick = (Int8(-128), Int8(127), Int8(-127), Int8(0), Int8(1))
    xqs = map(1:9) do c
        x = _MQ.Q8KVec(nb)
        for i in eachindex(x.qs)
            x.qs[i] = c <= 2 ? (c == 1 ? Int8(-128) : Int8(127)) : rand(rng, pick)
        end
        for i in eachindex(x.bsums)
            x.bsums[i] = sum(Int16, @view x.qs[(16(i-1)+1):(16i)])
        end
        x.d .= 1.0f0
        x
    end
    for ntile in 1:8, t0 in (1, 2)
        @test _mq_check_tiles(ty, w, length(rows), rb, xqs, t0, ntile) == 0
    end
end

@testset "multi-column dot: prepped activations, every tile size, bit-equal to _dot_q8k" begin
    rng = Xoshiro(0x9b6c)
    nb = 3
    for ty in (_MQ.GGML_TYPE_Q3_K, _MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K, _MQ.GGML_TYPE_Q6_K)
        rb = _MQ._qrow_bytes(ty, 256 * nb)
        w = _mq_random_rows(rng, ty, 4, nb)
        xps = map(_MQ._q6k_prep, _mq_acts(rng, 256 * nb, 9, ty))
        for ntile in 1:8, t0 in (1, 2)
            @test _mq_check_tiles(ty, w, 4, rb, xps, t0, ntile) == 0
        end
        out = zeros(Float32, 4)
        _MQ._dot_q8k_multi!(out, ty, w, 0, xps, 1, 4, true)
        @test (@allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xps, 1, 4, true)) == 0
        dt = _MQ._multi_d_table(xps)
        _MQ._dot_q8k_multi!(out, ty, w, 0, xps, 1, 4, true, dt)
        @test (@allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xps, 1, 4, true, dt)) == 0
    end
end

@testset "multi-column dot: the d table" begin
    rng = Xoshiro(0xd7ab)
    nb = 3
    for ty in (
        _MQ.GGML_TYPE_Q3_K,
        _MQ.GGML_TYPE_Q4_K,
        _MQ.GGML_TYPE_Q5_K,
        _MQ.GGML_TYPE_Q6_K,
        _MQ.GGML_TYPE_Q8_0,
        _MQ.GGML_TYPE_Q4_0,
        _MQ.GGML_TYPE_IQ4_NL,
        _MQ.GGML_TYPE_IQ2_XS,
    )
        rb = _MQ._qrow_bytes(ty, 256 * nb)
        w = _mq_random_rows(rng, ty, 2, nb)
        xqs = _mq_acts(rng, 256 * nb, 9, ty)
        dt = _MQ._multi_d_table(xqs)
        T = _MQ._MULTI_TILE
        @test length(dt) == T * length(xqs[1].d) * cld(9, T)
        for t in 1:9, ib in 0:(length(xqs[1].d)-1)
            i, c = divrem(t - 1, T)
            @test dt[T*(length(xqs[1].d)*i+ib)+c+1] === xqs[t].d[ib+1]
        end
        # every aligned tile of 9 columns, the last one short
        for t0 in 1:T:9
            K = min(T, 10 - t0)
            @test _mq_check_tiles(ty, w, 2, rb, xqs, t0, K) == 0
        end
        out = zeros(Float32, 9)
        @test_throws ErrorException _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 2, 4, true, dt)
        @test_throws ErrorException _MQ._dot_q8k_multi!(
            out,
            ty,
            w,
            0,
            xqs,
            1,
            T + 1,
            true,
            dt,
        )
        @test_throws ErrorException _MQ._dot_q8k_multi!(
            out,
            ty,
            w,
            0,
            xqs,
            1,
            4,
            true,
            dt[1:4],
        )
    end
end

@testset "multi-column dot: the packed IQ2_XS table" begin
    rng = Xoshiro(0x9ac4)
    nb = 3
    ty = _MQ.GGML_TYPE_IQ2_XS
    T = _MQ._MULTI_TILE
    rb = _MQ._qrow_bytes(ty, 256 * nb)
    w = _mq_random_rows(rng, ty, 2, nb)
    xqs = _mq_acts(rng, 256 * nb, 11, ty)
    p = _MQ._multi_packed(xqs)
    @test p.d == _MQ._multi_d_table(xqs)
    nch = 8 * nb
    @test length(p.q) == 32 * T * nch * cld(11, T)
    ok = true
    for t in 1:16, j in 0:(nch-1), k in 1:32
        i, c = divrem(t - 1, T)
        v = p.q[32*(T*(nch*i+j)+c)+k]
        ok &= t <= 11 ? v === xqs[t].qs[32j+k] : v === Int8(0)
    end
    @test ok
    # a reused buffer is resized to fit and its padding columns are zeroed
    q2 = _MQ._multi_packed!(Float32[], fill(Int8(7), 10^5), xqs[1:3])
    @test length(q2.q) == 32 * T * nch
    @test all(j -> all(iszero, @view q2.q[(32*(T*j+3)+1):(32*T*(j+1))]), 0:(nch-1))
    out = zeros(Float32, 11)
    for t0 in 1:T:11
        K = min(T, 12 - t0)
        _MQ._dot_q8k_multi!(out, ty, w, rb, xqs, t0, K, true, p)
        for c in 1:K
            @test out[t0+c-1] === _MQ._dot_q8k(ty, w, rb, xqs[t0+c-1], true)
        end
    end
    @test_throws ErrorException _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 2, 4, true, p)
    short = _MQ._MultiPacked(p.d, p.q[1:(32*T*nch)])
    @test_throws ErrorException _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 9, 3, true, short)
    _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true, p)
    @test (@allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true, p)) == 0
end
@testset "multi-column dot: the tile kernels do not allocate" begin
    rng = Xoshiro(7)
    nb = 4
    ne0 = 256 * nb
    for ty in (
        _MQ.GGML_TYPE_IQ2_XXS,
        _MQ.GGML_TYPE_IQ3_S,
        _MQ.GGML_TYPE_Q2_K,
        _MQ.GGML_TYPE_Q3_K,
        _MQ.GGML_TYPE_Q4_K,
        _MQ.GGML_TYPE_Q5_K,
        _MQ.GGML_TYPE_Q6_K,
        _MQ.GGML_TYPE_Q8_0,
        _MQ.GGML_TYPE_Q4_0,
        _MQ.GGML_TYPE_IQ4_NL,
    )
        rb = _MQ._qrow_bytes(ty, ne0)
        w = _mq_random_rows(rng, ty, 1, nb)
        xqs = _mq_acts(rng, ne0, 4, ty)
        out = zeros(Float32, 4)
        _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true)
        a = @allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true)
        @test a == 0
        if _mq_uses_table(ty, 1, 4)
            dt = _MQ._multi_d_table(xqs)
            _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true, dt)
            @test (@allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true, dt)) == 0
        end
    end
end

@testset "prefill: _quant_matmul! equals the per-token path bit for bit" begin
    rng = Xoshiro(0xbeef)
    nb = 3
    ne0 = 256 * nb
    nrows = 37
    seq = 7
    for (name, ty) in _MQ_TYPES
        rb = _MQ._qrow_bytes(ty, ne0)
        data = _mq_random_rows(rng, ty, nrows, nb)
        q = _MQ.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(256))
        x = randn(rng, Float32, seq, ne0)
        saved = _MQ._PREFILL_MULTI[]
        dst_multi = zeros(Float32, seq, nrows)
        dst_per_token = zeros(Float32, seq, nrows)
        try
            _MQ._PREFILL_MULTI[] = Int8(1)
            _MQ._quant_matmul!(dst_multi, x, q)
            _MQ._PREFILL_MULTI[] = Int8(0)
            _MQ._quant_matmul!(dst_per_token, x, q)
        finally
            _MQ._PREFILL_MULTI[] = saved
        end
        @testset "$name" begin
            @test all(
                reinterpret.(UInt32, dst_multi) .== reinterpret.(UInt32, dst_per_token),
            )
        end
    end
end

@testset "GESSO_PREFILL_MULTI switch is read as a flag" begin
    @test _MQ._prefill_multi_enabled() isa Bool
end

@testset "real tensors: multi-column vs single-column (Saluki GGUF)" begin
    path = get(ENV, "GESSO_SALUKI_GGUF", "")
    if !isfile(path)
        _skip(
            "set GESSO_SALUKI_GGUF to the Saluki 27B GGUF — real multi-column rows skipped",
        )
    else
        hdr = _MQ.gguf_header(path)
        rng = Xoshiro(31)
        for (name, ty) in _MQ_TYPES
            idx = findfirst(t -> t.ggml_type == ty, hdr.tensors)
            if idx === nothing
                _skip("no $name tensor in $(basename(path))")
                continue
            end
            info = hdr.tensors[idx]
            ne0, ne1 = info.dims[1], info.dims[2]
            rb = _MQ._qrow_bytes(ty, ne0)
            base = Int(hdr.data_start) + Int(info.offset)
            xqs = _mq_acts(rng, ne0, 8, ty)
            rows = unique(round.(Int, range(1, ne1; length=min(8, ne1))))
            nmis = 0
            out = zeros(Float32, 8)
            for simd in (true, false), j in rows, ntile in (1, 4, 8)
                _MQ._dot_q8k_multi!(
                    out,
                    ty,
                    hdr.data,
                    base + (j - 1) * rb,
                    xqs,
                    1,
                    ntile,
                    simd,
                )
                for c in 1:ntile
                    ref = _MQ._dot_q8k(ty, hdr.data, base + (j - 1) * rb, xqs[c], simd)
                    nmis += !(out[c] === ref)
                end
            end
            @testset "$name" begin
                @test nmis == 0
            end
        end
    end
end

@testset "single-column scratch: reuse across shapes is bit-equal to fresh buffers" begin
    rng = Xoshiro(0x5c4a)
    shapes = ((64, 4), (8, 1), (64, 4), (128, 8))
    # F32 is the only type on the plain GEMV branch of _quant_matmul! (every IQ
    # and K type is Q8_K-supported); Q4_K covers the Q8_K branch. Both reuse
    # the scratch, so both need to be covered.
    for (name, ty) in (("F32", _MQ.GGML_TYPE_F32), ("Q4_K", _MQ.GGML_TYPE_Q4_K))
        @testset "$name" begin
            for (nrows, nb) in shapes
                ne0 = 256 * nb
                rb = _MQ._qrow_bytes(ty, ne0)
                w =
                    ty == _MQ.GGML_TYPE_F32 ?
                    collect(reinterpret(UInt8, randn(rng, Float32, nrows * ne0))) :
                    _mq_random_rows(rng, ty, nrows, nb)
                q = _MQ.QuantMatrix(w, 0, ty, ne0, nrows, rb, Int64(256))
                x = randn(rng, Float32, 1, ne0)
                dst = zeros(Float32, 1, nrows)
                _MQ._quant_matmul!(dst, x, q)
                ref = zeros(Float32, nrows)
                xf = Float32.(x[1, :])
                bytes = view(w, 1:(nrows*rb))
                if _MQ._quant_q8k_native(ty)
                    _MQ._qgemv_q8k_f32!(ref, xf, bytes, ty, ne0, nrows; threaded=true)
                else
                    _MQ._qgemv_bytes_threaded!(ref, xf, bytes, ty, ne0, nrows, 1.0f0, 0.0f0)
                end
                @test all(dst[1, r] === ref[r] for r in 1:nrows)
            end
        end
    end
end

@testset "single-column scratch: concurrent drivers do not share buffers" begin
    rng = Xoshiro(0x5c4b)
    ty = _MQ.GGML_TYPE_Q4_K
    nrows, nb = 64, 8
    ne0 = 256 * nb
    rb = _MQ._qrow_bytes(ty, ne0)
    w = _mq_random_rows(rng, ty, nrows, nb)
    q = _MQ.QuantMatrix(w, 0, ty, ne0, nrows, rb, Int64(256))
    ntask = 4
    xs = [randn(rng, Float32, 1, ne0) for _ in 1:ntask]
    refs = map(xs) do x
        d = zeros(Float32, 1, nrows)
        _MQ._quant_matmul!(d, x, q)
        d
    end
    # Each task repeats its own input; a shared activation row shows up as a
    # result that matches another task's input instead of its own.
    bad = Threads.Atomic{Int}(0)
    @sync for k in 1:ntask
        Threads.@spawn begin
            d = zeros(Float32, 1, nrows)
            for _ in 1:400
                _MQ._quant_matmul!(d, xs[k], q)
                d == refs[k] || Threads.atomic_add!(bad, 1)
            end
        end
    end
    @test bad[] == 0
end
