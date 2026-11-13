# Multi-column Q8_K dot products (src/Inference/quant_cpu_multi.jl):
#   * every tile size 1..8, at several activation offsets, on random rows: each
#     column bit-equal (===) to the single-column _dot_q8k, for simd on and off
#   * the same on real rows when GESSO_SALUKI_GGUF is set (skipped otherwise)
#   * the prefill path through _quant_matmul! equals the per-token path bit for
#     bit, with GESSO_DISABLE=prefill_multi set and unset
#   * Q6_K at the integer extremes (u = 0 and 63, scales and activations at
#     -128 and 127), bit-equal to the single-column kernel
#   * Q4_K and Q5_K at the integer extremes (weights 0 and 15/31, scales and
#     mins at 0 and 63, activations at -128 and 127), plain and prepped
#   * Q3_K at the integer extremes (u = 0 and 7, scales at -32 and 31,
#     activations at -128 and 127), plain and prepped
#   * Q2_0 rows through the packed tile kernel, bit-equal to _dot_q8k
#   * Q8_0, Q4_0 and IQ4_NL at the integer extremes (weight bytes 0x00 and 0xff,
#     activations at -128 and 127), with and without a d table
#   * IQ2_XS at the integer extremes (grid 511 and 0, every sign set or none,
#     ls = 31 and 1, activations at -128 and 127), with and without a d table
#   * IQ2_XS, IQ2_XXS and IQ2_S rows decoded once and run over several tiles,
#     bit-equal to _dot_q8k
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
    ("Q5_1", _MQ.GGML_TYPE_Q5_1),
    ("Q2_0", _MQ.GGML_TYPE_Q2_0),
)

# Random valid rows: arbitrary bytes everywhere, with finite small f16 scales at
# each format's super-scale offsets (random f16 bits are often NaN or Inf).
function _mq_scale_offsets(ggml_type::Int32)
    ggml_type == _MQ.GGML_TYPE_Q2_K && return (80, 82)
    ggml_type in (_MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K, _MQ.GGML_TYPE_Q5_1) && return (0, 2)
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
# checked through the table, all but the K-quants through the packed table, and
# Q4_K and Q5_K through the mins table.
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
    if _MQ._multi_pack_type(ggml_type) && dt !== nothing
        modes = (modes..., (true, _MQ._multi_packed(xqs)))
    end
    if ggml_type in (_MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K) && dt !== nothing
        modes = (modes..., (true, _MQ._multi_q45k_tables(xqs)))
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

@testset "multi-column dot: the Q4_K / Q5_K mins table" begin
    rng = Xoshiro(0x3175)
    nb = 3
    T = _MQ._MULTI_TILE
    xqs = _mq_acts(rng, 256 * nb, 11, _MQ.GGML_TYPE_Q4_K)
    # bsums at both ends, so the pair sums reach -4096 and 4064
    xqs[2].bsums[1:16] .= Int16(-2048)
    xqs[3].bsums[1:16] .= Int16(2032)
    p = _MQ._multi_q45k_tables(xqs)
    @test p.d == _MQ._multi_d_table(xqs)
    @test length(p.m) == 64 * nb * cld(11, T)
    ok = true
    for t in 1:16, ib in 0:(nb-1), s in 0:7
        i, c = divrem(t - 1, T)
        v = p.m[64*(nb*i+ib)+16*(s>>1)+2c+(s&1)+1]
        ok &=
            t <= 11 ? v === xqs[t].bsums[16ib+2s+1] + xqs[t].bsums[16ib+2s+2] : v === Int16(0)
    end
    @test ok
    @test p.m[3] === Int16(-4096) && p.m[5] === Int16(4064)
    xps = map(_MQ._q6k_prep, xqs)
    @test _MQ._multi_q45k_tables(xps).m == p.m
    # a short mins table is refused
    for ty in (_MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K)
        rb = _MQ._qrow_bytes(ty, 256 * nb)
        w = _mq_random_rows(rng, ty, 2, nb)
        for t0 in 1:T:11
            @test _mq_check_tiles(ty, w, 2, rb, xqs, t0, min(T, 12 - t0)) == 0
            @test _mq_check_tiles(ty, w, 2, rb, xps, t0, min(T, 12 - t0)) == 0
        end
        out = zeros(Float32, 11)
        bad = _MQ._MultiQ45K(p.d, p.m[1:(end-1)])
        @test_throws ErrorException _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true, bad)
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

# Measured inside a function: a loop at testset scope allocates for its own
# boxing and would measure the harness.
_mq_decoded_row_bytes(out, u, ty, w, woff, xqs, n, p) =
    @allocated _MQ._dot_iq2_decoded_row!(out, u, ty, w, woff, xqs, n, p)

@testset "multi-column dot: IQ2 rows decoded once per row" begin
    # Random rows plus an all-0xff row (largest grid index, every sign, top
    # scale) at d = 1.0, against activations at -128 and 127: every column
    # bit-equal to _dot_q8k, for each type with a decoded-row path.
    rng = Xoshiro(0x2dec)
    nb = 3
    for ty in (_MQ.GGML_TYPE_IQ2_XS, _MQ.GGML_TYPE_IQ2_XXS, _MQ.GGML_TYPE_IQ2_S)
        @test _MQ._iq2_decoded_type(ty)
        rb = _MQ._qrow_bytes(ty, 256 * nb)
        bs = div(rb, nb)
        w = _mq_random_rows(rng, ty, 3, nb)
        hi = fill(0xff, rb)
        for blk in 0:(nb-1)
            hi[bs*blk+1] = 0x00
            hi[bs*blk+2] = 0x3c
        end
        append!(w, hi)
        xqs = _mq_acts(rng, 256 * nb, 23, ty)
        xqs[1].qs .= Int8(-128)
        xqs[2].qs .= Int8(127)
        for x in xqs[1:2], i in eachindex(x.bsums)
            x.bsums[i] = sum(Int16, @view x.qs[(16(i-1)+1):(16i)])
        end
        p = _MQ._multi_packed(xqs)
        u = UInt8[]
        for n in (1, 9, 16, 17, 23), j in 0:3
            out = fill(NaN32, n)
            _MQ._dot_iq2_decoded_row!(out, u, ty, w, j * rb, xqs, n, p)
            @test all(c -> out[c] === _MQ._dot_q8k(ty, w, j * rb, xqs[c], true), 1:n)
        end
        @test length(u) == _MQ._iq2_decoded_bytes(nb)
        o23 = zeros(Float32, 23)
        @test_throws ErrorException _MQ._dot_iq2_decoded_row!(o23, u, ty, w, 0, xqs, 24, p)
        @test_throws ErrorException _MQ._dot_iq2_decoded!(o23, UInt8[], xqs, 1, Val(8), p)
        _MQ._dot_iq2_decoded_row!(o23, u, ty, w, rb, xqs, 23, p)
        @test _mq_decoded_row_bytes(o23, u, ty, w, rb, xqs, 23, p) == 0
    end
    @test !_MQ._iq2_decoded_type(_MQ.GGML_TYPE_IQ3_XXS)
    xqs = _mq_acts(rng, 256, 9, _MQ.GGML_TYPE_IQ2_XS)
    w = _mq_random_rows(rng, _MQ.GGML_TYPE_IQ3_XXS, 1, 1)
    @test_throws ErrorException _MQ._dot_iq2_decoded_row!(
        zeros(Float32, 9), UInt8[], _MQ.GGML_TYPE_IQ3_XXS, w, 0, xqs, 9, _MQ._multi_packed(xqs))
end

@testset "multi-column dot: Q6_K decoded row blocks" begin
    # Rows at u = 0 and 63 with scales at -128 and 127, plus random rows, decoded
    # in blocks and run against activations at -128 and 127: every value bit-equal
    # to _dot_q8k, with and without the d table, for every tile size.
    _MQ._multi_kernels_ok() || return
    rng = Xoshiro(0x6dec)
    nb = 2
    ty = _MQ.GGML_TYPE_Q6_K
    bsz = _MQ._iq_block_bytes(ty)
    rb = _MQ._qrow_bytes(ty, 256 * nb)
    w = _mq_random_rows(rng, ty, 5, nb)
    for (r, fillb) in enumerate((0xff, 0x00))
        row = fill(fillb, rb)
        for blk in 0:(nb-1)
            o = blk * bsz
            for k in 1:16
                row[o+192+k] = isodd(k + r) ? 0x80 : 0x7f
            end
            row[o+209] = 0x00
            row[o+210] = 0x3c
        end
        append!(w, row)
    end
    nrows = div(length(w), rb)
    xqs = _mq_acts(rng, 256 * nb, 17, ty)
    for (c, v) in ((1, Int8(-128)), (2, Int8(127)))
        xqs[c].qs .= v
        xqs[c].d .= 1.0f0
        for i in eachindex(xqs[c].bsums)
            xqs[c].bsums[i] = sum(Int16, @view xqs[c].qs[(16(i-1)+1):(16i)])
        end
    end
    xps = map(_MQ._q6k_prep, xqs)
    ref = [_MQ._dot_q8k(ty, w, (j - 1) * rb, xps[t], true) for t in 1:17, j in 1:nrows]
    m = Memory{UInt8}(undef, nrows * nb * (_MQ._Q6K_DEC_BYTES + 4))
    buf = zeros(Float32, 17)
    dtab = _MQ._multi_d_table(xps)
    bad = 0
    for (j0, j1) in ((1, nrows), (2, 4), (nrows, nrows)), dt in (nothing, dtab)
        _MQ._q6k_decode_rows!(m, w, 0, rb, nb, j0, j1)
        n = j1 - j0 + 1
        starts = dt === nothing ? (1, 2) : (1, 9)
        for t0 in starts, K in 1:min(8, 17 - t0 + 1)
            dst = fill(NaN32, 17, nrows)
            _MQ._q6k_dec_tile!(dst, buf, m, n, nb, j0, xps, t0, K, dt)
            for j in j0:j1, t in t0:(t0+K-1)
                bad += reinterpret(UInt32, dst[t, j]) != reinterpret(UInt32, ref[t, j] + 0.0f0)
            end
        end
    end
    @test bad == 0
    @test_throws ErrorException _MQ._q6k_decode_rows!(Memory{UInt8}(undef, 8), w, 0, rb, nb, 1, 2)
    @test_throws ErrorException _MQ._q6k_decode_rows!(m, w, 0, rb + 1, nb, 1, 2)
    @test_throws ErrorException _MQ._q6k_dec_tile!(zeros(Float32, 17, nrows), buf, m, 1, nb, 1, xps, 1, 9, nothing)
    # The decoded and the undecoded prefill give the same bits.
    q = _MQ.QuantMatrix(w, 0, ty, 256 * nb, nrows, rb, Int64(256))
    x = randn(rng, Float32, 19, 256 * nb)
    a = zeros(Float32, 19, nrows)
    b = zeros(Float32, 19, nrows)
    saved = _MQ._PREFILL_MULTI[]
    try
        _MQ._PREFILL_MULTI[] = Int8(1)
        withenv(() -> _MQ._quant_matmul!(a, x, q), "GESSO_DISABLE" => nothing)
        withenv(() -> _MQ._quant_matmul!(b, x, q), "GESSO_DISABLE" => "q6k_decode")
    finally
        _MQ._PREFILL_MULTI[] = saved
    end
    @test all(reinterpret.(UInt32, a) .== reinterpret.(UInt32, b))
    @test _MQ._q6k_dec_rows(56) * 56 * (_MQ._Q6K_DEC_BYTES + 4) <= _MQ._Q6K_DEC_BLOCK_BYTES
    @test _MQ._q6k_dec_rows(1) == 32
end

@testset "multi-column dot: Q4_K and Q5_K row blocks" begin
    # _q45k_rows_tile! over row ranges and every tile size, plain and prepped
    # columns, with no table, the d table and the d + mins tables: every value
    # bit-equal to _dot_q8k. Then the prefill with it on and off.
    _MQ._multi_kernels_ok() || return
    rng = Xoshiro(0x45b0)
    for ty in (_MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K)
        nb = 2
        q5 = Val(ty == _MQ.GGML_TYPE_Q5_K)
        rb = _MQ._qrow_bytes(ty, 256 * nb)
        w = _mq_random_rows(rng, ty, 7, nb)
        nrows = div(length(w), rb)
        xqs = _mq_acts(rng, 256 * nb, 17, ty)
        xqs[1].qs .= Int8(-128)
        xqs[2].qs .= Int8(127)
        for c in 1:2, i in eachindex(xqs[c].bsums)
            xqs[c].bsums[i] = sum(Int16, @view xqs[c].qs[(16(i-1)+1):(16i)])
        end
        bad = 0
        for cols in (xqs, map(_MQ._q6k_prep, xqs))
            ref = [_MQ._dot_q8k(ty, w, (j - 1) * rb, cols[t], true) for t in 1:17, j in 1:nrows]
            for dt in (nothing, _MQ._multi_d_table(cols), _MQ._multi_q45k_tables(cols))
                starts = dt === nothing ? (1, 2) : (1, 9)
                for (b0, b1) in ((1, nrows), (2, 4), (nrows, nrows)), t0 in starts
                    for K in 1:min(8, 17 - t0 + 1)
                        dst = fill(NaN32, 17, nrows)
                        _MQ._q45k_rows_tile!(dst, w, 0, rb, b0, b1, cols, t0, K, q5, dt)
                        for j in 1:nrows, t in 1:17
                            want = b0 <= j <= b1 && t0 <= t < t0 + K ? ref[t, j] + 0.0f0 : NaN32
                            bad += reinterpret(UInt32, dst[t, j]) != reinterpret(UInt32, want)
                        end
                    end
                end
            end
        end
        @test bad == 0
        @test_throws ErrorException _MQ._q45k_rows_tile!(zeros(Float32, 17, nrows), w, 0, rb + 1, 1, 2, xqs, 1, 8, q5, nothing)
        @test_throws BoundsError _MQ._q45k_rows_tile!(zeros(Float32, 17, nrows), w, 0, rb, 1, nrows + 1, xqs, 1, 8, q5, nothing)
        @test_throws ErrorException _MQ._q45k_rows_tile!(zeros(Float32, 17, nrows + 1), w, 0, rb, 1, nrows + 1, xqs, 1, 8, q5, nothing)
        @test_throws ErrorException _MQ._q45k_rows_tile!(zeros(Float32, 17, nrows), w, 0, rb, 1, 2, xqs, 1, 9, q5, nothing)
        @test_throws ErrorException _MQ._q45k_rows_tile!(zeros(Float32, 17, nrows), w, 0, rb, 1, 2, xqs, 2, 8, q5, _MQ._multi_d_table(xqs))
        q = _MQ.QuantMatrix(w, 0, ty, 256 * nb, nrows, rb, Int64(256))
        x = randn(rng, Float32, 19, 256 * nb)
        a = zeros(Float32, 19, nrows)
        b = zeros(Float32, 19, nrows)
        saved = _MQ._PREFILL_MULTI[]
        try
            _MQ._PREFILL_MULTI[] = Int8(1)
            withenv(() -> _MQ._quant_matmul!(a, x, q), "GESSO_DISABLE" => nothing)
            withenv(() -> _MQ._quant_matmul!(b, x, q), "GESSO_DISABLE" => "q45k_rows")
        finally
            _MQ._PREFILL_MULTI[] = saved
        end
        @test all(reinterpret.(UInt32, a) .== reinterpret.(UInt32, b))
    end
end

@testset "multi-column dot: the packed Q8_0-family tables" begin
    rng = Xoshiro(0x51b0)
    T = _MQ._MULTI_TILE
    for ty in (_MQ.GGML_TYPE_Q8_0, _MQ.GGML_TYPE_Q4_0, _MQ.GGML_TYPE_IQ4_NL, _MQ.GGML_TYPE_Q2_0)
        rb = _MQ._qrow_bytes(ty, 512)
        w = _mq_random_rows(rng, ty, 2, 2)
        xqs = _mq_acts(rng, 512, 11, ty)
        p = _MQ._multi_packed(xqs)
        nch = length(xqs[1].d)
        @test length(p.q) == 32 * T * nch * cld(11, T)
        @test all(t -> all(j -> p.q[32*(T*(nch*((t-1)÷T)+j)+(t-1)%T)+1] === xqs[t].qs[32j+1], 0:(nch-1)), 1:11)
        out = zeros(Float32, 11)
        for t0 in 1:T:11
            K = min(T, 12 - t0)
            _MQ._dot_q8k_multi!(out, ty, w, rb, xqs, t0, K, true, p)
            @test all(c -> out[t0+c-1] === _MQ._dot_q8k(ty, w, rb, xqs[t0+c-1], true), 1:K)
        end
        # quants that do not come in the d table's 32-value blocks are refused
        bad = _MQ._MultiPacked(p.d, p.q[1:(end-32)])
        @test_throws ErrorException _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true, bad)
        _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true, p)
        @test (@allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true, p)) == 0
    end
end
@testset "multi-column dot: the packed Q5_1 table" begin
    rng = Xoshiro(0x51b1)
    T = _MQ._MULTI_TILE
    ty = _MQ.GGML_TYPE_Q5_1
    for n in (8, 64, 65, 72)
        rb = _MQ._qrow_bytes(ty, 512)
        w = _mq_random_rows(rng, ty, 2, 2)
        xqs = _mq_acts(rng, 512, n, ty)
        p = _MQ._multi_packed(xqs)
        nch = length(xqs[1].d)
        # s holds each block's activation sum times its scale, laid out like d
        @test length(p.s) == length(p.d)
        @test all(t -> all(j -> p.s[T*(nch*((t-1)÷T)+j)+(t-1)%T+1] ===
            xqs[t].d[j+1] * Float32(sum(Int, @view xqs[t].qs[32j+1:32j+32])), 0:(nch-1)), 1:n)
        out = zeros(Float32, n)
        for t0 in 1:T:n
            K = min(T, n + 1 - t0)
            _MQ._dot_q8k_multi!(out, ty, w, rb, xqs, t0, K, true, p)
            @test all(c -> out[t0+c-1] === _MQ._dot_q8k(ty, w, rb, xqs[t0+c-1], true), 1:K)
        end
    end
    xqs = _mq_acts(rng, 512, 8, ty)
    w = _mq_random_rows(rng, ty, 1, 2)
    p = _MQ._multi_packed(xqs)
    out = zeros(Float32, 8)
    # without the s table the tile falls back to the single-column kernel
    _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true, _MQ._MultiPacked(p.d, p.q))
    @test all(c -> out[c] === _MQ._dot_q8k(ty, w, 0, xqs[c], true), 1:8)
    @test_throws ErrorException _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true,
        _MQ._MultiPacked(p.d, p.q[1:(end-32)], p.s))
    @test_throws ErrorException _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 8, true,
        _MQ._MultiPacked(p.d, p.q, p.s[1:(end-1)]))
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
        _MQ.GGML_TYPE_Q5_1,
        _MQ.GGML_TYPE_Q2_0,
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
        if _mq_uses_table(ty, 1, 4) && ty in (_MQ.GGML_TYPE_Q4_K, _MQ.GGML_TYPE_Q5_K)
            mt = _MQ._multi_q45k_tables(xqs)
            _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true, mt)
            @test (@allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true, mt)) == 0
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

@testset "prefill: ragged sequence lengths and row counts equal the per-token path" begin
    rng = Xoshiro(0xfade)
    nb = 2
    ne0 = 256 * nb
    for (name, ty) in _MQ_TYPES, nrows in (1, 33), seq in (1, 7, 8, 9, 63, 512)
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
        @test all(reinterpret.(UInt32, dst_multi) .== reinterpret.(UInt32, dst_per_token))
    end
end

@testset "prefill_multi switch is read as a flag" begin
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
