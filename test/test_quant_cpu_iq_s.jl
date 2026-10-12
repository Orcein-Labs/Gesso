# Native CPU IQ2_XXS, IQ2_S, IQ3_S, IQ1_S and IQ1_M dot products
# (src/Inference/quant_cpu_iq_s.jl):
#   * grid tables: sizes, value ranges the AVX2 saturation argument relies on
#   * scalar references against the F32 dequant references (ports of
#     dequantize_row_*), at tolerance
#   * AVX2 kernels bit-exact against the scalar references on random blocks and
#     on edge fields
#   * F32 dequant of IQ2_XXS and IQ1_S bit-equal to ggml's to_float (env
#     GESSO_GGML_DIR; skipped if libggml-base.so is absent)
#   * zero allocations per dot; GEMV threading does not change any bit
#   * real tensors: the Qwen3.8-27B UD-IQ2_M GGUF (env GESSO_QWEN38_27B_IQ2M_GGUF)
#     for IQ2_S/IQ3_S/IQ1_M, and the Saluki 27B GGUF (env GESSO_SALUKI_GGUF) for
#     IQ2_XXS/IQ1_S. Each is skipped if its env var is unset. Header and rows only.

using Random: Xoshiro
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _IQS = Gesso.Inference
const _IQS_TYPES = (
    ("IQ2_XXS", _IQS.GGML_TYPE_IQ2_XXS),
    ("IQ2_S", _IQS.GGML_TYPE_IQ2_S),
    ("IQ3_S", _IQS.GGML_TYPE_IQ3_S),
    ("IQ1_S", _IQS.GGML_TYPE_IQ1_S),
    ("IQ1_M", _IQS.GGML_TYPE_IQ1_M),
)
# The two formats added for the Saluki GGUF (IQ2_XXS and IQ1_S).
const _IQS_NEW_TYPES = (
    ("IQ2_XXS", _IQS.GGML_TYPE_IQ2_XXS),
    ("IQ1_S", _IQS.GGML_TYPE_IQ1_S),
)
# Formats present in the Qwen3.8-27B UD-IQ2_M file.
const _IQS_QWEN_TYPES = (
    ("IQ2_S", _IQS.GGML_TYPE_IQ2_S),
    ("IQ3_S", _IQS.GGML_TYPE_IQ3_S),
    ("IQ1_M", _IQS.GGML_TYPE_IQ1_M),
)

# Random valid rows: arbitrary bytes in every field, with finite small f16
# super-scales. IQ2_S and IQ3_S keep d at byte 0; IQ1_M has no d field, so its
# scale nibbles are written instead.
function _iqs_random_rows(rng, ggml_type::Int32, nrows::Int, nb::Int; scale_lo=0.005f0, scale_span=0.05f0)
    bsz = _IQS._iq_block_bytes(ggml_type)
    w = rand(rng, UInt8, nrows * nb * bsz)
    for blk in 0:(nrows * nb - 1)
        dv = scale_lo + scale_span * rand(rng, Float32)
        if ggml_type == _IQS.GGML_TYPE_IQ1_M
            _IQS._iq1m_write_scale!(w, blk * bsz, dv)
        else
            bits = reinterpret(UInt16, Float16(dv))
            w[blk * bsz + 1] = UInt8(bits & 0xff)
            w[blk * bsz + 2] = UInt8(bits >> 8)
        end
    end
    return w
end

# F64 reference dot of one row: dequantized weights against dequantized Q8_K
# activations, so only float rounding separates it from the integer kernels.
function _iqs_ref_dot(ggml_type::Int32, w, woff::Int, ne0::Int, xq)
    wdq = zeros(Float32, ne0)
    _IQS._iqs_dequant_row!(wdq, w, woff, ggml_type, ne0)
    xd = _IQS.dequantize_q8_K(xq)
    return sum(Float64.(wdq) .* Float64.(xd))
end

# Sum of the term magnitudes |w_i x_i|: the scale that float rounding error of
# the dot product is bounded by.
function _iqs_ref_abs(ggml_type::Int32, w, woff::Int, ne0::Int, xq)
    wdq = zeros(Float32, ne0)
    _IQS._iqs_dequant_row!(wdq, w, woff, ggml_type, ne0)
    xd = _IQS.dequantize_q8_K(xq)
    return sum(abs.(Float64.(wdq) .* Float64.(xd)))
end

@testset "IQ2_XXS / IQ2_S / IQ3_S / IQ1_S / IQ1_M block sizes" begin
    @test _IQS._iq_block_bytes(_IQS.GGML_TYPE_IQ2_XXS) == 66
    @test _IQS._iq_block_bytes(_IQS.GGML_TYPE_IQ1_S) == 50
    @test _IQS._qrow_bytes(_IQS.GGML_TYPE_IQ2_XXS, 512) == 2 * 66
    @test _IQS._qrow_bytes(_IQS.GGML_TYPE_IQ1_S, 512) == 2 * 50
    @test _IQS._iq_block_bytes(_IQS.GGML_TYPE_IQ2_S) == 2 + 64 + 8 + 8
    @test _IQS._iq_block_bytes(_IQS.GGML_TYPE_IQ3_S) == 2 + 64 + 8 + 32 + 4
    @test _IQS._iq_block_bytes(_IQS.GGML_TYPE_IQ1_M) == 32 + 16 + 8
    @test _IQS._qrow_bytes(_IQS.GGML_TYPE_IQ2_S, 512) == 2 * 82
end

@testset "grid tables: sizes and value ranges" begin
    @test length(_IQS.IQ2XXS_GRID) == 256
    @test length(_IQS.IQ2S_GRID) == 1024
    @test length(_IQS.IQ3S_GRID) == 512
    @test length(_IQS.IQ1S_GRID) == 2048
    # The AVX2 pmaddubsw step cannot saturate when grid bytes are at most 127
    # (two products of grid byte times Q8_K quant fit in int16).
    @test maximum(UInt8((g >> (8 * j)) & 0xff) for g in _IQS.IQ2XXS_GRID for j in 0:7) <= 43
    @test maximum(UInt8((g >> (8 * j)) & 0xff) for g in _IQS.IQ2S_GRID for j in 0:7) <= 43
    @test maximum(UInt8((g >> (8 * j)) & 0xff) for g in _IQS.IQ3S_GRID for j in 0:3) <= 15
    # IQ1_M / IQ1_S grid values are the signed set {-1, 0, +1}
    vals = Set(UInt8((g >> (8 * j)) & 0xff) for g in _IQS.IQ1S_GRID for j in 0:7)
    @test vals == Set(UInt8[0x00, 0x01, 0xff])
    # sign mask for s: byte j is 0xff when bit j of s is set, else 0x01
    for s in (0x00, 0x01, 0x05, 0x80, 0xa7, 0xff)
        expected = reduce(|, (UInt64(((s >> j) & 1) == 1 ? 0xff : 0x01) << (8 * j) for j in 0:7))
        @test _IQS._IQ_SIGN8_FULL_TAB[Int(s) + 1] == expected
    end
    @test _IQS._IQ_SIGN8_FULL_TAB[1] == 0x0101010101010101
    @test _IQS._IQ_SIGN8_FULL_TAB[256] == 0xffffffffffffffff
end

@testset "IQ1_M scale nibbles round-trip through the f16 super-scale" begin
    rng = Xoshiro(20)
    for _ in 1:200
        dv = Float16(0.001f0 + 0.1f0 * rand(rng, Float32))
        w = zeros(UInt8, 56)
        for k in 0:7
            w[48 + k + 1] = rand(rng, UInt8) & 0x0f   # low nibbles: 3-bit sub-block scales
        end
        before = copy(w)
        _IQS._iq1m_write_scale!(w, 0, Float32(dv))
        s = ntuple(k -> UInt16(w[49 + 2 * (k - 1)]) | (UInt16(w[50 + 2 * (k - 1)]) << 8), 4)
        @test reinterpret(Float16, _IQS._iq1m_scale_bits(s[1], s[2], s[3], s[4])) == dv
        # the sub-block scales in the low nibbles are untouched
        @test all(w[i] & 0x0f == before[i] & 0x0f for i in 1:56)
    end
end

@testset "scalar references vs F32 dequant reference (tolerance)" begin
    rng = Xoshiro(21)
    nb, nrow = 4, 6
    for (name, ty) in _IQS_TYPES
        W = _iqs_random_rows(rng, ty, nrow, nb)
        x = randn(rng, Float32, 256 * nb)
        xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(nb), x)
        rb = _IQS._qrow_bytes(ty, 256 * nb)
        ys = Float64[_IQS._dot_q8k(ty, W, (j - 1) * rb, xq, false) for j in 1:nrow]
        yref = Float64[_iqs_ref_dot(ty, W, (j - 1) * rb, 256 * nb, xq) for j in 1:nrow]
        tabs = Float64[_iqs_ref_abs(ty, W, (j - 1) * rb, 256 * nb, xq) for j in 1:nrow]
        @testset "$name" begin
            @test all(abs.(ys .- yref) .<= 1.0e-5 .* tabs)
        end
    end
end

# Independent oracle: ggml's own to_float (libggml-base.so, via GESSO_GGML_DIR).
# Random rows plus an all-ones-field row with d = 1.0, so the sign and grid
# paths are both exercised at full magnitude.
@testset "F32 dequant of IQ2_XXS / IQ1_S bit-equal to ggml to_float" begin
    ggml_dir = get(ENV, "GESSO_GGML_DIR", "")
    if !isfile(joinpath(ggml_dir, "libggml-base.so"))
        _skip("set GESSO_GGML_DIR to a directory with libggml-base.so — IQ2_XXS/IQ1_S dequant vs ggml to_float skipped")
    else
        rng = Xoshiro(29)
        for (name, ty) in _IQS_NEW_TYPES
            nb = 3
            ne0 = 256 * nb
            rb = _IQS._qrow_bytes(ty, ne0)
            W = _iqs_random_rows(rng, ty, 2, nb)
            bsz = _IQS._iq_block_bytes(ty)
            sat = fill(0xff, 2 * rb)
            for ib in 0:(2 * nb - 1)
                sat[ib * bsz + 1] = 0x00     # d = 1.0 (f16 0x3c00) for every block
                sat[ib * bsz + 2] = 0x3c
            end
            for (kind, data) in (("random", W), ("saturated", sat))
                for j in 1:2
                    mine = Vector{Float32}(undef, ne0)
                    _IQS._iqs_dequant_row!(mine, data, (j - 1) * rb, ty, ne0)
                    row = data[((j - 1) * rb + 1):(j * rb)]
                    ggml = _IQS._dequant_f32(row, 0, ty, ne0)
                    @testset "$name $kind row $j" begin
                        @test reinterpret.(UInt32, mine) == reinterpret.(UInt32, ggml)
                    end
                end
            end
        end
    end
end

@testset "AVX2 kernels bit-exact against scalar reference" begin
    if !_IQS._cpu_has_avx2()
        _skip("no AVX2 on this CPU: IQ2_S/IQ3_S/IQ1_M AVX2-vs-scalar check skipped")
    else
        rng = Xoshiro(22)
        for (name, ty) in _IQS_TYPES
            ntrial = 0
            nmismatch = 0
            for trial in 1:300
                nb = 1 + (trial % 4)
                W = _iqs_random_rows(rng, ty, 4, nb; scale_lo=0.0f0, scale_span=0.1f0 + 2 * rand(rng, Float32))
                x = randn(rng, Float32, 256 * nb) .* Float32(0.1 + 10 * rand(rng))
                if trial % 7 == 0       # extreme activations: quants saturate at ±127
                    x .= x .* 1000.0f0
                end
                if trial % 11 == 0      # zero activations in a sub-block
                    x[1:64] .= 0.0f0
                end
                xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(nb), x)
                rb = _IQS._qrow_bytes(ty, 256 * nb)
                for j in 1:4
                    r = _IQS._dot_q8k(ty, W, (j - 1) * rb, xq, false)
                    a = _IQS._dot_q8k(ty, W, (j - 1) * rb, xq, true)
                    ntrial += 1
                    nmismatch += reinterpret(UInt32, r) != reinterpret(UInt32, a)
                end
            end
            @testset "$name" begin
                @test nmismatch == 0
                @test ntrial == 1200
            end
        end
    end
end

@testset "edge fields: all-ones, all-zero and sign-saturated blocks" begin
    if _IQS._cpu_has_avx2()
        for (name, ty) in _IQS_TYPES
            bsz = _IQS._iq_block_bytes(ty)
            nb = 2
            # every field 0xff (grid index 511 / 1023, all signs set, scale nibbles 15)
            for fill_byte in (0x00, 0xff, 0x55, 0xaa)
                W = fill(UInt8(fill_byte), nb * bsz)
                for ib in 0:(nb - 1)
                    if ty == _IQS.GGML_TYPE_IQ1_M
                        _IQS._iq1m_write_scale!(W, ib * bsz, 0.25f0)
                    else
                        W[ib * bsz + 1] = 0x00
                        W[ib * bsz + 2] = 0x3c    # d = 1.0 in f16
                    end
                end
                x = randn(Xoshiro(23 + Int(fill_byte)), Float32, 256 * nb)
                xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(nb), x)
                r = _IQS._dot_q8k(ty, W, 0, xq, false)
                a = _IQS._dot_q8k(ty, W, 0, xq, true)
                @test reinterpret(UInt32, r) == reinterpret(UInt32, a)
                ref = _iqs_ref_dot(ty, W, 0, 256 * nb, xq)
                @test abs(Float64(r) - ref) <= 1.0e-5 * _iqs_ref_abs(ty, W, 0, 256 * nb, xq)
            end
            # d = 0 everywhere: the row contributes exactly zero
            W = fill(UInt8(0xff), nb * bsz)
            for ib in 0:(nb - 1)
                if ty == _IQS.GGML_TYPE_IQ1_M
                    _IQS._iq1m_write_scale!(W, ib * bsz, 0.0f0)
                else
                    W[ib * bsz + 1] = 0x00
                    W[ib * bsz + 2] = 0x00
                end
            end
            xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(nb), fill(1.0f0, 256 * nb))
            r = _IQS._dot_q8k(ty, W, 0, xq, false)
            a = _IQS._dot_q8k(ty, W, 0, xq, true)
            @test r === 0.0f0
            @test reinterpret(UInt32, r) == reinterpret(UInt32, a)
        end
    else
        _skip("no AVX2 on this CPU: edge-field bit-exact check skipped")
    end
end

@testset "dot kernels do not allocate" begin
    rng = Xoshiro(24)
    for (name, ty) in _IQS_TYPES
        nb = 4
        W = _iqs_random_rows(rng, ty, 1, nb)
        x = randn(rng, Float32, 256 * nb)
        xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(nb), x)
        for simd in (false, true)
            if simd && !_IQS._cpu_has_avx2()
                continue
            end
            _IQS._dot_q8k(ty, W, 0, xq, simd)
            @test (@allocated _IQS._dot_q8k(ty, W, 0, xq, simd)) == 0
        end
    end
end

@testset "GEMV: threading does not change any output bit" begin
    rng = Xoshiro(25)
    nb, nrow = 3, 37
    for (name, ty) in _IQS_TYPES
        W = _iqs_random_rows(rng, ty, nrow, nb)
        x = randn(rng, Float32, 256 * nb)
        y1 = zeros(Float32, nrow)
        _IQS._qgemv_q8k_f32!(y1, x, W, ty, 256 * nb, nrow; threaded=false)
        xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(nb), x)
        for nc in (1, 3, 7, nrow)
            yt = zeros(Float32, nrow)
            _IQS._qgemv_q8k_threaded!(yt, xq, W, ty, 256 * nb, nrow; nchunks=nc)
            @test reinterpret.(UInt32, yt) == reinterpret.(UInt32, y1)
        end
        # the F32 reference agrees within activation-quantization error
        y32 = zeros(Float32, nrow)
        _IQS._qgemv_bytes!(y32, x, W, ty, 256 * nb, nrow, 1.0f0, 0.0f0)
        @test maximum(abs, y1 .- y32) <= 2.0e-2 * maximum(abs, y32)
    end
end

@testset "GEMV: alpha and beta follow _qgemv_bytes!" begin
    rng = Xoshiro(26)
    nb, nrow = 2, 9
    for (name, ty) in _IQS_TYPES
        W = _iqs_random_rows(rng, ty, nrow, nb)
        x = randn(rng, Float32, 256 * nb)
        xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(nb), x)
        base = zeros(Float32, nrow)
        _IQS._qgemv_q8k!(base, xq, W, ty, 256 * nb, nrow; simd=false)
        y = fill(2.0f0, nrow)
        _IQS._qgemv_q8k!(y, xq, W, ty, 256 * nb, nrow; alpha=3.0f0, beta=0.5f0, simd=false)
        @test y ≈ 3.0f0 .* base .+ 0.5f0 * 2.0f0 rtol = 1e-6
    end
end

@testset "QuantMatrix matmul uses the native kernels, not the ggml fallback" begin
    rng = Xoshiro(28)
    nb, nrow, seq = 2, 12, 3
    ne0 = 256 * nb
    for (name, ty) in _IQS_TYPES
        W = _iqs_random_rows(rng, ty, nrow, nb)
        q = _IQS.QuantMatrix(W, 0, ty, ne0, nrow, _IQS._qrow_bytes(ty, ne0), C_NULL, Int64(256))
        X = randn(rng, Float32, seq, ne0)
        dst = zeros(Float32, seq, nrow)
        _IQS._quant_matmul!(dst, X, q)
        @test !(ty in _IQS._QUANT_FALLBACK_TYPES)
        ref = zeros(Float64, seq, nrow)
        for j in 1:nrow
            wd = zeros(Float32, ne0)
            _IQS._iqs_dequant_row!(wd, W, (j - 1) * _IQS._qrow_bytes(ty, ne0), ty, ne0)
            for t in 1:seq
                ref[t, j] = sum(Float64.(wd) .* Float64.(X[t, :]))
            end
        end
        @test sqrt(sum(abs2, Float64.(dst) .- ref) / sum(abs2, ref)) <= 2.0e-2
    end
end

# Real rows from one GGUF, one tensor per format in `types`, found by ggml type
# id. Reads the header and up to 16 rows of each tensor; the model is never loaded.
function _iqs_real_tensor_check(path::AbstractString, types, seed::Int)
    hdr = _IQS.gguf_header(path)
    rng = Xoshiro(seed)
    for (name, ty) in types
        idx = findfirst(t -> t.ggml_type == ty, hdr.tensors)
        if idx === nothing
            _skip("no $name tensor in $(basename(path))")
            continue
        end
        info = hdr.tensors[idx]
        ne0, ne1 = info.dims[1], info.dims[2]
        rb = _IQS._qrow_bytes(ty, ne0)
        base = Int(hdr.data_start) + Int(info.offset)
        rows = unique(round.(Int, range(1, ne1; length=min(16, ne1))))
        x = randn(rng, Float32, ne0)
        xq = _IQS.quantize_q8_K!(_IQS.Q8KVec(div(ne0, 256)), x)
        err2 = 0.0
        sig2 = 0.0
        nmis = 0
        for j in rows
            off = base + (j - 1) * rb
            ys = _IQS._dot_q8k(ty, hdr.data, off, xq, false)
            ya = _IQS._dot_q8k(ty, hdr.data, off, xq, true)
            nmis += reinterpret(UInt32, ys) != reinterpret(UInt32, ya)
            wdq = zeros(Float32, ne0)
            _IQS._iqs_dequant_row!(wdq, hdr.data, off, ty, ne0)
            yexact = sum(Float64.(wdq) .* Float64.(x))
            ydeq = _iqs_ref_dot(ty, hdr.data, off, ne0, xq)
            # float rounding scales with the sum of the term magnitudes, not with the result
            tabs = sum(abs.(Float64.(wdq) .* Float64.(_IQS.dequantize_q8_K(xq))))
            @test abs(Float64(ys) - ydeq) <= 1.0e-5 * tabs
            err2 += (Float64(ys) - yexact)^2
            sig2 += yexact^2
        end
        relrms = sqrt(err2 / sig2)
        println(
            "real ", basename(path), " ", info.name, " type=", name, " ne0=", ne0, " ne1=", ne1,
            " rows=", length(rows), " rel_rms_err_vs_f32=", relrms,
            " avx2_eq_scalar=", nmis == 0,
        )
        @test nmis == 0
        @test relrms <= 2.0e-2
    end
end

@testset "real tensors: IQ2_S / IQ3_S / IQ1_M vs F32 reference" begin
    path = get(ENV, "GESSO_QWEN38_27B_IQ2M_GGUF", "")
    if !isfile(path)
        _skip("set GESSO_QWEN38_27B_IQ2M_GGUF to the Qwen3.8-27B UD-IQ2_M GGUF — real IQ2_S/IQ3_S/IQ1_M rows skipped")
    else
        _iqs_real_tensor_check(path, _IQS_QWEN_TYPES, 27)
    end
end

@testset "real tensors: Saluki IQ2_XXS / IQ1_S vs F32 reference" begin
    path = get(ENV, "GESSO_SALUKI_GGUF", "")
    if !isfile(path)
        _skip("set GESSO_SALUKI_GGUF to the Saluki 27B GGUF — real IQ2_XXS/IQ1_S rows skipped")
    else
        _iqs_real_tensor_check(path, _IQS_NEW_TYPES, 30)
    end
end
