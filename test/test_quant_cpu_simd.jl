# Native CPU quantized dot products (quant_q8k.jl, quant_cpu_simd.jl):
#   * Q8_K quantizer against an independent scalar port of quantize_row_q8_K_ref
#   * scalar reference dots against the existing F32 dequant path
#   * AVX2 kernels bit-exact against the scalar reference, per format
#   * row-chunked GEMV, threading-invariant
#   * accuracy on a real Ornith expert slice (env GESSO_ORNITH35_GGUF; skipped if unset)

using Random: Xoshiro
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _QSIMD = Gesso.Inference
const _QTYPES = (
    ("IQ4_XS", _QSIMD.GGML_TYPE_IQ4_XS),
    ("IQ2_XS", _QSIMD.GGML_TYPE_IQ2_XS),
    ("IQ3_XXS", _QSIMD.GGML_TYPE_IQ3_XXS),
)

# Random valid rows: arbitrary bytes for the grid/sign/scale fields, a finite
# small f16 super-scale per block (random f16 bits would be NaN or Inf often).
function _q_random_rows(rng, ggml_type::Int32, nrows::Int, nb::Int)
    bsz = _QSIMD._iq_block_bytes(ggml_type)
    w = rand(rng, UInt8, nrows * nb * bsz)
    for blk in 0:(nrows * nb - 1)
        d = Float16(0.005f0 + 0.05f0 * rand(rng, Float32))
        bits = reinterpret(UInt16, d)
        w[blk * bsz + 1] = UInt8(bits & 0xff)
        w[blk * bsz + 2] = UInt8(bits >> 8)
    end
    return w
end

# Independent scalar port of quantize_row_q8_K_ref. Uses round(·) (ties to
# even) instead of the magic-number nearest_int, so it does not share the
# implementation under test.
function _q_ref_quantize(x::AbstractVector{Float32})
    nb = length(x) ÷ 256
    qs = zeros(Int8, length(x))
    bs = zeros(Int16, 16 * nb)
    d = zeros(Float32, nb)
    for i in 1:nb
        blk = x[((i - 1) * 256 + 1):(i * 256)]
        amax, mx = 0.0f0, 0.0f0
        for v in blk
            if abs(v) > amax
                amax, mx = abs(v), v
            end
        end
        amax == 0 && continue
        iscale = -127.0f0 / mx
        for j in 1:256
            qs[(i - 1) * 256 + j] = Int8(min(127, round(Int, iscale * blk[j])))
        end
        for j in 1:16
            bs[(i - 1) * 16 + j] = Int16(sum(Int(qs[(i - 1) * 256 + (j - 1) * 16 + ii]) for ii in 1:16))
        end
        d[i] = 1.0f0 / iscale
    end
    return d, qs, bs
end

@testset "Q8_K block layout (ggml block_q8_K: 4 + 256 + 2*16 bytes)" begin
    @test _QSIMD.Q8K_BLOCK_BYTES == 4 + 256 + 32
    @test _QSIMD.QK_K == 256
end

@testset "Q8_K quantizer: rounding helper" begin
    # Julia's round() is ties-to-even, the same rule as nearest_int's magic add
    for f in Float32[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999997f0, 126.5f0, -126.5f0, 3.0f0]
        @test _QSIMD._q8k_nearest_int(f) == Int32(round(f))
    end
    rng = Xoshiro(1)
    fs = [(rand(rng, Float32) - 0.5f0) * 254.0f0 for _ in 1:20000]
    @test all(f -> _QSIMD._q8k_nearest_int(f) == Int32(round(f)), fs)
end

@testset "Q8_K quantizer: hand example" begin
    x = zeros(Float32, 256)
    x[1] = -2.0f0
    y = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(1), x)
    @test y.qs[1] == Int8(-127)
    @test all(==(0), y.qs[2:end])
    @test y.bsums[1] == Int16(-127)
    @test all(==(0), y.bsums[2:end])
    @test y.d[1] ≈ 2.0f0 / 127.0f0 rtol = 1e-6
    # all-zero block: d = 0 and zero quants
    z = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(2), zeros(Float32, 512))
    @test all(==(0.0f0), z.d)
    @test all(==(0), z.qs)
    @test all(==(0), z.bsums)
end

@testset "Q8_K quantizer: bit-equal to independent port" begin
    rng = Xoshiro(2)
    for nb in (1, 3)
        x = randn(rng, Float32, 256 * nb) .* 3.0f0
        x[5] = 0.0f0
        y = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(nb), x)
        d, qs, bs = _q_ref_quantize(x)
        @test y.qs == qs
        @test y.bsums == bs
        @test reinterpret.(UInt32, y.d) == reinterpret.(UInt32, d)
        @test all(-127 <= q <= 127 for q in y.qs)
    end
    # error budget: |x - d*q| ≤ |d|/2 when no value is clipped
    x = randn(rng, Float32, 1024)
    y = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(4), x)
    xr = _QSIMD.dequantize_q8_K(y)
    for b in 1:4
        r = ((b - 1) * 256 + 1):(b * 256)
        @test maximum(abs.(xr[r] .- x[r])) <= 0.5f0 * abs(y.d[b]) * (1 + 1.0f-5)
    end
end

@testset "scalar reference vs F32 dequant path (structure check)" begin
    rng = Xoshiro(3)
    nb, nrow = 4, 8
    for (name, ty) in _QTYPES
        W = _q_random_rows(rng, ty, nrow, nb)
        x = randn(rng, Float32, 256 * nb)
        xq = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(nb), x)
        xdq = _QSIMD.dequantize_q8_K(xq)
        rb = _QSIMD._qrow_bytes(ty, 256 * nb)
        y32 = zeros(Float32, nrow)
        _QSIMD._qgemv_bytes!(y32, xdq, W, ty, 256 * nb, nrow, 1.0f0, 0.0f0)
        ys = [_QSIMD._dot_q8k(ty, W, (j - 1) * rb, xq, false) for j in 1:nrow]
        maxabs = maximum(abs, y32)
        @test maximum(abs, ys .- y32) <= 1.0f-5 * maxabs
    end
end

@testset "AVX2 kernels bit-exact against scalar reference" begin
    if !_QSIMD._cpu_has_avx2()
        _skip("no AVX2 on this CPU: AVX2-vs-scalar bit-exact check skipped")
    else
        rng = Xoshiro(4)
        for (name, ty) in _QTYPES
            ntrial = 0
            nmismatch = 0
            for trial in 1:300
                nb = 1 + (trial % 4)
                W = _q_random_rows(rng, ty, 4, nb)
                x = randn(rng, Float32, 256 * nb) .* Float32(0.1 + 10 * rand(rng))
                if trial % 7 == 0       # extreme activations: quants saturate at ±127
                    x .= x .* 1000.0f0
                end
                if trial % 11 == 0      # zeros and sign-saturated bytes
                    x[1:64] .= 0.0f0
                end
                xq = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(nb), x)
                rb = _QSIMD._qrow_bytes(ty, 256 * nb)
                for j in 1:4
                    r = _QSIMD._dot_q8k(ty, W, (j - 1) * rb, xq, false)
                    a = _QSIMD._dot_q8k(ty, W, (j - 1) * rb, xq, true)
                    ntrial += 1
                    nmismatch += reinterpret(UInt32, r) != reinterpret(UInt32, a)
                end
            end
            @testset "$name" begin
                @test nmismatch == 0
                @test ntrial == 1200
            end
        end
        # a row whose every byte is 0xff in the sign/grid fields, and a zero-scale block
        @testset "saturating fields" begin
            for (name, ty) in _QTYPES
                nb = 2
                bsz = _QSIMD._iq_block_bytes(ty)
                W = fill(UInt8(0xff), nb * bsz)
                for ib in 0:(nb - 1)
                    W[ib * bsz + 1] = 0x00
                    W[ib * bsz + 2] = 0x00   # d = 0: contributes exactly zero
                end
                x = fill(1.0f0, 256 * nb)
                xq = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(nb), x)
                r = _QSIMD._dot_q8k(ty, W, 0, xq, false)
                a = _QSIMD._dot_q8k(ty, W, 0, xq, true)
                @test r === 0.0f0
                @test reinterpret(UInt32, r) == reinterpret(UInt32, a)
            end
        end
    end
end

@testset "dot kernels do not allocate" begin
    rng = Xoshiro(5)
    for (name, ty) in _QTYPES
        nb = 4
        W = _q_random_rows(rng, ty, 1, nb)
        x = randn(rng, Float32, 256 * nb)
        xq = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(nb), x)
        for simd in (false, true)
            if simd && !_QSIMD._cpu_has_avx2()
                continue
            end
            _QSIMD._dot_q8k(ty, W, 0, xq, simd)
            @test (@allocated _QSIMD._dot_q8k(ty, W, 0, xq, simd)) == 0
        end
    end
end

@testset "GEMV: threading does not change any output bit" begin
    rng = Xoshiro(6)
    nb, nrow = 3, 37
    for (name, ty) in _QTYPES
        W = _q_random_rows(rng, ty, nrow, nb)
        x = randn(rng, Float32, 256 * nb)
        y1 = zeros(Float32, nrow)
        _QSIMD._qgemv_q8k_f32!(y1, x, W, ty, 256 * nb, nrow; threaded=false)
        for nc in (1, 3, 7, nrow)
            yt = zeros(Float32, nrow)
            xq = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(nb), x)
            _QSIMD._qgemv_q8k_threaded!(yt, xq, W, ty, 256 * nb, nrow; nchunks=nc)
            @test reinterpret.(UInt32, yt) == reinterpret.(UInt32, y1)
        end
        # the F32 path agrees within activation-quantization error
        y32 = zeros(Float32, nrow)
        _QSIMD._qgemv_bytes!(y32, x, W, ty, 256 * nb, nrow, 1.0f0, 0.0f0)
        @test maximum(abs, y1 .- y32) <= 2.0f-2 * maximum(abs, y32)
    end
end

@testset "GEMV: alpha and beta follow _qgemv_bytes!" begin
    rng = Xoshiro(8)
    nb, nrow = 2, 9
    ty = _QSIMD.GGML_TYPE_IQ2_XS
    W = _q_random_rows(rng, ty, nrow, nb)
    x = randn(rng, Float32, 256 * nb)
    xq = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(nb), x)
    base = zeros(Float32, nrow)
    _QSIMD._qgemv_q8k!(base, xq, W, ty, 256 * nb, nrow; simd=false)
    y = fill(2.0f0, nrow)
    _QSIMD._qgemv_q8k!(y, xq, W, ty, 256 * nb, nrow; alpha=3.0f0, beta=0.5f0, simd=false)
    @test y ≈ 3.0f0 .* base .+ 0.5f0 * 2.0f0 rtol = 1e-6
end

@testset "GEMV: shape checks fail explicitly" begin
    ty = _QSIMD.GGML_TYPE_IQ4_XS
    W = zeros(UInt8, 136 * 2)
    xq = _QSIMD.Q8KVec(1)
    @test_throws ErrorException _QSIMD._qgemv_q8k!(zeros(Float32, 2), xq, W, ty, 100, 2)
    @test_throws ErrorException _QSIMD._qgemv_q8k!(zeros(Float32, 3), xq, W, ty, 256, 2)
    @test_throws ErrorException _QSIMD._qgemv_q8k!(zeros(Float32, 2), xq, W, _QSIMD.GGML_TYPE_F32, 256, 2)
    @test_throws ErrorException _QSIMD._qgemv_q8k!(zeros(Float32, 4), xq, W, ty, 256, 4)
end

# Accuracy on the live Ornith expert slices, against the existing F32 path.
# Header and slice reads only; the 35B model is never loaded.
@testset "Ornith expert slices: Q8_K path vs F32 path" begin
    gguf = get(ENV, "GESSO_ORNITH35_GGUF", "")
    ggml_dir = get(ENV, "GESSO_GGML_DIR", "")
    if !isfile(gguf)
        _skip("set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF — Q8_K accuracy on real slices skipped")
    elseif !isfile(joinpath(ggml_dir, "libggml-base.so"))
        # _expert_slice reads ggml block traits (sizes) from libggml-base; no ggml compute runs here
        _skip("set GESSO_GGML_DIR to a directory with libggml-base.so — Ornith slices skipped")
    else
        hdr = Gesso.gguf_header(gguf)
        rng = Xoshiro(9)
        # layer 0 carries IQ2_XS (gate, up) and IQ3_XXS (down); IQ4_XS is only in blk.1 down
        for tname in (
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.1.ffn_down_exps.weight",
        )
            info = only(t for t in hdr.tensors if t.name == tname)
            slice = _QSIMD._expert_slice(hdr, info)
            n_exp = slice.n_exp
            x = randn(rng, Float32, slice.ne0)
            xq = _QSIMD.quantize_q8_K!(_QSIMD.Q8KVec(div(slice.ne0, 256)), x)
            for e in (1, n_exp)
                q = _QSIMD._quant_expert_bytes(slice, e)
                y32 = zeros(Float32, slice.ne1)
                _QSIMD._qgemv_expert!(y32, x, slice, e)
                y8s = zeros(Float32, slice.ne1)
                _QSIMD._qgemv_q8k!(y8s, xq, q, slice.ggml_type, slice.ne0, slice.ne1; simd=false)
                y8 = zeros(Float32, slice.ne1)
                _QSIMD._qgemv_q8k_threaded!(y8, xq, q, slice.ggml_type, slice.ne0, slice.ne1)
                ymax = maximum(abs, y32)
                abserr = maximum(abs, y8 .- y32)
                relmax = abserr / ymax
                relrms = sqrt(sum(abs2, y8 .- y32) / sum(abs2, y32))
                println(
                    "ornith ", tname, " type=", slice.ggml_type, " expert=", e,
                    " ne0=", slice.ne0, " ne1=", slice.ne1,
                    " max_abs_err=", abserr, " max_rel_err(vs max|y|)=", relmax,
                    " rel_rms_err=", relrms,
                )
                @test reinterpret.(UInt32, y8) == reinterpret.(UInt32, y8s)
                @test relrms <= 2.0e-2
                if _QSIMD._cpu_has_avx2()
                    y8a = zeros(Float32, slice.ne1)
                    _QSIMD._qgemv_q8k!(y8a, xq, q, slice.ggml_type, slice.ne0, slice.ne1; simd=true)
                    @test reinterpret.(UInt32, y8a) == reinterpret.(UInt32, y8s)
                end
            end
        end
    end
end
