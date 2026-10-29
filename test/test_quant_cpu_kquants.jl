# Native CPU K-quant and Q8_0 dot products (src/Inference/quant_cpu_kquants.jl):
#   * Q8_0 quantizer against an independent port of quantize_row_q8_0_ref
#   * scalar dots against an F32 dequant reference (ports of dequantize_row_*)
#   * row dequant bit-equal to ggml's to_float (env GESSO_GGML_DIR; skipped if unset)
#   * AVX2 kernels bit-exact against the scalar reference, per format
#   * zero allocations per dot; GEMV threading invariance
#   * real tensors from the Ornith 35B and Qwen3.8-27B GGUFs and the Qwen3.8-27B
#     MTP head in Q4_0 (env-gated; reads only the header and the rows it tests,
#     the models are never loaded)

using Random: Xoshiro
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _KQ = Gesso.Inference
const _KQTYPES = (
    ("Q2_K", _KQ.GGML_TYPE_Q2_K),
    ("Q3_K", _KQ.GGML_TYPE_Q3_K),
    ("Q4_K", _KQ.GGML_TYPE_Q4_K),
    ("Q5_K", _KQ.GGML_TYPE_Q5_K),
    ("Q6_K", _KQ.GGML_TYPE_Q6_K),
    ("Q8_0", _KQ.GGML_TYPE_Q8_0),
    ("Q4_0", _KQ.GGML_TYPE_Q4_0),
    ("IQ4_NL", _KQ.GGML_TYPE_IQ4_NL),
)

_kq_nvals(ty::Int32) = _KQ._q80_act_type(ty) ? 32 : _KQ.QK_K

function _kq_put_f16!(w::Vector{UInt8}, off0::Int, v::Float32)
    bits = reinterpret(UInt16, Float16(v))
    w[off0+1] = UInt8(bits & 0xff)
    w[off0+2] = UInt8(bits >> 8)
    return w
end

# Random valid rows: arbitrary bytes everywhere (grid, scale, quant and sign
# fields), finite small f16 super-scales at the format's d and dmin offsets.
function _kq_random_rows(rng, ty::Int32, nrows::Int, nb::Int)
    bsz = _KQ._iq_block_bytes(ty)
    w = rand(rng, UInt8, nrows * nb * bsz)
    doff = _KQ._q8k_d_offset(ty)
    dmo = _KQ._q8k_dmin_offset(ty)
    for blk in 0:(nrows*nb-1)
        _kq_put_f16!(w, blk * bsz + doff, 0.005f0 + 0.05f0 * rand(rng, Float32))
        dmo >= 0 && _kq_put_f16!(w, blk * bsz + dmo, 0.005f0 + 0.05f0 * rand(rng, Float32))
    end
    return w
end

function _kq_dequant_q80(x::_KQ.Q80Vec)
    out = Vector{Float32}(undef, length(x))
    @inbounds for i in eachindex(out)
        out[i] = x.d[(i-1)÷32+1] * Float32(x.qs[i])
    end
    return out
end

_kq_dequant_act(ty::Int32, xq) =
    _KQ._q80_act_type(ty) ? _kq_dequant_q80(xq) : _KQ.dequantize_q8_K(xq)

# Independent port of quantize_row_q8_0_ref. roundf (half away from zero) is
# written out with trunc, so this does not share the implementation under test.
function _kq_ref_quantize_q80(x::AbstractVector{Float32})
    nb = length(x) ÷ 32
    d = zeros(Float32, nb)
    qs = zeros(Int8, length(x))
    for i in 1:nb
        blk = x[((i-1)*32+1):(i*32)]
        amax = maximum(abs, blk)
        dd = amax / 127.0f0
        id = dd != 0 ? 1.0f0 / dd : 0.0f0
        d[i] = Float32(Float16(dd))
        for j in 1:32
            v = blk[j] * id
            t = trunc(v)
            r = abs(v - t) >= 0.5f0 ? t + sign(v) : t
            qs[(i-1)*32+j] = Int8(r)
        end
    end
    return d, qs
end

# Float64 dot of the F32 dequantized row and dequantized activations, plus the
# sum of absolute terms (the scale for a float-rounding tolerance).
function _kq_ref_dot(ty::Int32, W, woff::Int, ne0::Int, xq)
    wdq = Vector{Float32}(undef, ne0)
    _KQ._kq_dequant_row!(wdq, W, woff, ty, ne0)
    xdq = _kq_dequant_act(ty, xq)
    s = 0.0
    a = 0.0
    @inbounds for k in 1:ne0
        t = Float64(wdq[k]) * Float64(xdq[k])
        s += t
        a += abs(t)
    end
    return s, a
end

@testset "Q8_0 quantizer: bit-equal to independent port" begin
    rng = Xoshiro(20)
    for nb in (1, 4)
        x = randn(rng, Float32, 32 * nb) .* 3.0f0
        x[3] = 0.0f0
        y = _KQ.quantize_q8_0!(_KQ.Q80Vec(nb), x)
        d, qs = _kq_ref_quantize_q80(x)
        @test y.qs == qs
        @test reinterpret.(UInt32, y.d) == reinterpret.(UInt32, d)
    end
    # all-zero block: d = 0 and zero quants
    z = _KQ.quantize_q8_0!(_KQ.Q80Vec(2), zeros(Float32, 64))
    @test all(==(0.0f0), z.d)
    @test all(==(0), z.qs)
    # hand example: amax 2 gives d = 2/127 and quant -127 for the max value
    x = zeros(Float32, 32)
    x[1] = -2.0f0
    y = _KQ.quantize_q8_0!(_KQ.Q80Vec(1), x)
    @test y.qs[1] == Int8(-127)
    @test all(==(0), y.qs[2:end])
    @test y.d[1] == Float32(Float16(2.0f0 / 127.0f0))
    # ties away from zero: x*id = 2.5 -> 3 (roundf), ties-to-even would give 2
    xt = zeros(Float32, 32)
    xt[1] = 127.0f0
    xt[2] = 2.5f0
    yt = _KQ.quantize_q8_0!(_KQ.Q80Vec(1), xt)
    @test yt.qs[2] == Int8(3)
    @test yt.qs[1] == Int8(127)
end

@testset "scalar reference vs F32 dequant reference" begin
    rng = Xoshiro(21)
    nb, nrow = 4, 6
    for (name, ty) in _KQTYPES
        nval = _kq_nvals(ty)
        ne0 = nval * nb
        W = _kq_random_rows(rng, ty, nrow, nb)
        x = randn(rng, Float32, ne0)
        xq = _KQ._q8k_quantize_act(ty, x)
        rb = _KQ._qrow_bytes(ty, ne0)
        for j in 1:nrow
            ref, mag = _kq_ref_dot(ty, W, (j - 1) * rb, ne0, xq)
            got = _KQ._dot_q8k(ty, W, (j - 1) * rb, xq, false)
            @test abs(Float64(got) - ref) <= 1.0e-5 * mag + 1.0e-6
        end
    end
end

@testset "F32 dequant reference bit-equal to ggml to_float" begin
    ggml_dir = get(ENV, "GESSO_GGML_DIR", "")
    if !isfile(joinpath(ggml_dir, "libggml-base.so"))
        _skip(
            "set GESSO_GGML_DIR to a directory with libggml-base.so — dequant vs ggml to_float skipped",
        )
    else
        rng = Xoshiro(22)
        for (name, ty) in _KQTYPES
            nb = 3
            ne0 = _kq_nvals(ty) * nb
            W = _kq_random_rows(rng, ty, 2, nb)
            rb = _KQ._qrow_bytes(ty, ne0)
            for j in 1:2
                mine = Vector{Float32}(undef, ne0)
                _KQ._kq_dequant_row!(mine, W, (j - 1) * rb, ty, ne0)
                row = W[((j-1)*rb+1):(j*rb)]
                ggml = _KQ._ggml_lib_dequant(row, 0, ty, ne0)
                @testset "$name row $j" begin
                    @test reinterpret.(UInt32, mine) == reinterpret.(UInt32, ggml)
                end
            end
        end
    end
end

@testset "AVX2 kernels bit-exact against scalar reference" begin
    if !_KQ._cpu_has_avx2()
        _skip("no AVX2 on this CPU: K-quant AVX2-vs-scalar bit-exact check skipped")
    else
        for (name, ty) in _KQTYPES
            rng = Xoshiro(30 + Int(ty))
            ntrial = 0
            nmismatch = 0
            for trial in 1:300
                nb = 1 + (trial % 4)
                W = _kq_random_rows(rng, ty, 4, nb)
                ne0 = _kq_nvals(ty) * nb
                x = randn(rng, Float32, ne0) .* Float32(0.1 + 10 * rand(rng))
                if trial % 7 == 0       # extreme activations: quants saturate at ±127
                    x .= x .* 1000.0f0
                end
                if trial % 11 == 0      # zeros in part of the activation block
                    x[1:min(64, ne0)] .= 0.0f0
                end
                xq = _KQ._q8k_quantize_act(ty, x)
                rb = _KQ._qrow_bytes(ty, ne0)
                for j in 1:4
                    r = _KQ._dot_q8k(ty, W, (j - 1) * rb, xq, false)
                    a = _KQ._dot_q8k(ty, W, (j - 1) * rb, xq, true)
                    ntrial += 1
                    nmismatch += reinterpret(UInt32, r) != reinterpret(UInt32, a)
                end
            end
            @testset "$name" begin
                @test nmismatch == 0
                @test ntrial == 1200
            end
        end
        # saturating fields: every packed byte 0xff (max nibble, max 2-bit, sign
        # bits set, Q6 scales -1) with d = 0 contributes zero, and with a large
        # d and dmin the sums stay exact.
        @testset "saturating fields" begin
            for (name, ty) in _KQTYPES
                nb = 2
                bsz = _KQ._iq_block_bytes(ty)
                nval = _kq_nvals(ty)
                W = fill(UInt8(0xff), nb * bsz)
                for ib in 0:(nb-1)
                    _kq_put_f16!(W, ib * bsz + _KQ._q8k_d_offset(ty), 0.0f0)
                    dmo = _KQ._q8k_dmin_offset(ty)
                    dmo >= 0 && _kq_put_f16!(W, ib * bsz + dmo, 0.0f0)
                end
                x = fill(1.0f0, nval * nb)
                xq = _KQ._q8k_quantize_act(ty, x)
                r = _KQ._dot_q8k(ty, W, 0, xq, false)
                a = _KQ._dot_q8k(ty, W, 0, xq, true)
                @test r === 0.0f0
                @test reinterpret(UInt32, r) == reinterpret(UInt32, a)
                # large scales, all-saturated fields and extreme activations
                for ib in 0:(nb-1)
                    _kq_put_f16!(W, ib * bsz + _KQ._q8k_d_offset(ty), 60000.0f0)
                    dmo = _KQ._q8k_dmin_offset(ty)
                    dmo >= 0 && _kq_put_f16!(W, ib * bsz + dmo, -60000.0f0)
                end
                x = randn(Xoshiro(40 + Int(ty)), Float32, nval * nb) .* 1000.0f0
                xq = _KQ._q8k_quantize_act(ty, x)
                r = _KQ._dot_q8k(ty, W, 0, xq, false)
                a = _KQ._dot_q8k(ty, W, 0, xq, true)
                @test reinterpret(UInt32, r) == reinterpret(UInt32, a)
            end
        end
    end
end

@testset "K-quant dot kernels do not allocate" begin
    rng = Xoshiro(50)
    for (name, ty) in _KQTYPES
        nb = 4
        W = _kq_random_rows(rng, ty, 1, nb)
        x = randn(rng, Float32, _kq_nvals(ty) * nb)
        xq = _KQ._q8k_quantize_act(ty, x)
        for simd in (false, true)
            if simd && !_KQ._cpu_has_avx2()
                continue
            end
            _KQ._dot_q8k(ty, W, 0, xq, simd)
            @test (@allocated _KQ._dot_q8k(ty, W, 0, xq, simd)) == 0
        end
    end
end

@testset "K-quant GEMV: threading does not change any output bit" begin
    rng = Xoshiro(60)
    nb, nrow = 3, 37
    for (name, ty) in _KQTYPES
        ne0 = _kq_nvals(ty) * nb
        W = _kq_random_rows(rng, ty, nrow, nb)
        x = randn(rng, Float32, ne0)
        y1 = zeros(Float32, nrow)
        _KQ._qgemv_q8k_f32!(y1, x, W, ty, ne0, nrow; threaded=false)
        for nc in (1, 3, 7, nrow)
            yt = zeros(Float32, nrow)
            xq = _KQ._q8k_quantize_act(ty, x)
            _KQ._qgemv_q8k_threaded!(yt, xq, W, ty, ne0, nrow; nchunks=nc)
            @test reinterpret.(UInt32, yt) == reinterpret.(UInt32, y1)
        end
        # the dequant reference agrees within activation-quantization error
        yref = zeros(Float64, nrow)
        rb = _KQ._qrow_bytes(ty, ne0)
        for j in 1:nrow
            yref[j], _ = _kq_ref_dot(ty, W, (j - 1) * rb, ne0, _KQ._q8k_quantize_act(ty, x))
        end
        @test maximum(abs.(Float64.(y1) .- yref)) <= 2.0e-2 * maximum(abs, yref)
    end
end

@testset "K-quant GEMV: shape checks fail explicitly" begin
    xq = _KQ._q8k_quantize_act(_KQ.GGML_TYPE_Q5_K, zeros(Float32, 256))
    W = zeros(UInt8, 176 * 2)
    @test_throws ErrorException _KQ._qgemv_q8k!(
        zeros(Float32, 2),
        xq,
        W,
        _KQ.GGML_TYPE_Q5_K,
        100,
        2,
    )
    @test_throws ErrorException _KQ._qgemv_q8k_f32!(
        zeros(Float32, 2),
        zeros(Float32, 100),
        W,
        _KQ.GGML_TYPE_Q5_K,
        100,
        2,
    )
    @test_throws ErrorException _KQ._qgemv_q8k_f32!(
        zeros(Float32, 2),
        zeros(Float32, 40),
        zeros(UInt8, 34 * 2),
        _KQ.GGML_TYPE_Q8_0,
        40,
        2,
    )
    @test_throws ErrorException _KQ._qgemv_q8k_f32!(
        zeros(Float32, 2),
        zeros(Float32, 256),
        W,
        _KQ.GGML_TYPE_F32,
        256,
        2,
    )
    # Q8_0 with ne0 = 64 is valid (two blocks per row)
    y = zeros(Float32, 2)
    @test _KQ._qgemv_q8k_f32!(
        y,
        zeros(Float32, 64),
        zeros(UInt8, 34 * 2 * 2),
        _KQ.GGML_TYPE_Q8_0,
        64,
        2;
        threaded=false,
    ) === y
end

@testset "K-quant dots reject an activation shorter than the row" begin
    # d holds 2 blocks but qs holds only 1: the loop would index qs past its end
    short_k = _KQ.Q8KVec(zeros(Float32, 2), zeros(Int8, 256), zeros(Int16, 16))
    short_q80 = _KQ.Q80Vec(zeros(Float32, 2), zeros(Int8, 32))
    for (ty, bpb) in (
        (_KQ.GGML_TYPE_Q2_K, 84),
        (_KQ.GGML_TYPE_Q3_K, 110),
        (_KQ.GGML_TYPE_Q4_K, 144),
        (_KQ.GGML_TYPE_Q5_K, 176),
        (_KQ.GGML_TYPE_Q6_K, 210),
    )
        W = zeros(UInt8, bpb * 2)
        @test_throws DimensionMismatch _KQ._dot_q8k(ty, W, 0, short_k, false)
        @test_throws DimensionMismatch _KQ._dot_q8k(ty, W, 0, short_k, true)
    end
    W = zeros(UInt8, 34 * 2)
    @test_throws DimensionMismatch _KQ._dot_q8_0_q8_0(W, 0, short_q80, false)
    @test_throws DimensionMismatch _KQ._dot_q8_0_q8_0(W, 0, short_q80, true)
    # a row that runs past the end of the weights still fails as a BoundsError
    @test_throws BoundsError _KQ._dot_q4k_q8k(zeros(UInt8, 144), 1, _KQ.Q8KVec(2), false)
end

# Real tensors. Header and row reads only: the rows tested are mapped from the
# file, and the rest of the model is never touched.
function _kq_real_tensors(path::AbstractString)
    hdr = Gesso.gguf_header(path)
    found = Dict{Int32, Any}()
    for info in hdr.tensors
        length(info.dims) == 2 || continue
        occursin("_exps", info.name) && continue
        haskey(found, info.ggml_type) || (found[info.ggml_type] = info)
    end
    return hdr, found
end

# OPTIONAL oracle: ggml's exported generic kernels, called through
# libggml-cpu.so when GESSO_GGML_DIR is set. Test-only. No runtime path calls
# ggml compute. The ground truth is the scalar port of the generic code, and
# the AVX2 kernels must match it bit for bit (checked above).
#
# The compiled ggml build contracts a*b+c into fused multiply-add in some
# float accumulations (vfmadd231ss in the generic Q8_0 kernel), while the port
# follows the C source operation by operation. The two therefore differ, but
# only by rounding. The error is measured against the sum of absolute terms,
# |r - g| / (eps32 * sum |w_k x_k|), since a ulp count relative to the result
# blows up when the result is near zero by cancellation. Measured over 60
# trials per format (rows of 1 to 3 blocks, some with activations scaled by
# 1000): max 0.33 eps32 (Q8_0), 0.19 (Q3_K), 0.13 (Q2_K), and below 0.11 for
# Q4_K, Q5_K and Q6_K. The bound is 2 eps32, about six times the worst case.
const _KQ_GGML_ORACLE_MAX_EPS = 2.0

function _kq_ggml_generic_sym(ty::Int32)
    names = Dict(
        _KQ.GGML_TYPE_Q2_K => :ggml_vec_dot_q2_K_q8_K_generic,
        _KQ.GGML_TYPE_Q3_K => :ggml_vec_dot_q3_K_q8_K_generic,
        _KQ.GGML_TYPE_Q4_K => :ggml_vec_dot_q4_K_q8_K_generic,
        _KQ.GGML_TYPE_Q5_K => :ggml_vec_dot_q5_K_q8_K_generic,
        _KQ.GGML_TYPE_Q6_K => :ggml_vec_dot_q6_K_q8_K_generic,
        _KQ.GGML_TYPE_Q8_0 => :ggml_vec_dot_q8_0_q8_0_generic,
        _KQ.GGML_TYPE_Q4_0 => :ggml_vec_dot_q4_0_q8_0_generic,
        _KQ.GGML_TYPE_IQ4_NL => :ggml_vec_dot_iq4_nl_q8_0_generic,
    )
    _KQ._ensure_ggml!()
    hcpu = _KQ._GGML_LIBS[][2]
    hcpu == C_NULL && return C_NULL
    return _KQ.Libdl.dlsym(hcpu, names[ty]; throw_error=false)
end

function _kq_ggml_act_bytes(xq)
    if xq isa _KQ.Q80Vec
        nb = length(xq.d)
        out = Vector{UInt8}(undef, 34 * nb)
        for i in 1:nb
            o = 34 * (i - 1)
            bits = reinterpret(UInt16, Float16(xq.d[i]))
            out[o+1] = UInt8(bits & 0xff)
            out[o+2] = UInt8(bits >> 8)
            out[(o+3):(o+34)] = reinterpret(UInt8, xq.qs[(32*(i-1)+1):(32*i)])
        end
        return out
    end
    nb = length(xq.d)
    out = Vector{UInt8}(undef, 292 * nb)
    for i in 1:nb
        o = 292 * (i - 1)
        out[(o+1):(o+4)] = reinterpret(UInt8, [xq.d[i]])
        out[(o+5):(o+260)] = reinterpret(UInt8, xq.qs[(256*(i-1)+1):(256*i)])
        out[(o+261):(o+292)] = reinterpret(UInt8, xq.bsums[(16*(i-1)+1):(16*i)])
    end
    return out
end

function _kq_ggml_generic_dot(fp::Ptr{Cvoid}, ne0::Int, w::Vector{UInt8}, a::Vector{UInt8})
    s = Ref{Float32}(0.0f0)
    ccall(
        fp,
        Cvoid,
        (Cint, Ptr{Float32}, Csize_t, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Cint),
        Cint(ne0),
        s,
        Csize_t(0),
        pointer(w),
        Csize_t(0),
        pointer(a),
        Csize_t(0),
        Cint(1),
    )
    return s[]
end

@testset "optional oracle: scalar vs compiled ggml generic kernels (ulp bound)" begin
    ggml_dir = get(ENV, "GESSO_GGML_DIR", "")
    if !isfile(joinpath(ggml_dir, "libggml-cpu.so"))
        _skip(
            "GESSO_GGML_DIR unset or has no libggml-cpu.so — optional ggml oracle skipped",
        )
    else
        for (name, ty) in _KQTYPES
            fp = _kq_ggml_generic_sym(ty)
            if fp == C_NULL
                _skip("$name: ggml_vec_dot_*_generic not exported by libggml-cpu.so")
                continue
            end
            rng = Xoshiro(80 + Int(ty))
            maxerr = 0.0
            ntrial = 0
            for trial in 1:60
                nb = 1 + (trial % 3)
                ne0 = _kq_nvals(ty) * nb
                W = _kq_random_rows(rng, ty, 1, nb)
                x = randn(rng, Float32, ne0) .* Float32(0.1 + 10 * rand(rng))
                trial % 7 == 0 && (x .= x .* 1000.0f0)
                xq = _KQ._q8k_quantize_act(ty, x)
                g = _kq_ggml_generic_dot(fp, ne0, W, _kq_ggml_act_bytes(xq))
                r = _KQ._dot_q8k(ty, W, 0, xq, false)
                a = _KQ._dot_q8k(ty, W, 0, xq, true)
                ntrial += 1
                _, mag = _kq_ref_dot(ty, W, 0, ne0, xq)
                maxerr = max(
                    maxerr,
                    abs(Float64(r) - Float64(g)) / (mag * Float64(eps(Float32))),
                )
                @test reinterpret(UInt32, a) == reinterpret(UInt32, r)  # hard gate
            end
            println(
                "ggml oracle ",
                name,
                " max_err_over_sum_abs_terms_eps32=",
                round(maxerr; digits=3),
                " bound=",
                _KQ_GGML_ORACLE_MAX_EPS,
                " trials=",
                ntrial,
            )
            @testset "$name" begin
                @test maxerr <= _KQ_GGML_ORACLE_MAX_EPS
                @test ntrial == 60
            end
        end
    end
end

# QuantMatrix dispatch: K-quant and Q8_0 weights run natively (no to_float
# fallback), and the matmul matches a dequant reference within the activation
# quantization error.
@testset "QuantMatrix matmul runs K-quant and Q8_0 natively" begin
    rng = Xoshiro(90)
    for (name, ty) in _KQTYPES
        nval = _kq_nvals(ty)
        ne0 = 2 * nval
        nrows, seq = 9, 3
        rb = _KQ._qrow_bytes(ty, ne0)
        W = _kq_random_rows(rng, ty, nrows, 2)
        q = _KQ.QuantMatrix(W, 0, ty, ne0, nrows, rb, Int64(nval))
        x = randn(rng, Float32, seq, ne0)
        dst = zeros(Float32, seq, nrows)
        _KQ._quant_matmul!(dst, x, q)
        yref = zeros(Float64, seq, nrows)
        for r in 1:nrows
            wdq = Vector{Float32}(undef, ne0)
            _KQ._kq_dequant_row!(wdq, W, (r - 1) * rb, ty, ne0)
            for t in 1:seq
                yref[t, r] = sum(Float64.(wdq) .* Float64.(x[t, :]))
            end
        end
        @testset "$name" begin
            @test !(ty in _KQ._QUANT_FALLBACK_TYPES)
            @test maximum(abs, Float64.(dst) .- yref) <= 2.0e-2 * maximum(abs, yref)
        end
    end
end

@testset "real tensors: K-quant path vs F32 reference" begin
    named = (
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_out.weight",
        "output.weight",
        "blk.0.ffn_down_shexp.weight",
    )
    ggml_ok_dir = isfile(joinpath(get(ENV, "GESSO_GGML_DIR", ""), "libggml-base.so"))
    ran = false
    for path in (
        get(ENV, "GESSO_ORNITH35_GGUF", ""),
        get(ENV, "GESSO_QWEN38_27B_IQ4XS_GGUF", ""),
        get(ENV, "GESSO_QWEN38_MTP_GGUF", ""),
    )
        if !isfile(path)
            _skip(
                "set GESSO_ORNITH35_GGUF / GESSO_QWEN38_27B_IQ4XS_GGUF / GESSO_QWEN38_MTP_GGUF to a GGUF — real tensors skipped",
            )
            continue
        end
        ran = true
        hdr, found = _kq_real_tensors(path)
        targets = Any[]
        for nm in named
            info = findfirst(t -> t.name == nm, hdr.tensors)
            info === nothing || push!(targets, hdr.tensors[info])
        end
        for ty in (
            _KQ.GGML_TYPE_Q5_K,
            _KQ.GGML_TYPE_Q6_K,
            _KQ.GGML_TYPE_Q4_K,
            _KQ.GGML_TYPE_Q3_K,
            _KQ.GGML_TYPE_Q2_K,
            _KQ.GGML_TYPE_Q8_0,
            _KQ.GGML_TYPE_Q4_0,
            _KQ.GGML_TYPE_IQ4_NL,
        )
            haskey(found, ty) && push!(targets, found[ty])
        end
        rng = Xoshiro(70)
        seen = Set{String}()
        for info in targets
            info.name in seen && continue
            _KQ._kq_supported(info.ggml_type) || continue
            push!(seen, info.name)
            ty = info.ggml_type
            ne0, ne1 = info.dims
            rb = _KQ._qrow_bytes(ty, ne0)
            base = Int(hdr.data_start) + Int(info.offset)
            rows = unique(round.(Int, range(1, ne1; length=min(16, ne1))))
            x = randn(rng, Float32, ne0)
            xq = _KQ._q8k_quantize_act(ty, x)
            err_exact2 = 0.0   # kernel vs F32 dequant of the unquantized activations
            sig2 = 0.0
            err_deq2 = 0.0     # kernel vs F32 dequant of the quantized activations (arithmetic only)
            nmis = 0
            ggml_ok = true
            for j in rows
                off = base + (j - 1) * rb
                wdq = Vector{Float32}(undef, ne0)
                _KQ._kq_dequant_row!(wdq, hdr.data, off, ty, ne0)
                yexact = sum(Float64.(wdq) .* Float64.(x))
                ydeq, _ = _kq_ref_dot(ty, hdr.data, off, ne0, xq)
                ys = _KQ._dot_q8k(ty, hdr.data, off, xq, false)
                ya = _KQ._dot_q8k(ty, hdr.data, off, xq, true)
                nmis += reinterpret(UInt32, ys) != reinterpret(UInt32, ya)
                err_exact2 += (Float64(ys) - yexact)^2
                sig2 += yexact^2
                err_deq2 += (Float64(ys) - ydeq)^2
            end
            if ggml_ok_dir
                for j in rows[1:min(4, length(rows))]
                    off = base + (j - 1) * rb
                    mine = Vector{Float32}(undef, ne0)
                    _KQ._kq_dequant_row!(mine, hdr.data, off, ty, ne0)
                    gg = _KQ._ggml_lib_dequant(
                        Vector{UInt8}(hdr.data[(off+1):(off+rb)]),
                        0,
                        ty,
                        ne0,
                    )
                    ggml_ok &= reinterpret.(UInt32, mine) == reinterpret.(UInt32, gg)
                end
            end
            relrms = sqrt(err_exact2 / sig2)
            println(
                "real ",
                basename(path),
                " ",
                info.name,
                " type=",
                ty,
                " ne0=",
                ne0,
                " ne1=",
                ne1,
                " rows=",
                length(rows),
                " rel_rms_err_vs_f32=",
                relrms,
                " rel_rms_dot_vs_dequant=",
                sqrt(err_deq2 / sig2),
                " avx2_eq_scalar=",
                nmis == 0,
                ggml_ok_dir ? " dequant_eq_ggml=$ggml_ok" : "",
            )
            @test nmis == 0
            @test relrms <= 2.0e-2
            ggml_ok_dir && @test ggml_ok
        end
    end
    ran || nothing
end
