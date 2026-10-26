# Flash-Next quant types: IQ4_NL (ggml 20) and Q2_0 (ggml 42).
#   * random valid blocks (f16 scale at byte 0, random code bytes)
#   * dequant bit-equal to ggml's to_float, row by row (env GESSO_GGML_DIR;
#     skipped if libggml-base.so is absent)
#   * CPU GEMV against an F32 reference matmul of the dequantized weights
#   * CUDA expert GEMV against the same reference (skipped without a GPU)
# Tolerance for the GEMV checks is relative error <= 2e-3 against max |ref|.

using Random: MersenneTwister
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _FN = Gesso.Inference
const _FN_TYPES = (("IQ4_NL", _FN.GGML_TYPE_IQ4_NL, 32), ("Q2_0", _FN.GGML_TYPE_Q2_0, 64))
const _FN_TOL = 2.0e-3

# Random valid rows: d is a small finite f16 at byte 0 of every block, every
# other byte uniform random.
function _fn_random_rows(rng, ty::Int32, nrows::Int, nb::Int)
    bsz = _FN._iq_block_bytes(ty)
    w = rand(rng, UInt8, nrows * nb * bsz)
    for blk in 0:(nrows*nb-1)
        dv = Float16((rand(rng, Float32) * 0.04f0 + 0.002f0) * (rand(rng) < 0.5 ? -1 : 1))
        w[(blk*bsz+1):(blk*bsz+2)] .= reinterpret(UInt8, [dv])
    end
    return w
end

# Gesso's own dequant: the CPU fused kernel applied to unit vectors. Each
# product is exact, so column k of the result is element k of every row.
function _fn_gesso_dequant(data::Vector{UInt8}, ty::Int32, nrows::Int, ne0::Int)
    W = zeros(Float32, nrows, ne0)
    x = zeros(Float32, ne0)
    y = zeros(Float32, nrows)
    for k in 1:ne0
        fill!(x, 0.0f0)
        x[k] = 1.0f0
        _FN._qgemv_bytes!(y, x, data, ty, ne0, nrows, 1.0f0, 0.0f0)
        W[:, k] = y
    end
    return W
end

function _fn_matrix(data::Vector{UInt8}, ty::Int32, nrows::Int, ne0::Int, blck::Int)
    rb = _FN._qrow_bytes(ty, ne0)
    return _FN.QuantMatrix(data, 0, ty, ne0, nrows, rb, C_NULL, Int64(blck))
end

_fn_relerr(y, ref) = maximum(abs, Float64.(y) .- ref) / maximum(abs, ref)

# Activations rounded to int8 with one scale per 32 values, as the q8_1 path does.
function _fn_q8_round(x::Vector{Float32})
    y = similar(x)
    for b in 0:32:(length(x)-1)
        blk = view(x, (b+1):(b+32))
        s = maximum(abs, blk) / 127
        for i in 1:32
            y[b+i] = s == 0 ? 0.0f0 : round(blk[i] / s) * s
        end
    end
    return y
end

const _FN_ggml_ok = isfile(joinpath(get(ENV, "GESSO_GGML_DIR", ""), "libggml-base.so"))

@testset "IQ4_NL and Q2_0: block sizes match ggml traits" begin
    @test _FN._iq_block_bytes(_FN.GGML_TYPE_IQ4_NL) == 18
    @test _FN._iq_block_bytes(_FN.GGML_TYPE_Q2_0) == 18
    @test _FN._qrow_bytes(_FN.GGML_TYPE_IQ4_NL, 256) == 8 * 18
    @test _FN._qrow_bytes(_FN.GGML_TYPE_Q2_0, 256) == 4 * 18
end

@testset "IQ4_NL and Q2_0: dequant bit-equal to ggml to_float" begin
    if !_FN_ggml_ok
        _skip(
            "set GESSO_GGML_DIR to a directory with libggml-base.so — IQ4_NL/Q2_0 dequant vs ggml to_float skipped",
        )
    else
        rng = MersenneTwister(20261026)
        for (name, ty, blck) in _FN_TYPES
            nrows, ne0 = 5, 256
            data = _fn_random_rows(rng, ty, nrows, div(ne0, blck))
            qm = _fn_matrix(data, ty, nrows, ne0, blck)
            ggml = Float32.(_FN._quant_host_f32(qm))
            mine = _fn_gesso_dequant(data, ty, nrows, ne0)
            # The accumulator never yields -0.0, so zeros are compared as +0.0.
            # Every nonzero value is compared bit for bit.
            @testset "$name random rows" begin
                @test reinterpret.(UInt32, mine .+ 0.0f0) ==
                      reinterpret.(UInt32, ggml .+ 0.0f0)
            end
        end
    end
end

@testset "IQ4_NL and Q2_0: CPU GEMV vs F32 reference" begin
    rng = MersenneTwister(7)
    for (name, ty, blck) in _FN_TYPES
        nrows, ne0 = 37, 512
        data = _fn_random_rows(rng, ty, nrows, div(ne0, blck))
        W = _fn_gesso_dequant(data, ty, nrows, ne0)
        x = randn(rng, Float32, ne0)
        ref = Float64.(W) * Float64.(x)
        y = zeros(Float32, nrows)
        _FN._qgemv_bytes!(y, x, data, ty, ne0, nrows, 1.0f0, 0.0f0)
        rel = _fn_relerr(y, ref)
        @info "CPU GEMV" type = name max_rel_err = rel
        @testset "$name" begin
            @test rel <= _FN_TOL
        end
    end
end

if !isdefined(@__MODULE__, :_FN_CUDA_OK)
    const _FN_CUDA_OK = let
        ok = true
        try
            @eval Main using CUDA
        catch
            ok = false
        end
        ok
    end
end

if !_FN_CUDA_OK || !Main.CUDA.functional()
    _skip("no NVIDIA device — CUDA IQ4_NL/Q2_0 GEMV skipped")
else
    _fn_ext = Base.get_extension(Gesso, :GessoCUDAExt)
    @testset "IQ4_NL and Q2_0: CUDA expert GEMV vs F32 reference" begin
        rng = MersenneTwister(8)
        for (name, ty, blck) in _FN_TYPES
            nrows, ne0 = 37, 512
            data = _fn_random_rows(rng, ty, nrows, div(ne0, blck))
            W = _fn_gesso_dequant(data, ty, nrows, ne0)
            x = randn(rng, Float32, ne0)
            # The CUDA expert kernel quantizes activations to int8 per 32 values,
            # so the reference uses the same rounded activations.
            ref = Float64.(W) * Float64.(_fn_q8_round(x))
            ref_f32 = Float64.(W) * Float64.(x)
            q = Main.CUDA.CuArray{UInt8}(data)
            xd = Main.CUDA.CuArray{Float32}(x)
            yd = Main.CUDA.zeros(Float32, nrows)
            _fn_ext._cuda_launch_qgemv!(yd, xd, q, ty, 0, ne0, nrows, 1.0f0, 0.0f0)
            rel = _fn_relerr(Array(yd), ref)
            @info "CUDA GEMV" type = name max_rel_err = rel max_rel_err_vs_f32 =
                _fn_relerr(Array(yd), ref_f32)
            @testset "$name" begin
                @test rel <= _FN_TOL
            end
        end
    end
end
