# Flash-Next quant types: IQ4_NL (ggml 20), Q2_0 (ggml 42), and the CUDA
# GEMV additions IQ2_XXS (16), IQ2_S (22) and IQ3_S (21).
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
const _FN_TYPES = (
    ("IQ4_NL", _FN.GGML_TYPE_IQ4_NL, 32),
    ("Q2_0", _FN.GGML_TYPE_Q2_0, 64),
    ("IQ2_XXS", _FN.GGML_TYPE_IQ2_XXS, 256),
    ("IQ2_S", _FN.GGML_TYPE_IQ2_S, 256),
    ("IQ3_S", _FN.GGML_TYPE_IQ3_S, 256),
)
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

# The IQ2_XXS/IQ2_S/IQ3_S CPU kernels run on Q8_K activations, so a unit-vector
# product is not an exact dequant for them; their F32 reference row dequant is.
const _FN_IQS = (_FN.GGML_TYPE_IQ2_XXS, _FN.GGML_TYPE_IQ2_S, _FN.GGML_TYPE_IQ3_S)

function _fn_dequant(data::Vector{UInt8}, ty::Int32, nrows::Int, ne0::Int)
    ty in _FN_IQS || return _fn_gesso_dequant(data, ty, nrows, ne0)
    rb = _FN._qrow_bytes(ty, ne0)
    W = zeros(Float32, nrows, ne0)
    row = Vector{Float32}(undef, ne0)
    for j in 1:nrows
        _FN._iqs_dequant_row!(row, data, (j - 1) * rb, ty, ne0)
        W[j, :] = row
    end
    return W
end

function _fn_matrix(data::Vector{UInt8}, ty::Int32, nrows::Int, ne0::Int, blck::Int)
    rb = _FN._qrow_bytes(ty, ne0)
    return _FN.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(blck))
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

# Activations rounded through Q8_K (256-value blocks), as the CPU IQ2/IQ3 kernels see them.
function _fn_q8k_round(x::Vector{Float32})
    xq = _FN.quantize_q8_K!(_FN.Q8KVec(div(length(x), 256)), x)
    return Float32.(_FN.dequantize_q8_K(xq))
end

const _FN_ggml_ok = isfile(joinpath(get(ENV, "GESSO_GGML_DIR", ""), "libggml-base.so"))

@testset "Flash-Next types: block sizes match ggml traits" begin
    @test _FN._iq_block_bytes(_FN.GGML_TYPE_IQ4_NL) == 18
    @test _FN._iq_block_bytes(_FN.GGML_TYPE_Q2_0) == 18
    @test _FN._qrow_bytes(_FN.GGML_TYPE_IQ4_NL, 256) == 8 * 18
    @test _FN._qrow_bytes(_FN.GGML_TYPE_Q2_0, 256) == 4 * 18
end

@testset "Flash-Next types: dequant bit-equal to ggml to_float" begin
    if !_FN_ggml_ok
        _skip(
            "set GESSO_GGML_DIR to a directory with libggml-base.so — IQ4_NL/Q2_0 dequant vs ggml to_float skipped",
        )
    else
        rng = MersenneTwister(20261026)
        for (name, ty, blck) in _FN_TYPES
            nrows, ne0 = 5, 256
            data = _fn_random_rows(rng, ty, nrows, div(ne0, blck))
            ggml = permutedims(reshape(_FN._ggml_lib_dequant(data, 0, ty, nrows * ne0), ne0, nrows))
            mine = _fn_dequant(data, ty, nrows, ne0)
            # The accumulator never yields -0.0, so zeros are compared as +0.0.
            # Every nonzero value is compared bit for bit.
            @testset "$name random rows" begin
                @test reinterpret.(UInt32, mine .+ 0.0f0) ==
                      reinterpret.(UInt32, ggml .+ 0.0f0)
            end
        end
    end
end

@testset "Flash-Next types: CPU GEMV vs F32 reference" begin
    rng = MersenneTwister(7)
    for (name, ty, blck) in _FN_TYPES
        nrows, ne0 = 37, 512
        data = _fn_random_rows(rng, ty, nrows, div(ne0, blck))
        W = _fn_dequant(data, ty, nrows, ne0)
        x = randn(rng, Float32, ne0)
        y = zeros(Float32, nrows)
        _FN._qgemv_bytes!(y, x, data, ty, ne0, nrows, 1.0f0, 0.0f0)
        # The IQ2/IQ3 CPU kernels quantize activations to Q8_K, so their
        # reference uses the same rounded activations; the F32 gap is reported.
        xr = ty in _FN_IQS ? _fn_q8k_round(x) : x
        ref = Float64.(W) * Float64.(xr)
        rel = _fn_relerr(y, ref)
        rel_f32 = _fn_relerr(y, Float64.(W) * Float64.(x))
        @info "CPU GEMV" type = name max_rel_err = rel max_rel_err_vs_f32 = rel_f32
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
    @testset "Flash-Next types: CUDA expert GEMV vs F32 reference" begin
        rng = MersenneTwister(8)
        for (name, ty, blck) in _FN_TYPES
            nrows, ne0 = 37, 512
            data = _fn_random_rows(rng, ty, nrows, div(ne0, blck))
            W = _fn_dequant(data, ty, nrows, ne0)
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
