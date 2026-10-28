# Native GGUF dequant (gguf_dequant.jl). Loading a GGUF needs no libggml, so:
#   * every type in the block table decodes with no library loaded
#   * Q4_1 / Q5_0 / Q5_1 / IQ4_NL blocks with hand-computed values
#   * the decoders unfused from the dot kernels agree with those kernels
#   * every type is bit-equal to ggml's to_float (env GESSO_GGML_DIR; skipped
#     if libggml-base.so is absent)

using Test
using Random
using Gesso

const _DQ = Gesso.Inference

_dq_skip(msg) = @test_skip msg

function _dq_random_blocks(rng, ty::Int32, nb::Int)
    blck, tsize = _DQ._GGML_KNOWN_BLOCKS[ty]
    data = rand(rng, UInt8, nb * tsize)
    return data, nb * blck
end

# Bitwise, except that any NaN matches any NaN: random f16 scales can be NaN
# and ggml does not promise a payload.
_dq_same(a::Float32, b::Float32) = a === b || (isnan(a) && isnan(b))

_dq_f16(x) = reinterpret(UInt8, [Float16(x)])

@testset "native dequant: every known type, no library" begin
    @test _DQ._GGML_LIBS[][1] == C_NULL || get(ENV, "GESSO_GGML_DIR", "") != ""
    rng = Xoshiro(3)
    for ty in sort!(collect(keys(_DQ._GGML_KNOWN_BLOCKS)))
        data, n = _dq_random_blocks(rng, ty, 3)
        @testset "type $ty" begin
            y = _DQ._dequant_f32(data, 0, ty, n)
            @test length(y) == n
            # Offsets are honoured: the same blocks behind a 5-byte prefix.
            @test all(_dq_same.(_DQ._dequant_f32(vcat(zeros(UInt8, 5), data), 5, ty, n), y))
        end
    end
    @test_throws Gesso.GessoError _DQ._dequant_f32(zeros(UInt8, 64), 0, Int32(99), 32)
    @test_throws Gesso.GessoError _DQ._dequant_f32(zeros(UInt8, 17), 0, _DQ.GGML_TYPE_Q8_0, 32)
end

@testset "native dequant: hand-computed blocks" begin
    f = Float32[1.5, -0.0, Inf, 3.0f-41]
    @test _DQ._dequant_f32(collect(reinterpret(UInt8, f)), 0, _DQ.GGML_TYPE_F32, 4) == f
    lo = UInt8[j | (15 - j) << 4 for j in 0:15]
    # Q4_1: x * d + m with d = 1, m = 0.5.
    y = _DQ._dequant_f32(vcat(_dq_f16(1.0), _dq_f16(0.5), lo), 0, _DQ.GGML_TYPE_Q4_1, 32)
    @test y == Float32[(0:15) .+ 0.5; (15:-1:0) .+ 0.5]
    # Q5_0: bit 4 of value j comes from bit j of qh; values are centred on 16.
    qh = reinterpret(UInt8, [UInt32(0x0000ffff)])
    y = _DQ._dequant_f32(vcat(_dq_f16(2.0), qh, lo), 0, _DQ.GGML_TYPE_Q5_0, 32)
    @test y == 2.0f0 .* Float32[(0:15); (15:-1:0) .- 16]
    # Q5_1: the same bits, unsigned, plus the minimum.
    qh = reinterpret(UInt8, [UInt32(0xffff0000)])
    y = _DQ._dequant_f32(vcat(_dq_f16(1.0), _dq_f16(-3.0), qh, lo), 0, _DQ.GGML_TYPE_Q5_1, 32)
    @test y == Float32[(0:15) .- 3; (31:-1:16) .- 3]
    # IQ4_NL: d times the non-linear code table.
    y = _DQ._dequant_f32(vcat(_dq_f16(0.5), lo), 0, _DQ.GGML_TYPE_IQ4_NL, 32)
    kv = Float32.(_DQ.KVALUES_IQ4NL)
    @test y == 0.5f0 .* [kv[1:16]; kv[16:-1:1]]
end

# IQ2_XS, IQ3_XXS, IQ4_XS, IQ4_NL and Q2_0 decoders are the dot kernels with
# the dot taken out, so a row decoded then dotted must match the fused GEMV.
@testset "native dequant agrees with the fused GEMV" begin
    rng = Xoshiro(11)
    for ty in (_DQ.GGML_TYPE_IQ2_XS, _DQ.GGML_TYPE_IQ3_XXS, _DQ.GGML_TYPE_IQ4_XS,
        _DQ.GGML_TYPE_IQ4_NL, _DQ.GGML_TYPE_Q2_0)
        blck, tsize = _DQ._GGML_KNOWN_BLOCKS[ty]
        ne0, ne1 = 4 * max(blck, 64), 6
        nb = div(ne0, blck) * ne1
        data = rand(rng, UInt8, nb * tsize)
        for b in 0:(nb-1)   # keep every scale finite and moderate
            data[b*tsize+1:b*tsize+2] .= _dq_f16(rand(rng, (-1, 1)) * (0.01 + rand(rng)))
        end
        W = permutedims(reshape(_DQ._dequant_f32(data, 0, ty, ne0 * ne1), ne0, ne1))
        x = randn(rng, Float32, ne0)
        y = zeros(Float32, ne1)
        _DQ._qgemv_bytes!(y, x, data, ty, ne0, ne1, 1.0f0, 0.0f0)
        @testset "type $ty" begin
            @test y ≈ W * x rtol = 1e-4
        end
    end
end

@testset "native dequant bit-equal to ggml to_float" begin
    if !isfile(joinpath(get(ENV, "GESSO_GGML_DIR", ""), "libggml-base.so"))
        _dq_skip("set GESSO_GGML_DIR to a directory with libggml-base.so — native vs ggml dequant skipped")
    else
        rng = Xoshiro(29)
        # ggml has no to_float for F32; it is a copy, checked by hand above.
        for ty in sort!(collect(keys(_DQ._GGML_KNOWN_BLOCKS)))
            ty == _DQ.GGML_TYPE_F32 && continue
            data, n = _dq_random_blocks(rng, ty, 4)
            mine = _DQ._dequant_f32(data, 0, ty, n)
            ggml = _DQ._ggml_lib_dequant(data, 0, ty, n)
            @testset "type $ty" begin
                @test all(_dq_same.(mine, ggml))
            end
        end
    end
end
