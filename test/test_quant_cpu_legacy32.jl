# CPU fused GEMV for the 32-value legacy blocks IQ4_NL (type 20) and Q5_1
# (type 7), used by the gemma4 expert banks:
#   * synthetic bytes: fused GEMV against a Julia dequant reference, always run
#   * the Julia dequant against ggml to_float, bit-exact (needs GESSO_GGML_DIR)
#   * real expert slices from the gemma4 GGUF files (env GESSO_GEMMA4_GGUF,
#     skipped if unset): fused GEMV against ggml to_float + F32 product

using Test
using Random: Xoshiro

const _G4T_KV = Gesso.Inference.KVALUES_IQ4NL
const _G4T_IQ4NL = Gesso.Inference.GGML_TYPE_IQ4_NL
const _G4T_Q51 = Gesso.Inference.GGML_TYPE_Q5_1
const _G4T_GGML_DIR = get(ENV, "GESSO_GGML_DIR", "")
const _G4T_GGUF = get(ENV, "GESSO_GEMMA4_GGUF", "")

_g4t_f16_bytes(v::Float32) = let b = reinterpret(UInt16, Float16(v))
    UInt8[UInt8(b & 0xff), UInt8(b >> 8)]
end

# Random blocks with finite f16 scales (d, and m for Q5_1) at their offsets.
function _g4t_random_bytes(rng, ty::Int32, nblk::Int)
    bsz = Int(Gesso.Inference._iq_block_bytes(ty))
    w = rand(rng, UInt8, nblk * bsz)
    for blk in 0:(nblk - 1)
        o = blk * bsz
        w[o + 1:o + 2] = _g4t_f16_bytes(0.01f0 + 0.1f0 * rand(rng, Float32))
        if ty == _G4T_Q51
            w[o + 3:o + 4] = _g4t_f16_bytes(0.2f0 * (rand(rng, Float32) - 0.5f0))
        end
    end
    return w
end

_g4t_f16(b, o) = Float32(reinterpret(Float16, UInt16(b[o + 1]) | (UInt16(b[o + 2]) << 8)))

# One row of ne0 values, straight from the block layout in the format docs.
function _g4t_ref_row(ty::Int32, b::AbstractVector{UInt8}, ne0::Int)
    out = Vector{Float32}(undef, ne0)
    for ib in 0:(div(ne0, 32) - 1)
        if ty == _G4T_IQ4NL
            o = ib * 18
            d = _g4t_f16(b, o)
            for t in 0:15
                qs = b[o + 3 + t]
                out[ib * 32 + t + 1] = d * Float32(_G4T_KV[(qs & 0x0f) + 1])
                out[ib * 32 + t + 17] = d * Float32(_G4T_KV[(qs >> 4) + 1])
            end
        else
            o = ib * 24
            d = _g4t_f16(b, o)
            m = _g4t_f16(b, o + 2)
            qh = UInt32(b[o + 5]) | (UInt32(b[o + 6]) << 8) | (UInt32(b[o + 7]) << 16) | (UInt32(b[o + 8]) << 24)
            for t in 0:15
                qs = b[o + 9 + t]
                x0 = UInt32(qs & 0x0f) | (((qh >> t) << 4) & UInt32(0x10))
                x1 = UInt32(qs >> 4) | ((qh >> (t + 12)) & UInt32(0x10))
                out[ib * 32 + t + 1] = Float32(x0) * d + m
                out[ib * 32 + t + 17] = Float32(x1) * d + m
            end
        end
    end
    return out
end

function _g4t_ref_matrix(ty::Int32, b::AbstractVector{UInt8}, ne0::Int, nrows::Int)
    rb = Int(Gesso.Inference._qrow_bytes(ty, ne0))
    W = Matrix{Float32}(undef, nrows, ne0)
    for j in 1:nrows
        W[j, :] = _g4t_ref_row(ty, view(b, ((j - 1) * rb + 1):(j * rb)), ne0)
    end
    return W
end

@testset "IQ4_NL and Q5_1 block sizes" begin
    @test Gesso.Inference._iq_block_bytes(_G4T_IQ4NL) == 18
    @test Gesso.Inference._iq_block_bytes(_G4T_Q51) == 24
    @test Gesso.Inference._qrow_bytes(_G4T_IQ4NL, 256) == 8 * 18
    @test Gesso.Inference._qrow_bytes(_G4T_Q51, 256) == 8 * 24
end

@testset "IQ4_NL and Q5_1 fused GEMV vs dequant reference (synthetic)" begin
    rng = Xoshiro(20261011)
    ne0 = 256
    nrows = 5
    for ty in (_G4T_IQ4NL, _G4T_Q51)
        nblk = nrows * div(ne0, 32)
        b = _g4t_random_bytes(rng, ty, nblk)
        W = _g4t_ref_matrix(ty, b, ne0, nrows)
        x = randn(rng, Float32, ne0)
        y_ref = W * x
        y = zeros(Float32, nrows)
        Gesso.Inference._qgemv_bytes!(y, x, b, ty, ne0, nrows, 1.0f0, 0.0f0)
        @test y ≈ y_ref rtol = 1e-5 atol = 1e-4

        yt = zeros(Float32, nrows)
        Gesso.Inference._qgemv_bytes_threaded!(yt, x, b, ty, ne0, nrows, 1.0f0, 0.0f0; nchunks=3)
        @test yt == y

        y2 = fill(2.0f0, nrows)
        Gesso.Inference._qgemv_bytes!(y2, x, b, ty, ne0, nrows, 0.5f0, 3.0f0)
        @test y2 ≈ 0.5f0 .* y_ref .+ 6.0f0 rtol = 1e-5 atol = 1e-4
    end
end

@testset "IQ4_NL and Q5_1 dequant vs ggml to_float (bit-exact)" begin
    if !isfile(joinpath(_G4T_GGML_DIR, "libggml-base.so"))
        @test_skip "set GESSO_GGML_DIR to a directory with libggml-base.so"
        return
    end
    rng = Xoshiro(7)
    ne0 = 512
    for ty in (_G4T_IQ4NL, _G4T_Q51)
        b = _g4t_random_bytes(rng, ty, div(ne0, 32) * 3)
        ref = vec(_g4t_ref_matrix(ty, b, ne0, 3)')
        ggml = Gesso.Inference._ggml_lib_dequant(b, 0, ty, 3 * ne0)
        @test ggml == ref
    end
end

if isfile(_G4T_GGUF) && isfile(joinpath(_G4T_GGML_DIR, "libggml-base.so"))
    @testset "gemma4 expert slices: fused GEMV vs ggml to_float" begin
        hdr = Gesso.gguf_header(_G4T_GGUF)
        seen = Set{Int32}()
        for info in hdr.tensors
            info.ggml_type in (_G4T_IQ4NL, _G4T_Q51) || continue
            occursin("_exps", info.name) || continue
            info.ggml_type in seen && continue
            push!(seen, info.ggml_type)
            slice = Gesso.Inference._expert_slice(hdr, info)
            rng = Xoshiro(Int(info.ggml_type))
            x = randn(rng, Float32, slice.ne0)
            W = Gesso.gguf_dequant_expert(slice, 1)
            y = zeros(Float32, slice.ne1)
            Gesso.Inference._qgemv_expert!(y, x, slice, 1)
            @test y ≈ W * x atol = 2e-4 rtol = 2e-4
        end
        @test !isempty(seen)
    end
else
    @testset "gemma4 expert slices" begin
        @test_skip "set GESSO_GEMMA4_GGUF and GESSO_GGML_DIR to run the gemma4 expert checks"
    end
end
