# Decode stage profiler (src/Inference/decode_profile.jl): off is a no-op that
# allocates nothing; on accumulates only inside a token window.

using Test
using Gesso
import Gesso.Inference: @decode_stage
const DPI = Gesso.Inference

stage_value(x) = @decode_stage :embed Gesso.CPUBackend() (x + 1)

function stage_loop(n)
    acc = 0
    for i in 1:n
        acc += @decode_stage :norm nothing i
    end
    return acc
end

@testset "decode stage profiler (GESSO_PROFILE)" begin
    saved = DPI._PROFILE[]
    try
        @testset "off: transparent and allocation free" begin
            DPI.profile_disable!()
            @test DPI._PROFILE[] === nothing
            @test stage_value(1) == 2
            stage_loop(10)
            @test stage_loop(1000) == sum(1:1000)
            @test (@allocated stage_loop(1000)) == 0
            @test DPI._profile_token_begin() === UInt64(0)
            DPI._profile_token_end(UInt64(0), true)
            DPI._profile_copy!(true)
            @test DPI._PROFILE[] === nothing
            @test_throws ErrorException DPI.profile_snapshot()
        end

        @testset "on: accumulates inside a token window only" begin
            DPI.profile_enable!()
            p = DPI._PROFILE[]
            @test p isa DPI.DecodeProfile
            @test all(==(0), values(p.ns))

            # outside a window: transparent, not recorded
            @test stage_value(1) == 2
            @test p.calls[:embed] == 0

            t0 = DPI._profile_token_begin()
            @test stage_value(1) == 2
            @test p.calls[:embed] == 1
            @decode_stage :gdn_mixer nothing sleep(0.01)
            @test p.calls[:gdn_mixer] == 1
            @test p.ns[:gdn_mixer] >= 5_000_000
            DPI._profile_copy!(true)
            DPI._profile_copy!(false)
            DPI._profile_copy!(false)
            DPI._profile_token_end(t0, true)
            @test p.window == false
            @test p.tokens == 1
            @test p.d2h == 1
            @test p.h2d == 2
            @test p.token_ns >= p.ns[:gdn_mixer]

            # a head-only window (no decode) adds time but no token
            t1 = DPI._profile_token_begin()
            DPI._profile_token_end(t1, false)
            @test p.tokens == 1

            # closed window again: not recorded
            stage_value(1)
            @test p.calls[:embed] == 1

            DPI.profile_reset!()
            @test p.tokens == 0 && p.d2h == 0 && p.ns[:gdn_mixer] == 0
        end

        @testset "delta and report" begin
            DPI.profile_enable!()
            t = DPI._profile_token_begin()
            @decode_stage :moe_host nothing sum(1:1000)
            DPI._profile_token_end(t, true)
            a = DPI.profile_snapshot()

            t = DPI._profile_token_begin()
            @decode_stage :moe_host nothing sum(1:1000)
            DPI._profile_copy!(true)
            DPI._profile_token_end(t, true)
            b = DPI.profile_snapshot()

            d = DPI.profile_delta(b, a)
            @test d.tokens == 1
            @test d.calls[:moe_host] == 1
            @test d.d2h == 1
            @test d.h2d == 0
            @test d.token_ns > 0

            io = IOBuffer()
            DPI.print_decode_profile(io, d; label="unit")
            out = String(take!(io))
            @test occursin("moe_router", out)
            @test occursin("other", out)
            @test occursin("copies/token  d2h=1.0  h2d=0.0", out)

            @test_throws ErrorException DPI.print_decode_profile(
                IOBuffer(),
                DPI.DecodeProfile(),
            )
        end
    finally
        DPI._PROFILE[] = saved
    end
end
