# Prefill pass profiler (src/Inference/decode_profile.jl, GESSO_PREFILL_PROFILE).
# CPU only: the stages that run on the CPU backend are the mixers and the dense
# FFN, and the MoE and GPU stages are exercised by the CUDA suite instead.

using Test
using Gesso
const PPI = Gesso.Inference

const PREFILL_PROMPT = [1, 3, 4, 5]

function _prefill_run(ts; max_new_tokens=0)
    return reference_generate(ts.model, ts, PREFILL_PROMPT; max_new_tokens)
end

@testset "prefill profiler (GESSO_PREFILL_PROFILE)" begin
    saved_decode = PPI._PROFILE[]
    saved_prefill = PPI._PREFILL_PROFILE[]
    saved_chunk = get(ENV, "GESSO_PREFILL_CHUNK", nothing)
    try
        ts = toy2_tensors()

        @testset "off: no records, no output, no allocation" begin
            ENV["GESSO_PREFILL_CHUNK"] = "2"
            PPI.prefill_profile_disable!()
            PPI._PROFILE[] = nothing
            _prefill_run(ts)
            @test PPI._PREFILL_PROFILE[] === nothing
            @test isempty(PPI.prefill_profile_records())
            @test PPI._PROFILE[] === nothing
            cpu = Gesso.CPUBackend()
            @test PPI._prefill_pass_begin(cpu, 1, 2, 4) === nothing
            PPI._prefill_pass_begin(cpu, 1, 2, 4)
            @test (@allocated PPI._prefill_pass_begin(cpu, 1, 2, 4)) == 0
            @test (@allocated PPI._prefill_pass_end(nothing)) == 0
        end

        @testset "on: one record per pass, rows cover the prompt" begin
            ENV["GESSO_PREFILL_CHUNK"] = "2"
            PPI.prefill_profile_enable!()
            _prefill_run(ts)
            recs = PPI.prefill_profile_records()
            @test length(recs) == 2
            @test [r.pass for r in recs] == [1, 2]
            @test [r.n for r in recs] == [2, 2]
            @test sum(r.n for r in recs) == length(PREFILL_PROMPT)
            @test [r.s0 for r in recs] == [1, 3]
            @test [r.K for r in recs] == [2, 4]
            @test PPI._PROFILE[] === saved_decode
        end

        @testset "on: stages and other add to the pass wall" begin
            for r in PPI.prefill_profile_records()
                @test r.wall_ns > 0
                @test r.other_ns >= 0
                @test sum(values(r.stage_ns)) + r.other_ns == r.wall_ns
                @test all(>=(0), values(r.stage_ns))
            end
            @test any(
                r -> r.stage_ns.attn_mixer + r.stage_ns.gdn_mixer > 0,
                PPI.prefill_profile_records(),
            )
        end

        @testset "on: records and output match the off run" begin
            ENV["GESSO_PREFILL_CHUNK"] = "2"
            PPI.prefill_profile_disable!()
            off = _prefill_run(ts; max_new_tokens=3)
            PPI.prefill_profile_enable!()
            on = _prefill_run(ts; max_new_tokens=3)
            @test on == off
        end

        @testset "on: PFPASS per pass and one PFSUM, flushed to stdout" begin
            ENV["GESSO_PREFILL_CHUNK"] = "2"
            PPI.prefill_profile_enable!()
            path, io = mktemp()
            redirect_stdout(io) do
                _prefill_run(ts)
            end
            close(io)
            lines = split(read(path, String), '\n'; keepempty=false)
            rm(path; force=true)
            pf = filter(startswith("PFPASS"), lines)
            sm = filter(startswith("PFSUM"), lines)
            @test length(pf) == 2
            @test length(sm) == 1
            @test startswith(pf[1], "PFPASS i=1 n=2 K=2 wall_ms=")
            @test startswith(pf[2], "PFPASS i=2 n=2 K=4 wall_ms=")
            @test occursin("PFSUM passes=2 rows=4", sm[1])
        end

        @testset "on: decode profile is untouched by prefill" begin
            PPI.profile_enable!()
            dp = PPI._PROFILE[]
            PPI.prefill_profile_enable!()
            _prefill_run(ts)
            @test PPI._PROFILE[] === dp
            @test !dp.window
            @test all(==(0), values(dp.ns))
            @test dp.token_ns == 0
        end
    finally
        PPI._PROFILE[] = saved_decode
        PPI._PREFILL_PROFILE[] = saved_prefill
        if saved_chunk === nothing
            delete!(ENV, "GESSO_PREFILL_CHUNK")
        else
            ENV["GESSO_PREFILL_CHUNK"] = saved_chunk
        end
    end
end
