# Standalone campaign probe. Invocation supplies pinned local checkpoint and oracle.
using Gesso, JSON, SHA, Test, LinearAlgebra
const checkpoint = ENV["GESSO_SMOLLM2_DIR"]
const ref = JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
function campaign_model()
    model, ts, cfg = Gesso.load_llama(checkpoint)
    return model, ts, cfg
end
function campaign_session(model, ts, cfg; context_length=64, page_size=4)
    return Gesso.Session(
        model,
        ts;
        context_length,
        page_size,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
        sink=Gesso.InMemorySink(),
    )
end
arm=ARGS[1];
output=ARGS[2]
if arm == "3"
    records=[]
    @testset "Regime I reproducibility, independent processes" begin
        model, ts, cfg=campaign_model()
        for case in ref["cases"]
            ids=Int.(case["prompt_ids"])
            expected=Int.(case["generated_ids"])
            runs=[]
            for repetition in 1:3
                session=campaign_session(model, ts, cfg)
                got=Gesso.generate(session, ids; max_new_tokens=8)
                @test got == expected
                push!(runs, got)
            end
            @test all(==(runs[1]), runs)
            push!(records, (; prompt=case["prompt"], runs))
        end
        for (name, digest) in ref["checkpoint_sha256"]
            @test bytes2hex(open(sha256, joinpath(checkpoint, name)))==digest
        end
    end
    write(
        output,
        JSON.json((;
            schema="gesso-reproducibility-v1",
            arm=3,
            julia=string(VERSION),
            julia_threads=Threads.nthreads(),
            blas_threads=BLAS.get_num_threads(),
            blas=string(BLAS.get_config()),
            checkpoint_sha256=ref["checkpoint_sha256"],
            oracle=ref["oracle"],
            arithmetic="CPU Float64 greedy",
            records,
        )),
    )
elseif arm == "4"
    records=[]
    @testset "Regime I real-model memory lifecycle" begin
        model, ts, cfg=campaign_model()
        case=ref["cases"][2]
        ids=Int.(case["prompt_ids"])
        expected=Int.(case["generated_ids"])
        s=campaign_session(model, ts, cfg)
        ws=s.ws
        Gesso.generate(s, ids; max_new_tokens=8)
        GC.gc(true)
        live_before=Base.gc_live_bytes()
        retained_before=Base.summarysize(s)
        for repetition in 1:12
            @test Gesso.generate(s, ids; max_new_tokens=8)==expected
            @test s.ws === ws
            pages=2*length(model.blocks)*cld(s.seqlen, s.page_size)
            @test Gesso.Inference.page_count(s.mgr)==pages
            @test Gesso.Inference.kv_bytes(s.mgr)==pages*s.page_size*s.n_kv_heads*s.d_head*sizeof(
                Float64,
            )
            @test all(
                Gesso.Inference.filled_len(s.mgr, l, k)==s.seqlen for
                l in 1:length(model.blocks), k in (:k, :v)
            )
        end
        GC.gc(true)
        live_after=Base.gc_live_bytes()
        retained_after=Base.summarysize(s)
        @test live_after-live_before < 20*1024^2
        @test retained_after-retained_before < 1024^2
        small=campaign_session(model, ts, cfg; context_length=length(ids), page_size=2)
        Gesso.prefill!(small, ids)
        caught=try
            Gesso.decode!(small)
            nothing
        catch err
            err
        end
        @test caught isa Gesso.GessoError
        @test caught.code==Gesso.ERR_RESOURCE_LIMIT
        @test Gesso.generate(small, [0]; max_new_tokens=1)==Gesso.reference_generate(
            model,
            ts,
            [0];
            max_new_tokens=1,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        )
        push!(
            records,
            (;
                live_before,
                live_after,
                retained_before,
                retained_after,
                pages=Gesso.Inference.page_count(s.mgr),
                kv_bytes=Gesso.Inference.kv_bytes(s.mgr),
            ),
        )
        for reload in 1:2
            let (rm, rt, rc)=campaign_model()
                @test Gesso.generate(campaign_session(rm, rt, rc), ids; max_new_tokens=8)==expected
            end
            GC.gc(true)
        end
    end
    write(output, JSON.json((; schema="gesso-memory-lifecycle-v1", arm=4, records)))
else
    error("Unknown campaign arm $arm")
end
