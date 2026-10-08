using Gesso, Test, JSON
mode=ARGS[1]
if mode=="lava"
    using Lava
    backend=Gesso.LavaBackend()
else
    backend=Gesso.CPUBackend()
end
ref=JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
model, ts, cfg=Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"])
mode=="lava" && (ts=Gesso.to_device(backend, ts))
sink=Gesso.InMemorySink()
s=Gesso.Session(
    model,
    ts;
    backend,
    context_length=64,
    page_size=4,
    eos_token_id=0,
    eps=cfg.rms_norm_eps,
    sink,
)
prompt=Int.(ref["cases"][1]["prompt_ids"]);
expected=Int.(ref["cases"][1]["generated_ids"])
errors=String[]
@testset "Contained $mode failure and recovery" begin
    for exception in
        (ErrorException("injected operator-facing callback failure"), InterruptException())
        count=Ref(0)
        emitted=Int[]
        err=try
            Gesso.generate(
                s,
                prompt;
                max_new_tokens=8,
                on_token=id->begin
                    count[]+=1
                    push!(emitted, id)
                    count[]==2 && throw(exception)
                end,
            )
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_RUNTIME
        @test err.detail[:interrupted]==(exception isa InterruptException)
        @test emitted==expected[(length(prompt)+1):(length(prompt)+2)]
        @test count[]==2 && !s.ready && !s.busy && !islocked(s.run_lock)
        @test sink.buf[end].failure===err && sink.buf[end].token_usage.new_tokens==2
        err2=try
            Gesso.decode!(s)
            nothing
        catch e
            e
        end
        @test err2 isa Gesso.GessoError && err2.code==Gesso.ERR_INVALID_PLAN
        @test Gesso.generate(s, prompt; max_new_tokens=8)==expected
        push!(errors, string(typeof(exception)))
    end
    for bad in (-1, model.vocab_size, typemax(Int))
        err=try
            Gesso.generate(s, [bad]; max_new_tokens=1)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
        @test !s.ready && sink.buf[end].failure===err
        @test Gesso.generate(s, prompt; max_new_tokens=8)==expected
    end
    table=ts.embedding
    dst=Gesso.Activation(;
        shape=(1, size(table.storage, 2)),
        storage=fill!(similar(table.storage, (1, size(table.storage, 2))), 0),
    )
    for wl in (Gesso.PrefillWorkload(), Gesso.DecodeWorkload()),
        id in (-1, model.vocab_size)

        err=try
            Gesso.embedding_lookup!(backend, dst, table, [id], wl)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    end
    # Structurally malformed device/host parameter fails before a projection launch.
    badhead=Gesso.EmbeddingTable(; shape=(1, 1), storage=ts.embedding.storage)
    err=try
        Gesso.Session(
            model,
            merge(ts, (; lm_head=badhead));
            backend,
            context_length=16,
            eos_token_id=0,
        )
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    # EOS is a terminal batch result on the same primary engine.
    eos=expected[length(prompt)+1]
    eoss=Gesso.Session(
        model,
        ts;
        backend,
        context_length=16,
        eos_token_id=eos,
        eps=cfg.rms_norm_eps,
    )
    result=only(
        Gesso.Runtime.run_batch([
            Gesso.Runtime.BatchRequest(eoss, prompt; max_new_tokens=8),
        ]),
    )
    @test result.status==:eos && result.ids==vcat(prompt, eos)
end
write(
    ARGS[2],
    JSON.json((;
        schema="gesso-containment-v1",
        backend=mode,
        interrupted_callbacks=errors,
        invalid_ids=[-1, model.vocab_size, string(typemax(Int))],
        recovery="exact HF eight-token sequence",
        driver_loss="not forced or certified",
    )),
)
