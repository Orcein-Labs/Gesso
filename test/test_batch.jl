@testset "Runtime ordinary scheduling contract" begin
    R=Gesso.Runtime
    @test Set(names(R))==Set([:Runtime, :BatchRequest, :BatchResult, :run_batch, :cancel!])
    model=toy2_modelir()
    ts=toy2_tensors()
    session(; eos=2) = Gesso.Session(
        model,
        ts;
        context_length=32,
        eos_token_id=eos,
        sink=Gesso.InMemorySink(),
    )
    prompts=[[1, 3], [1, 3, 4], [1, 3, 4, 5]]
    requests=[
        R.BatchRequest(session(), p; max_new_tokens=i) for (i, p) in enumerate(prompts)
    ]
    results=R.run_batch(requests)
    for i in 1:3
        @test results[i].ids==Gesso.reference_generate(
            model,
            ts,
            prompts[i];
            max_new_tokens=i,
        )
        @test results[i].error===nothing
    end
    q=R.BatchRequest(session(), prompts[1]; max_new_tokens=0)
    @test R.run_batch([q])[1].ids==prompts[1]
    @test R.run_batch(R.BatchRequest[])==R.BatchResult[]
    @test_throws Gesso.GessoError R.BatchRequest(session(), prompts[1]; max_new_tokens=-1)
    @test_throws Gesso.GessoError R.run_batch([q, q])
    eos=argmax(Gesso.reference_prefill(model, ts, prompts[1])[:, end])-1
    q=R.BatchRequest(session(; eos), prompts[1]; max_new_tokens=5)
    result=only(R.run_batch([q]))
    @test result.status==:eos && result.ids==vcat(prompts[1], eos)
    qs=[R.BatchRequest(session(), p; max_new_tokens=3) for p in prompts]
    results=R.run_batch(qs; on_token=(i, id) -> i==1 && error("injected callback failure"))
    @test results[1].status==:failed && length(results[1].ids)==length(prompts[1])+1
    @test results[1].error isa Gesso.GessoError && results[1].error.code==Gesso.ERR_RUNTIME
    @test all(
        results[i].ids==Gesso.reference_generate(model, ts, prompts[i]; max_new_tokens=3)
        for i in 2:3
    )
    s=session()
    errs=Any[]
    Gesso.generate(
        s,
        prompts[1];
        max_new_tokens=1,
        on_token=id->push!(errs, try
            Gesso.generate(s, prompts[1])
            nothing
        catch e
            e
        end),
    )
    @test only(errs) isa Gesso.GessoError && only(errs).code==Gesso.ERR_INVALID_PLAN
    @test !s.busy && !islocked(s.run_lock)
    lock(s.run_lock)
    try
        task=Threads.@spawn try
            Gesso.fork(s)
            nothing
        catch e
            e
        end
        err=fetch(task)
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    finally
        unlock(s.run_lock)
    end

end
