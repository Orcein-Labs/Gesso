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
make_session() = Gesso.Session(
    model,
    ts;
    backend,
    context_length=64,
    page_size=4,
    eos_token_id=0,
    eps=cfg.rms_norm_eps,
)
request(i, n) = Gesso.Runtime.BatchRequest(
    make_session(),
    Int.(ref["cases"][i]["prompt_ids"]);
    max_new_tokens=n,
)
expected(i, n) =
    Int.(ref["cases"][i]["generated_ids"])[1:(length(ref["cases"][i]["prompt_ids"])+n)]
R=Gesso.Runtime
memories=[]
function observe_memory(i, id)
    if mode=="lava"
        pool=Lava.pool(Lava.vk_context())
        mapped=[b for b in pool.live_buffers if b.mapped_ptr!=C_NULL]
        stats=Lava.gpu_memory_usage()
        push!(
            memories,
            (;
                request=i,
                token=id,
                mapped_buffers=length(mapped),
                mapped_logical_bytes=sum(b.size for b in mapped),
                stats,
            ),
        )
        @info "Completed owned token" request=i mapped_buffers=length(mapped) live_bytes=stats.live_bytes
    end
end
@testset "Ordinary $mode owned sequence scheduling" begin
    requests=[request(1, 2), request(2, 5), request(3, 8)]
    order=Int[]
    results=R.run_batch(
        requests;
        on_token=(i, id)->(push!(order, i); observe_memory(i, id)),
    )
    for (i, n) in enumerate((2, 5, 8))
        @test results[i].ids==expected(i, n)
        @test results[i].status==:complete
        @test results[i].error===nothing
    end
    @test order==[i for round in 1:8 for i in 1:3 if round<=(2, 5, 8)[i]]
    @test requests[1].session.ws!==requests[2].session.ws
    requests=[request(1, 8), request(2, 8), request(3, 8)]
    R.cancel!(requests[1])
    seen=zeros(Int, 3)
    results=R.run_batch(
        requests;
        on_token=(i, id)->begin
            seen[i]+=1
            i==2 && seen[i]==2 && R.cancel!(requests[i])
        end,
    )
    @test results[1].status==:cancelled && isempty(results[1].ids)
    @test Gesso.Inference.page_count(requests[1].session.mgr)==0
    @test results[2].status==:cancelled && results[2].ids==expected(2, 2)
    @test results[3].ids==expected(3, 8) && results[3].status==:complete
    bad=R.BatchRequest(make_session(), Int[]; max_new_tokens=2)
    results=R.run_batch([request(1, 2), bad, request(3, 4)])
    @test results[1].ids==expected(1, 2)
    @test results[2].status==:failed && results[2].error isa Gesso.GessoError
    @test results[3].ids==expected(3, 4)
    q=request(1, 2)
    err=try
        R.run_batch([q, q])
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    # Cross-task reuse must fail promptly while the original owner retains state.
    lock(q.session.run_lock)
    try
        task=Threads.@spawn try
            Gesso.generate(q.session, Int.(ref["cases"][1]["prompt_ids"]); max_new_tokens=2)
            nothing
        catch e
            e
        end
        err=fetch(task)
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    finally
        unlock(q.session.run_lock)
    end
    @test R.run_batch([q])[1].ids==expected(1, 2)
    # Independent concurrent batches use separate Sessions and scratch.
    if mode=="cpu"
        tasks=[Threads.@spawn R.run_batch([request(i, 2)])[1] for i in 1:3]
        for (i, t) in enumerate(tasks)
            @test fetch(t).ids==expected(i, 2)
        end
    end
end
write(
    ARGS[2],
    JSON.json((;
        schema="gesso-ordinary-batch-v1",
        backend=mode,
        budgets=[2, 5, 8],
        cancellation="before-prefill and after two tokens",
        failed_request="isolated",
        shared_session="rejected",
        gpu_scheduling="single host owner round-robin",
        cpu_concurrency="independent tasks tested",
        memories,
    )),
)
