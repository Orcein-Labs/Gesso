using Gesso, Test, JSON
mode=ARGS[1];
ref=JSON.parsefile(ARGS[2])
matjson(rows) = permutedims(reduce(hcat, Vector{Float64}.(rows)))
@testset "External HF positional frequencies" begin
    for case in ref["ropes"]
        kind=case["kind"]=="default" ? :none : Symbol(case["kind"])
        policy=Gesso.RoPEPolicy(;
            theta=100000.0,
            kind,
            factor=8.0,
            original_max_position_embeddings=8192,
        )
        freq=Gesso.rope_inv_freq(policy, 8)
        kind==:none && (freq=[100000.0^(-2i/8) for i in 0:3])
        @test isapprox(freq, Float64.(case["inv_freq"]); atol=1e-8, rtol=1e-6)
    end
end
mode=="frequency" && exit()
if mode=="lava"
    using Lava
    backend=Gesso.LavaBackend()
    F=Float32
    transfer(a) = Lava.LavaArray(F.(a))
else
    backend=Gesso.CPUBackend()
    F=Float64
    transfer(a) = F.(a)
end
wrap(a) = Gesso.Activation(; shape=size(a), storage=transfer(a))
records=[]
@testset "Primary $mode rotary, causal and ordinary cache context" begin
    for case in ref["ropes"]
        kind=case["kind"]=="default" ? :none : Symbol(case["kind"])
        p=Gesso.RoPEPolicy(;
            theta=100000.0,
            kind,
            factor=8.0,
            original_max_position_embeddings=8192,
        )
        input=reshape(matjson(case["input"]), 5, 1, 8)
        expected=reshape(matjson(case["output"]), 5, 1, 8)
        for wl in (Gesso.PrefillWorkload(), Gesso.DecodeWorkload())
            q=wrap(input)
            k=wrap(input)
            Gesso.rope!(
                backend,
                q,
                k,
                Int.(case["positions"]),
                wl;
                theta=p.theta,
                inv_freq=Gesso.rope_inv_freq(p, 8),
                interleaved=false,
            )
            @test isapprox(Float64.(Array(q.storage)), expected; atol=1e-3, rtol=1e-5)
            @test Array(q.storage)==Array(k.storage)
        end
    end
    for (input, positions, theta, freq) in (
        (zeros(1, 1, 3), [0], 10000.0, nothing),
        (zeros(1, 1, 4), [-1], 10000.0, nothing),
        (zeros(1, 1, 4), [0], NaN, nothing),
        (zeros(1, 1, 4), [0], 10000.0, [1.0]),
    )
        err=try
            Gesso.rope!(
                backend,
                wrap(input),
                wrap(input),
                positions,
                Gesso.PrefillWorkload();
                theta,
                inv_freq=freq,
            )
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    end
    model, ts, cfg=Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"])
    mode=="lava" && (ts=Gesso.to_device(backend, ts))
    # Omitted theta must use imported 100000 metadata, not legacy toy 10000.
    s=Gesso.Session(
        model,
        ts;
        backend,
        context_length=160,
        page_size=16,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
    )
    @test s.theta==cfg.rope_theta==100000.0
    for case in ref["cases"]
        ids=Int.(case["ids"])
        expected=Float64.(case["last_logits"])
        Gesso.Inference._session_reset!(s)
        full=Gesso.prefill!(s, ids)
        full_delta=maximum(abs.(full[:, end] .- expected))
        @test full_delta <= 1e-2
        @test argmax(full[:, end])-1==case["next_id"]
        Gesso.Inference._session_reset!(s)
        Gesso.prefill!(s, ids[1:(end-1)])
        Gesso.Inference._session_consume!(s, ids[end])
        id=Gesso.Inference._session_greedy_id!(s, backend, Gesso.DecodeWorkload())
        cached=Float64.(Array(s.ws.logits.storage)[1, :])
        cached_delta=maximum(abs.(cached .- expected))
        @test cached_delta <= 1e-2
        @test id==case["next_id"]
        @test s.seqlen==length(ids)==Gesso.Inference.kv_len(s.mgr)
        @test Gesso.Inference.page_count(s.mgr)==2*length(model.blocks)*cld(length(ids), 16)
        push!(records, (; length=length(ids), full_delta, cached_delta))
        @info "Context boundary parity" backend=mode length=length(ids) full_delta cached_delta
    end
    ids=Int.(ref["cases"][1]["ids"])
    Gesso.Inference._session_reset!(s)
    before=Gesso.prefill!(s, ids)
    ids[end]=99
    Gesso.Inference._session_reset!(s)
    after=Gesso.prefill!(s, ids)
    @test isapprox(before[:, 1:5], after[:, 1:5]; atol=1e-3, rtol=0)
    tiny=Gesso.Session(
        model,
        ts;
        backend,
        context_length=17,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
    )
    Gesso.prefill!(tiny, ids)
    err=try
        Gesso.Inference._session_consume!(tiny, 1)
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_RESOURCE_LIMIT
end
write(
    ARGS[3],
    JSON.json((;
        schema="gesso-context-gate-v1",
        backend=mode,
        dtype=string(F),
        positions=[0, 1, 8191, 8192, 32768],
        model_lengths=[17, 33, 65, 129],
        records,
    )),
)
