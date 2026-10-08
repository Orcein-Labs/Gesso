using Gesso, Test, JSON
mode=ARGS[1]
if mode=="lava"
    using Lava
    backend=Gesso.LavaBackend()
    F=Float32
    transfer(a) = Lava.LavaArray(F.(a))
    atol=1e-3
    rtol=2e-5
else
    backend=Gesso.CPUBackend()
    F=Float64
    transfer(a) = F.(a)
    atol=1e-10
    rtol=1e-12
end
matjson(rows) = permutedims(reduce(hcat, Vector{F}.(rows)))
wrap(f, a) = f(; shape=size(a), storage=transfer(a))
ref=JSON.parsefile(ARGS[2]);
expected=ref["cases"][F===Float64 ? "torch.float64" : "torch.float32"]
x=matjson(ref["rows"]);
sc=F.(ref["scale"]);
weights=matjson(ref["weights"]);
scores=matjson(ref["scores"])
A=Gesso.Activation;
W=Gesso.TemporaryWorkspace
function numerical_failure(f)
    err=try
        f()
        nothing
    catch e
        e
    end
    @info "Numerical rejection" actual_type=typeof(err) actual_error=(
        err===nothing ? "nothing" : sprint(showerror, err)
    )
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_NUMERICAL_INSTABILITY
end
@testset "Independent native $mode numeric gates" begin
    for wl in (Gesso.PrefillWorkload(), Gesso.DecodeWorkload())
        y=wrap(A, zeros(F, size(x)))
        Gesso.rmsnorm!(
            backend,
            y,
            wrap(A, x),
            wrap(Gesso.FrozenParameter, sc),
            wl;
            eps=ref["eps"],
        )
        @test isapprox(Array(y.storage), matjson(expected["rmsnorm"]); atol, rtol)
        Gesso.swiglu!(
            backend,
            y,
            wrap(A, x),
            wrap(A, repeat(reshape(sc, 1, :), size(x, 1))),
            wl,
        )
        @test isapprox(Array(y.storage), matjson(expected["swiglu"]); atol, rtol)
        projection=wrap(A, zeros(F, size(x, 1), size(weights, 1)))
        Gesso.matmul!(
            backend,
            projection,
            wrap(A, x),
            wrap(Gesso.ProjectionWeight, weights),
            wl,
        )
        @test isapprox(Array(projection.storage), matjson(expected["matmul"]); atol, rtol)
        probs=wrap(W, zeros(F, 3, 3))
        Gesso.softmax!(backend, probs, wrap(W, scores), wl)
        @test isapprox(Array(probs.storage), matjson(expected["softmax"]); atol, rtol)
        for bad in (NaN, Inf, -Inf)
            numerical_failure(()->Gesso.Inference._greedy_id(transfer(F[0, bad, 1])))
        end
        @test Gesso.Inference._greedy_id(transfer(F[2, 2, 1]))==0
        numerical_failure(
            ()->Gesso.softmax!(
                backend,
                wrap(W, zeros(F, 1, 2)),
                wrap(W, fill(-Inf, 1, 2)),
                wl,
            ),
        )
        huge=F===Float64 ? 1e200 : 1e20
        numerical_failure(
            ()->Gesso.rmsnorm!(
                backend,
                wrap(A, zeros(F, 1, 4)),
                wrap(A, fill(huge, 1, 4)),
                wrap(Gesso.FrozenParameter, ones(F, 4)),
                wl,
            ),
        )
        for eps in
            (mode=="lava" ? (0.0, -1.0, NaN, Inf, 1e-100, 1e100) : (0.0, -1.0, NaN, Inf))
            err=try
                Gesso.rmsnorm!(
                    backend,
                    y,
                    wrap(A, x),
                    wrap(Gesso.FrozenParameter, sc),
                    wl;
                    eps,
                )
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
        end
    end
    for n in (3, 64, 65, 129, 1025)
        values=zeros(F, n)
        values[end]=2
        @test Gesso.Inference._greedy_id(transfer(values))==n-1
        for index in (1, n)
            poisoned=copy(values)
            poisoned[index]=F(NaN)
            numerical_failure(()->Gesso.Inference._greedy_id(transfer(poisoned)))
        end
    end
    model, ts, cfg=Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"])
    mode=="lava" && (ts=Gesso.to_device(backend, ts))
    hf=JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
    s=Gesso.Session(
        model,
        ts;
        backend,
        context_length=64,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )
    scale=ts.final_rms.storage
    backup=copy(scale)
    fill!(scale, F(NaN))
    numerical_failure(()->Gesso.prefill!(s, Int.(hf["cases"][1]["prompt_ids"])))
    scale .= backup
    for case in hf["cases"]
        ids=Int.(case["prompt_ids"])
        @test Gesso.generate(s, ids; max_new_tokens=8)==Int.(case["generated_ids"])
    end
end
if mode=="lava"
    include("testhelpers.jl")
    include("toyfixtures.jl")
    include("test_modelir.jl")
    include("test_reference_prefill.jl")
    const LAVA_LOADED=true
    const VULKAN_OK=true
    include("test_numeric_lava.jl")
end

write(
    ARGS[3],
    JSON.json((;
        schema="gesso-numerical-gate-v1",
        backend=mode,
        dtype=string(F),
        atol,
        rtol,
        oracle_torch=ref["torch"],
        operators=4,
        workloads=2,
        real_model_prompts=3,
    )),
)
