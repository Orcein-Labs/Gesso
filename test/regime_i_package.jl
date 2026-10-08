using Gesso, Test, JSON, SHA
mode=ARGS[1]
backend=if mode=="lava"
    @eval using Lava
    Gesso.LavaBackend()
else
    Gesso.CPUBackend()
end
ref=JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
model, ts, cfg=Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"])
mode=="lava" && (ts=Gesso.to_device(backend, ts))
s=Gesso.Session(
    model,
    ts;
    backend,
    context_length=64,
    page_size=4,
    eos_token_id=0,
    eps=cfg.rms_norm_eps,
    sink=Gesso.InMemorySink(),
)
records=[]
@testset "Relocated fresh-depot $mode real inference" begin
    @test startswith(pathof(Gesso), abspath(ARGS[3]))
    @test Gesso.Inference.PrefillWorkload===Gesso.Semantics.PrefillWorkload
    @test Gesso.Inference.DecodeWorkload===Gesso.Semantics.DecodeWorkload
    @test Gesso.Inference.KVCache===Gesso.Parameters.KVCache
    for c in ref["cases"]
        prompt=Int.(c["prompt_ids"])
        logits=Gesso.prefill!(s, prompt)
        # Existing external reference's last full-vocabulary row.
        expected=Float64.(c["last_logits"])
        delta=maximum(abs.(Float64.(logits[:, end]) .- expected))
        @test delta<=0.01
        ids=Gesso.generate(s, prompt; max_new_tokens=8)
        @test ids==Int.(c["generated_ids"])
        r=s.sink.buf[end]
        @test r.failure===nothing && r.inference_request.actual_backend==Symbol(mode)
        push!(records, (; prompt=c["prompt"], delta, ids, digest=r.output_digest))
        Gesso.Inference._session_reset!(s)
    end
end
manifest=joinpath(ARGS[3], "Manifest.toml")
write(
    ARGS[2],
    JSON.json((;
        schema="gesso-fresh-package-v1",
        backend=mode,
        julia=string(VERSION),
        source=pathof(Gesso),
        depots=DEPOT_PATH,
        manifest_sha256=bytes2hex(sha256(read(manifest))),
        records,
        compiled_modules="existing only, upper depot initially empty",
        cache_boundary="readonly cached source/artifacts, no inherited compiled Gesso/Lava",
    )),
)
