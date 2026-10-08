# Regime II arm 2: decode one-wait + leftover op-body attribution.
# Compile is excluded. Oracle: Hello IDs exact, planted NaN still ERR.
start = time_ns()
using Gesso, JSON, Test
using Lava

backend = Gesso.LavaBackend()
ext = Base.get_extension(Gesso, :GessoLavaExt)
sync() = ext._lava_sync!()
initialization_seconds = (time_ns() - start) / 1e9

ref = JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
case = only(c for c in ref["cases"] if c["prompt"] == "Hello")
prompt = Int.(case["prompt_ids"])
expected = Int.(case["generated_ids"])

t = time_ns()
model, ts, cfg = Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"])
load_seconds = (time_ns() - t) / 1e9
t = time_ns()
ts = Gesso.to_device(backend, ts)
sync()
transfer_seconds = (time_ns() - t) / 1e9

sink = Gesso.InMemorySink()
s = Gesso.Session(
    model,
    ts;
    backend,
    context_length=64,
    page_size=16,
    eos_token_id=0,
    eps=cfg.rms_norm_eps,
    sink,
)

@testset "Regime II arm 2 Hello IDs" begin
    t = time_ns()
    cold = Gesso.generate(s, prompt; max_new_tokens=8)
    sync()
    first_use_seconds = (time_ns() - t) / 1e9
    @test cold == expected
    for _ in 1:2
        @test Gesso.generate(s, prompt; max_new_tokens=8) == expected
        sync()
    end

    Gesso.Inference._session_reset!(s)
    Gesso.prefill!(s, prompt)
    sync()
    ext._lava_audit_enable!(true)
    decode_ids = Int[]
    t = time_ns()
    for _ in 1:8
        push!(decode_ids, Gesso.decode!(s))
    end
    sync()
    audited_decode_seconds = (time_ns() - t) / 1e9
    snap = ext._lava_audit_snapshot()
    ext._lava_audit_enable!(false)
    @test vcat(prompt, decode_ids) == expected

    timings = Float64[]
    for _ in 1:3
        Gesso.Inference._session_reset!(s)
        Gesso.prefill!(s, prompt)
        sync()
        GC.gc(true)
        sync()
        got = Int[]
        stats = @timed begin
            for _ in 1:8
                push!(got, Gesso.decode!(s))
            end
            sync()
        end
        push!(timings, stats.time)
        @test vcat(prompt, got) == expected
    end

    n_layers = length(model.blocks)
    n_heads = s.n_heads
    audited_ns = max(UInt64(1), round(UInt64, audited_decode_seconds * 1e9))
    op_ns = snap.embed_ns + snap.rms_ns + snap.rope_ns + snap.matmul_ns + snap.softmax_ns + snap.swiglu_ns
    residual_ns = audited_ns > (snap.sync_ns + snap.finite_ns + op_ns) ?
        audited_ns - snap.sync_ns - snap.finite_ns - op_ns : UInt64(0)

    Gesso.Inference._session_reset!(s)
    Gesso.prefill!(s, prompt)
    sync()
    fill!(s.h, NaN32)
    planted = try
        Gesso.decode!(s)
        nothing
    catch e
        e
    end
    planted_ok = planted isa Gesso.GessoError && planted.code == Gesso.ERR_NUMERICAL_INSTABILITY
    @test planted_ok

    receipt = (;
        schema="gesso-regime-ii-arm2-v1",
        backend="lava",
        prompt="Hello",
        tokens=8,
        n_layers,
        n_heads,
        initialization_seconds,
        load_seconds,
        transfer_seconds,
        first_use_seconds,
        audited_decode_seconds,
        unaudited_decode_samples_seconds=timings,
        unaudited_median_seconds=sort(timings)[2],
        unaudited_decode_tokens_per_second=8 / sort(timings)[2],
        syncs=snap.syncs,
        finites=snap.finites,
        sync_ns=snap.sync_ns,
        finite_ns=snap.finite_ns,
        syncs_per_token=snap.syncs / 8,
        finites_per_token=snap.finites / 8,
        sync_fraction_of_audited=snap.sync_ns / audited_ns,
        finite_fraction_of_audited=snap.finite_ns / audited_ns,
        predicted_syncs_per_token=1,
        embed_n=snap.embed_n,
        rms_n=snap.rms_n,
        rope_n=snap.rope_n,
        matmul_n=snap.matmul_n,
        softmax_n=snap.softmax_n,
        swiglu_n=snap.swiglu_n,
        embed_ns=snap.embed_ns,
        rms_ns=snap.rms_ns,
        rope_ns=snap.rope_ns,
        matmul_ns=snap.matmul_ns,
        softmax_ns=snap.softmax_ns,
        swiglu_ns=snap.swiglu_ns,
        embed_fraction=snap.embed_ns / audited_ns,
        rms_fraction=snap.rms_ns / audited_ns,
        rope_fraction=snap.rope_ns / audited_ns,
        matmul_fraction=snap.matmul_ns / audited_ns,
        softmax_fraction=snap.softmax_ns / audited_ns,
        swiglu_fraction=snap.swiglu_ns / audited_ns,
        op_body_fraction=op_ns / audited_ns,
        residual_ns=residual_ns,
        residual_fraction=residual_ns / audited_ns,
        ids_exact=vcat(prompt, decode_ids) == expected,
        planted_nan_decode=planted_ok,
        wait_gate=snap.syncs / 8 <= 2,
        device=Lava.gpu_memory_usage(),
    )
    write(ARGS[1], JSON.json(receipt))
    @info "Regime II arm 2" receipt.syncs_per_token receipt.unaudited_decode_tokens_per_second receipt.sync_fraction_of_audited receipt.op_body_fraction receipt.residual_fraction
    @test receipt.wait_gate
end
