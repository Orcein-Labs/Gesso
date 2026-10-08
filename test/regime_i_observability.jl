using Gesso, Test, JSON
mode=ARGS[1]
if mode=="lava"
    using Lava
    backend=Gesso.LavaBackend()
else
    backend=Gesso.CPUBackend()
end
ref=JSON.parsefile(ENV["GESSO_HF_REFERENCE"]);
digests=JSON.parsefile(ARGS[2])
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
struct ThrowingSink <: Gesso.ReceiptSink end
Gesso.emit!(::ThrowingSink, ::Gesso.Receipt) = error("deliberately broken telemetry")
records=[]
@testset "Independent $mode observable inference" begin
    for (i, c) in enumerate(ref["cases"])
        prompt=Int.(c["prompt_ids"])
        expected=Int.(c["generated_ids"])
        ids=Gesso.generate(s, prompt; max_new_tokens=8)
        r=sink.buf[end]
        @test ids==expected
        @test r.output_digest.value==digests["cases"][i]
        @test r.context[:committed_ids]==expected
        @test r.timing.ttft_ns==r.timing.prefill_ns+r.timing.first_decode_ns
        @test 0<r.timing.first_decode_ns<r.timing.decode_ns
        @test r.timing.total_ns>=r.timing.prefill_ns+r.timing.decode_ns
        @test r.inference_request.actual_backend==Symbol(mode)
        @test r.inference_request.dtype==(mode=="lava" ? "Float32" : "Float64")
        @test r.token_usage.new_tokens==8 && r.token_usage.prompt_tokens==length(prompt)
        @test r.memory_usage.kv_len==length(expected)
        @test r.memory_usage.kv_bytes==sum(
            sizeof(p.storage) for lists in (s.mgr.k_pages, s.mgr.v_pages) for pages in
                                                                              lists for
            p in pages
        )
        push!(
            records,
            (;
                prompt=c["prompt"],
                digest=r.output_digest,
                timing=r.timing,
                token_usage=r.token_usage,
                memory=r.memory_usage,
            ),
        )
    end
    prompt=Int.(ref["cases"][1]["prompt_ids"])
    count=Ref(0)
    err=try
        Gesso.generate(
            s,
            prompt;
            max_new_tokens=8,
            on_token=id->begin
                count[]+=1
                count[]==2 && throw(InterruptException())
            end,
        )
        nothing
    catch e
        e
    end
    r=sink.buf[end]
    @test r.failure===err &&
          r.context[:committed_ids]==Int.(ref["cases"][1]["generated_ids"])[1:(length(
        prompt,
    )+2)]
    @test r.token_usage.new_tokens==2 && r.timing.first_decode_ns<r.timing.decode_ns
    s.tokenizer=Gesso.load_gpt2_tokenizer(ENV["GESSO_SMOLLM2_DIR"])
    @test Gesso.generate(s, ref["cases"][1]["prompt"]; max_new_tokens=8)==Int.(
        ref["cases"][1]["generated_ids"],
    )
    r=sink.buf[end]
    @test r.timing.tokenize_ns>0 &&
          r.timing.ttft_ns==r.timing.tokenize_ns+r.timing.prefill_ns+r.timing.first_decode_ns
    s.sink=ThrowingSink()
    @test Gesso.generate(s, prompt; max_new_tokens=8)==Int.(
        ref["cases"][1]["generated_ids"],
    )
    err=try
        Gesso.generate(s, Int[])
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
end
write(
    ARGS[3],
    JSON.json((;
        schema="gesso-observability-v1",
        backend=mode,
        records,
        partial_failure="two tokens, retained IDs",
        sink_failure="cannot alter inference",
    )),
)
