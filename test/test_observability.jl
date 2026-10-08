struct RegimeThrowSink <: Gesso.ReceiptSink end
function Gesso.emit!(::RegimeThrowSink, r::Gesso.Receipt)
    haskey(r.context, :committed_ids) && empty!(r.context[:committed_ids])
    error("injected receipt delivery failure")
end
@testset "Truthful engine observability" begin
    model=toy2_modelir()
    ts=toy2_tensors()
    prompt=[1, 3, 4]
    sink=Gesso.InMemorySink()
    s=Gesso.Session(model, ts; context_length=32, eos_token_id=2, sink)
    ids=Gesso.generate(s, prompt; max_new_tokens=4)
    r=sink.buf[end]
    t=r.timing
    @test r.context[:committed_ids]==ids
    @test r.output_digest.algorithm==:fnv1a64_u64le_v1
    @test t.ttft_ns==t.prefill_ns+t.first_decode_ns
    @test t.first_decode_ns<=t.decode_ns
    @test t.ttft_ns<t.prefill_ns+t.decode_ns
    @test r.inference_request.actual_backend==:cpu && r.inference_request.dtype=="Float64"
    Gesso.generate(s, prompt; max_new_tokens=4)
    @test sink.buf[end].output_digest==r.output_digest
    Gesso.generate(s, prompt; max_new_tokens=0)
    @test sink.buf[end].timing.ttft_ns===nothing
    @test sink.buf[end].token_usage.new_tokens==0
    err=try
        Gesso.generate(
            s,
            prompt;
            max_new_tokens=4,
            on_token=id->throw(InterruptException()),
        )
        nothing
    catch e
        e
    end
    r=sink.buf[end]
    @test r.failure===err && r.token_usage.new_tokens==1
    @test r.context[:committed_ids]==ids[1:(length(prompt)+1)]
    throwing=Gesso.Session(
        model,
        ts;
        context_length=32,
        eos_token_id=2,
        sink=RegimeThrowSink(),
    )
    @test Gesso.generate(throwing, prompt; max_new_tokens=4)==ids
    err=try
        Gesso.generate(throwing, Int[])
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    req=Gesso.Runtime.BatchRequest(throwing, prompt; max_new_tokens=2)
    @test only(Gesso.Runtime.run_batch([req])).ids==Gesso.reference_generate(
        model,
        ts,
        prompt;
        max_new_tokens=2,
    )
end
