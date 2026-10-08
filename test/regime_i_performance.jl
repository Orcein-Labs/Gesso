start=time_ns()
using Gesso, JSON, Test
median(x) = sort(x)[2]
mode=ARGS[1]
if mode=="lava"
    using Lava
    backend=Gesso.LavaBackend()
    sync() = Base.get_extension(Gesso, :GessoLavaExt)._lava_sync!()
    device() = Lava.gpu_memory_usage()
elseif mode=="cuda"
    using CUDA
    CUDA.functional() || error("comparison needs actual device")
    CUDA.allowscalar(false)
    CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    backend=Gesso.CUDABackend()
    sync() = CUDA.synchronize()
    device() = (; name=CUDA.name(CUDA.device()))
else
    backend=Gesso.CPUBackend()
    sync() = nothing
    device() = nothing
end
initialization_seconds=(time_ns()-start)/1e9
ref=JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
t=time_ns();
model, ts, cfg=Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"]);
load_seconds=(time_ns()-t)/1e9
t=time_ns();
mode=="cpu" || (ts=Gesso.to_device(backend, ts));
sync();
transfer_seconds=(time_ns()-t)/1e9
sink=Gesso.InMemorySink()
s=Gesso.Session(
    model,
    ts;
    backend,
    context_length=64,
    page_size=16,
    eos_token_id=0,
    eps=cfg.rms_norm_eps,
    sink,
)
records=[]
@testset "Warmed $mode ordinary performance comparison" begin
    for c in ref["cases"]
        prompt=Int.(c["prompt_ids"])
        expected=Int.(c["generated_ids"])
        t=time_ns()
        cold=Gesso.generate(s, prompt; max_new_tokens=8)
        sync()
        first_use_seconds=(time_ns()-t)/1e9
        @test cold==expected
        for warm in 1:2
            @test Gesso.generate(s, prompt; max_new_tokens=8)==expected
            sync()
        end
        timings=[]
        allocations=Int[]
        decode_tps=Float64[]
        gc_seconds=Float64[]
        for repeat in 1:3
            GC.gc(true)
            sync()
            stats=@timed begin
                ids=Gesso.generate(s, prompt; max_new_tokens=8)
                sync()
                ids
            end
            @test stats.value==expected
            r=sink.buf[end]
            push!(timings, stats.time)
            push!(allocations, stats.bytes)
            push!(gc_seconds, stats.gctime)
            push!(decode_tps, 8e9/r.timing.decode_ns)
        end
        @test all(isfinite, timings) && minimum(timings)>0
        @test maximum(timings)/minimum(timings)<2.0
        # Fixed stability gate, not a claim of matching another runtime's speed.
        @test maximum(allocations)-minimum(allocations)<1024^2
        # Warm individual decode allocation in the historical comparison shape.
        Gesso.Inference._session_reset!(s)
        Gesso.prefill!(s, prompt)
        Gesso.decode!(s)
        allocation=@allocated Gesso.decode!(s)
        @info "Warmed comparison" backend=mode prompt=c["prompt"] seconds=median(timings) allocation
        push!(
            records,
            (;
                prompt=c["prompt"],
                first_use_seconds,
                samples_seconds=timings,
                median_seconds=median(timings),
                end_to_end_new_tokens_per_second=8/median(timings),
                median_decode_tokens_per_second=median(decode_tps),
                host_allocations_bytes=allocations,
                warmed_decode_allocation_bytes=allocation,
                gc_seconds,
            ),
        )
    end
end
write(
    ARGS[2],
    JSON.json((;
        schema="gesso-performance-floor-v1",
        backend=mode,
        initialization_seconds,
        load_seconds,
        transfer_seconds,
        records,
        device=device(),
        warmup_calls=2,
        samples=3,
        compile_excluded=true,
        storage_dtype=mode=="cpu" ? "Float64" : "Float32",
        normalization_intermediate=get(
            ENV,
            "GESSO_NORMALIZATION_LABEL",
            mode=="cuda" ?
            "Float64 normalization intermediates, Float32 storage (legacy tolerance)" :
            "Float32 native",
        ),
        workload="single session greedy, host-visible IDs",
        comparison="implementation pipelines; not identical floating point instruction sequences",
    )),
)
