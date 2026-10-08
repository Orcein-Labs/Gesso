# Actual-host GPU probe, no silent skip. Optional dependency belongs to test env.
using Gesso, CUDA, JSON, Test, Profile
CUDA.functional() || error("campaign device gate requires a functional CUDA device")
CUDA.allowscalar(false)
checkpoint=ENV["GESSO_SMOLLM2_DIR"];
ref=JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
model, ts, cfg=Gesso.load_llama(checkpoint);
backend=Gesso.CUDABackend()
gpu=Gesso.to_device(backend, ts)
results=[]
@testset "Regime I CUDA real-model parity and memory" begin
    @test gpu.rope == ts.rope
    @test eltype(gpu.embedding.storage)==Float32
    s=Gesso.Session(
        model,
        gpu;
        backend,
        context_length=128,
        page_size=16,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )
    for case in (get(ENV, "GESSO_PROFILE_ALLOCATIONS", "")=="1" ? [] : ref["cases"])
        ids=Int.(case["prompt_ids"])
        Gesso.Inference._session_reset!(s)
        logits=Gesso.prefill!(s, ids)
        CUDA.synchronize()
        delta=maximum(
            abs.(Float64.(Array(logits)[:, end]) .- Float64.(case["last_logits"])),
        )
        @test delta <= 1e-2
        @test Gesso.generate(s, ids; max_new_tokens=8)==Int.(case["generated_ids"])
        @info "CUDA independent parity" prompt=case["prompt"] delta
        push!(results, (; prompt=case["prompt"], max_abs_delta=delta))
    end
    Gesso.Inference._session_reset!(s)
    Gesso.prefill!(s, Int.(ref["cases"][1]["prompt_ids"]))
    Gesso.decode!(s)
    Gesso.decode!(s)
    CUDA.synchronize()
    GC.gc(true)
    bytes=@allocated begin
        Gesso.decode!(s)
        CUDA.synchronize()
    end
    if get(ENV, "GESSO_PROFILE_ALLOCATIONS", "")=="1"
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=1.0 Gesso.decode!(s)
        CUDA.synchronize()
        groups=Dict{String, Tuple{Int, Int}}()
        for allocation in Profile.Allocs.fetch().allocs
            frames=filter(f->occursin("work/gesso", string(f.file)), allocation.stacktrace)
            isempty(frames) && continue
            frame=first(frames)
            key=string(frame.file, ":", frame.line, " ", frame.func)
            size, count=get(groups, key, (0, 0))
            groups[key]=(size+allocation.size, count+1)
        end
        write(ARGS[1]*".profile.json", JSON.json(sort(collect(groups); by=p->-last(p)[1])))
    end
    @info "Warmed CUDA decode allocation" bytes
    @test bytes <= 1024^2
    write(
        ARGS[1],
        JSON.json((;
            schema="gesso-device-memory-v1",
            device=string(CUDA.device()),
            dtype="Float32",
            allocation_bytes=bytes,
            ceiling_bytes=1024^2,
            results,
        )),
    )
end
