using Gesso, Test, JSON
include("testhelpers.jl")
include("test_cpu_ops.jl")
using Lava, CUDA
const LAVA_LOADED=true
const VULKAN_OK=true
include("test_lava_ops.jl")
include("test_device_boundaries.jl")
lava=Gesso.LavaBackend();
cuda=Gesso.CUDABackend()
check=Gesso.Inference._check_device_storage
act(s) = Gesso.Activation(; shape=size(s), storage=s)
@testset "Actual Lava backend/storage boundaries" begin
    for s in (
        ones(Float32, 2, 2),
        view(ones(Float32, 3, 2), 1:2, :),
        CUDA.CuArray(ones(Float32, 2, 2)),
        Lava.LavaArray(ones(Float16, 2, 2)),
    )
        err=try
            check(:probe, lava, act(s))
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    end
    ls=Lava.LavaArray(ones(Float32, 3, 2))
    v=view(ls, 1:2, :)
    @test check(:probe, lava, act(v)) === v
    for b in (Gesso.CPUBackend(), cuda)
        err=try
            check(:probe, b, act(ls))
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    end
end
ref=JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
model, ts, cfg=Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"]);
gpu=Gesso.to_device(lava, ts)
records=[]
@testset "Primary Lava real-model interpreter external parity" begin
    for case in ref["cases"]
        ids=Int.(case["prompt_ids"])
        logits=Gesso.reference_prefill(
            model,
            gpu,
            ids;
            backend=lava,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        )
        delta=maximum(
            abs.(Float64.(Array(logits)[:, end]) .- Float64.(case["last_logits"])),
        )
        @test delta <= 1e-2
        @test Gesso.reference_generate(
            model,
            gpu,
            ids;
            backend=lava,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
            max_new_tokens=8,
        )==Int.(case["generated_ids"])
        push!(records, (; prompt=case["prompt"], max_abs_delta=delta))
    end
end
write(
    ARGS[1],
    JSON.json((;
        schema="gesso-primary-device-v1",
        backend="lava",
        hardware="NVIDIA RTX 5060 Vulkan",
        records,
    )),
)
