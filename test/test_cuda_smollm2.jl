# Phase 4 item D (§LXXVII): SmolLM2-135M prefill on CUDA vs the CPU golden.
#
# Laws (§LXXVII, same shape as the Phase 3 real-model gate §LXXVI):
#   * Named skip unless BOTH a device AND GESSO_SMOLLM2_DIR exist. The gate
#     never downloads and never talks to huggingface.co.
#   * No golden CUDA logits file is faked. The CPU golden
#     (test/fixtures/smollm2/expected_logits.toml, oracle "gesso-cpu") is the
#     reference; CUDA compares to it at the WIDER declared atol=1e-2
#     (F32 device vs the frozen F64 oracle, 135M params).
#   * The fingerprint file on disk stays the CPU one; nothing here rewrites it.

using Test: AbstractTestSet, Broken, get_testset, record

using TOML

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const SMOLLM2_DIR = get(ENV, "GESSO_SMOLLM2_DIR", nothing)
const GOLDEN_PATH = joinpath(@__DIR__, "fixtures", "smollm2", "expected_logits.toml")
const SMOLLM2_PROMPT = "Hello"   # the same pinned prompt as the CPU gate

# CUDA reachability: the test env declares CUDA (§LXXVII); if it somehow
# fails to load, the gate degrades to the named skip, never an error.
const CUDA_OK = let
    ok = true
    try
        @eval Main using CUDA
        ok = CUDA.functional()
    catch
        ok = false
    end
    ok
end

@testset "SmolLM2-135M CUDA prefill vs CPU golden (§LXXVII item D)" begin
    if !CUDA_OK
        @test _skip(
            "no NVIDIA device (CUDA.functional() == false) — SmolLM2 CUDA gate skipped (§LXXVII skip law)",
        )
    elseif SMOLLM2_DIR === nothing
        @test _skip(
            "GESSO_SMOLLM2_DIR is unset — set it to a local snapshot of HuggingFaceTB/SmolLM2-135M to run the SmolLM2 CUDA gate (never downloads)",
        )
    elseif !isdir(SMOLLM2_DIR) ||
           !isfile(joinpath(SMOLLM2_DIR, "config.json")) ||
           !isfile(joinpath(SMOLLM2_DIR, "vocab.json")) ||
           !isfile(joinpath(SMOLLM2_DIR, "merges.txt")) ||
           (
               !isfile(joinpath(SMOLLM2_DIR, "model.safetensors")) &&
               !isfile(joinpath(SMOLLM2_DIR, "model.safetensors.index.json"))
           )
        @test _skip(
            "GESSO_SMOLLM2_DIR=$SMOLLM2_DIR is not a usable snapshot — the gate never downloads (§LXXVI)",
        )
    elseif !isfile(GOLDEN_PATH)
        # 10D item B: snapshot present + golden missing is a BROKEN TREE, not
        # a skip — the frozen oracle is committed (10C); a vanished golden is
        # a regression on this box, never "not frozen yet" (§LXX).
        error(
            "snapshot present but test/fixtures/smollm2/expected_logits.toml is MISSING — " *
            "the golden is frozen (oracle \"gesso-cpu\") and must not vanish; restore it, " *
            "or regenerate deliberately with test/freeze_smollm2_golden.jl --force",
        )
    else
        cuda = Gesso.CUDABackend()
        model, cpu_tensors, cfg = Gesso.load_llama(SMOLLM2_DIR)
        tk = Gesso.load_gpt2_tokenizer(SMOLLM2_DIR)
        ids = Gesso.encode(tk, SMOLLM2_PROMPT)
        @test isempty(ids) == false

        # released-config composition facts (§LXXVI pinned table)
        @test cfg.num_hidden_layers == 30
        @test cfg.num_attention_heads == 9
        @test cfg.num_key_value_heads == 3

        # device prefill; host copy back is explicit (§LXXVII)
        gpu_tensors = Gesso.to_device(cuda, cpu_tensors)
        gpu_logits = Array(
            Gesso.reference_prefill(
                model,
                gpu_tensors,
                ids;
                backend=cuda,
                eps=cfg.rms_norm_eps,
                theta=cfg.rope_theta,
            ),
        )
        @test size(gpu_logits) == (49152, length(ids))

        # the frozen CPU golden is the reference: last-position logits,
        # atol=1e-2 (F32 device vs frozen F64 oracle, 135M)
        golden = TOML.parsefile(GOLDEN_PATH)
        prov = get(golden, "provenance", Dict{String, Any}())
        @test get(prov, "oracle", "") == "gesso-cpu"
        # 10C item B format: one compact `values` array of 49152 Float64s
        values = Float64.(golden["values"])
        @test length(values) == 49152
        last = gpu_logits[:, end]
        @test all(i -> isapprox(last[i], values[i]; atol=1e-2, rtol=0.0), eachindex(values))
        println("smollm2 cuda-vs-cpu-golden max|Δlogit| = ", maximum(abs, last .- values))
    end
end
