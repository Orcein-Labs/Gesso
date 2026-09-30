# Phase 5 item D tests (§LXXVIII): SmolLM2 through the Session engine.
#
# Without GESSO_SMOLLM2_DIR: ONE named skip (CI never downloads, §LXXVI).
# With a local snapshot: Session generate("Hello"; max_new_tokens=8) ids
# EQUAL reference_generate on the same snapshot (CPU). CUDA: ids equal the
# CPU Session ids when a device exists — no CUDA golden file is invented.
# The CPU golden (test/fixtures/smollm2/expected_logits.toml) stays the
# oracle for logits; this file gates TOKEN ID identity, not logits.

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const SMOLLM2_DIR = get(ENV, "GESSO_SMOLLM2_DIR", nothing)

@testset "SmolLM2 Session engine (§LXXVIII item D)" begin
    if SMOLLM2_DIR === nothing
        @test _skip(
            "GESSO_SMOLLM2_DIR is unset — set it to a local snapshot of HuggingFaceTB/SmolLM2-135M to run the SmolLM2 Session gate (never downloads)",
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
    else
        model, cpu_tensors, cfg = Gesso.load_llama(SMOLLM2_DIR)
        tk = Gesso.load_gpt2_tokenizer(SMOLLM2_DIR)
        ids = Gesso.encode(tk, "Hello")

        _s2(; backend=Gesso.CPUBackend(), ts=cpu_tensors, kw...) = Gesso.Session(
            model,
            ts;
            backend=backend,
            context_length=128,
            eos_token_id=0,      # the released checkpoint's EOS (§LXXVI)
            tokenizer=tk,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
            kw...,
        )

        # string generate on the real model; ids equal the oracle's
        gen_session = Gesso.generate(_s2(), "Hello"; max_new_tokens=8)
        @test gen_session[1:length(ids)] == ids        # prompt round-trip
        @test gen_session == Gesso.reference_generate(
            model,
            cpu_tensors,
            ids;
            max_new_tokens=8,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        )
        # deterministic: two runs equal
        @test gen_session == Gesso.generate(_s2(), "Hello"; max_new_tokens=8)

        # CUDA: ids equal the CPU Session ids when a device exists
        cuda_ok = let
            ok = true
            try
                @eval Main using CUDA
                ok = CUDA.functional()
            catch
                ok = false
            end
            ok
        end
        if !cuda_ok
            @test _skip(
                "no NVIDIA device (CUDA.functional() == false) — SmolLM2 CUDA Session skipped (§LXXVIII skip law)",
            )
        else
            cuda = Gesso.CUDABackend()
            gpu_tensors = Gesso.to_device(cuda, cpu_tensors)
            gpu_gen =
                Gesso.generate(_s2(backend=cuda, ts=gpu_tensors), "Hello"; max_new_tokens=8)
            @test gpu_gen == gen_session                # argmax identity gate
        end
    end
end
