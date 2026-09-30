# Phase 5 item D tests (§LXXVIII): the engine runs the models we already
# import. llama_micro: Session generate ids EQUAL reference_generate (CPU),
# CUDA skip-or-green with ids equal when a device exists. The micro
# checkpoint is written in-test by the Phase 3 writer (mirror of the reader,
# byte-for-byte) — no fixture files, no downloads. Its written config has
# eos_token_id = 0; the Session takes that from the caller (the engine never
# hardcodes a model's EOS).

using .GessoTestHelpers: approx_eq

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

@testset "llama_micro Session vs oracle (§LXXVIII item D, GQA engine)" begin
    dir = mktempdir()
    make_micro_checkpoint(dir)
    model, cpu_tensors, cfg = Gesso.load_llama(dir)
    tokens = [0, 1, 2]

    _micro_session(; backend=Gesso.CPUBackend(), ts=cpu_tensors, kw...) = Gesso.Session(
        model,
        ts;
        backend=backend,
        context_length=32,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
        kw...,
    )

    cpu_ids = Gesso.generate(_micro_session(), tokens; max_new_tokens=3)
    @test cpu_ids == Gesso.reference_generate(
        model,
        cpu_tensors,
        tokens;
        max_new_tokens=3,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )

    # prefill! logits match the oracle at atol=0 on CPU (GQA engine path)
    logits = Gesso.prefill!(_micro_session(), tokens)
    @test approx_eq(
        logits,
        Gesso.reference_prefill(
            model,
            cpu_tensors,
            tokens;
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        );
        atol=0.0,
    )

    # page_size=4 crosses boundaries on the GQA model too
    @test Gesso.generate(_micro_session(; page_size=4), tokens; max_new_tokens=3) == cpu_ids

    # CUDA skip-or-green: ids equal the CPU Session ids when a device exists
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
            "no NVIDIA device (CUDA.functional() == false) — llama_micro CUDA Session skipped (§LXXVIII skip law)",
        )
    else
        cuda = Gesso.CUDABackend()
        gpu_tensors = Gesso.to_device(cuda, cpu_tensors)
        gpu_ids = Gesso.generate(
            _micro_session(backend=cuda, ts=gpu_tensors),
            tokens;
            max_new_tokens=3,
        )
        @test gpu_ids == cpu_ids                     # argmax identity gate
        gpu_ids_paged = Gesso.generate(
            _micro_session(backend=cuda, ts=gpu_tensors; page_size=4),
            tokens;
            max_new_tokens=3,
        )
        @test gpu_ids_paged == cpu_ids
    end
end
