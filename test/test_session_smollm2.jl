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

        # ---- Phase 10C item C: the named-model share figure (bytes) --------
        # `fork` on the named model shares pages AS BYTES: after prefill! of
        # "Hello" and BEFORE any decode, parent+child store ONE copy of the
        # KV; two independent prefills store TWO. Measured 2026-10-02 on the
        # demo snapshot (F64 CPU): N = 1_474_560 — "Hello" is ONE token, so
        # one live page per cache: 30 layers × 2 (K,V) × 16 × 3 kv-heads ×
        # 64 d-head × 8 (F64). Pinned so silent growth fails; the measured
        # N agrees with the 10C sanity class (token count is the why).
        parent = _s2(page_size=16)
        Gesso.prefill!(parent, ids)
        child = Gesso.fork(parent)
        N = Gesso.Profiling.kv_footprint(parent.mgr)
        @test N == 1_474_560
        @test Gesso.Profiling.unique_kv_bytes(parent.mgr, child.mgr) == N
        a = _s2(page_size=16)
        b = _s2(page_size=16)
        Gesso.prefill!(a, ids)
        Gesso.prefill!(b, ids)
        @test Gesso.Profiling.unique_kv_bytes(a.mgr, b.mgr) == 2N
        # no decode happened on parent/child — the byte check precedes decode
        @test child.seqlen == parent.seqlen == length(ids)

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

            # Phase 10C item C: CUDA repeats the IDENTITIES (fork == one
            # session; isolated == sum). Do not require CUDA bytes == CPU
            # bytes (F32 device storage vs F64 host).
            ps = _s2(backend=cuda, ts=gpu_tensors, page_size=16)
            Gesso.prefill!(ps, ids)
            pc = Gesso.fork(ps)
            n_gpu = Gesso.Profiling.kv_footprint(ps.mgr)
            @test Gesso.Profiling.unique_kv_bytes(ps.mgr, pc.mgr) == n_gpu
            qa = _s2(backend=cuda, ts=gpu_tensors, page_size=16)
            qb = _s2(backend=cuda, ts=gpu_tensors, page_size=16)
            Gesso.prefill!(qa, ids)
            Gesso.prefill!(qb, ids)
            @test Gesso.Profiling.unique_kv_bytes(qa.mgr, qb.mgr) == 2 * n_gpu
            println("smollm2 cuda unique_kv_bytes (fork pair) = ", n_gpu)
        end
    end
end
