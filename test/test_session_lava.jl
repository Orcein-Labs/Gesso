# Phase 8 item C tests (§LXXXI): the Lava engine — the item B path with
# pages. Gates mirror test_session_cuda.jl: device Session generate token
# ids EQUAL the CPU Session ids (argmax gate), prefill! logits at declared
# atol=1e-3, page-boundary crossing, and the no-copy law. On top of the
# CUDA template this file pins the Phase 7 CoW laws ON VULKAN STORAGE:
# fork aliases the prefix pages (`===`, shared=true) with NO special
# casing, and the child's first decode CoWs the partial dirty page — the
# parent's storage must be UNMOVED afterwards (§LXXX on LavaArray).
#
# One named skip without a usable Vulkan device. LAVA_LOADED / VULKAN_OK
# come from test_lava_seam.jl (include order load-bearing). Helper names
# carry the `_lava` prefix to avoid the `_session` overwrite tangle.

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

using .GessoTestHelpers: approx_eq

const LAVA_PROMPT = [1, 3, 4, 5]

# page_size=4 so the 4-token prompt fills page 1 exactly — the decode side
# must allocate NEW pages while the full shared prefix page stays aliased
_lava_session(model, ts; kw...) =
    Gesso.Session(model, ts; page_size=4, context_length=128, eos_token_id=2, kw...)

# the page table of one (layer, kind) cache
_pagesof_lava(mgr, layer, kind) = Gesso.Inference.kv_cache(mgr, layer, kind).storage

if !LAVA_LOADED || !VULKAN_OK
    _skip(
        "no usable Vulkan device (Lava.vk_context() failed) — Lava Session tests skipped (§LXXXI skip law)",
    )
else
    lava = Gesso.LavaBackend()
    ts = toy2_tensors()
    model = toy2_modelir()

    @testset "toy2 Lava Session vs CPU Session (ids EXACT, logits atol=1e-3)" begin
        gpu_ts = Gesso.to_device(lava, ts)
        cpu_ids = Gesso.generate(
            Gesso.Session(model, ts; context_length=128, eos_token_id=2),
            LAVA_PROMPT;
            max_new_tokens=8,
        )
        gpu_ids = Gesso.generate(
            Gesso.Session(model, gpu_ts; backend=lava, context_length=128, eos_token_id=2),
            LAVA_PROMPT;
            max_new_tokens=8,
        )
        @test gpu_ids == cpu_ids                     # argmax identity gate

        # prefill! logits: host copy is explicit, device-vs-CPU at declared atol
        s_cpu = Gesso.Session(model, ts; context_length=128, eos_token_id=2)
        s_gpu =
            Gesso.Session(model, gpu_ts; backend=lava, context_length=128, eos_token_id=2)
        cpu_logits = Gesso.prefill!(s_cpu, LAVA_PROMPT)
        gpu_logits = Gesso.prefill!(s_gpu, LAVA_PROMPT)   # host-visible (Array)
        @test gpu_logits isa Array{Float32}
        @test size(gpu_logits) == size(cpu_logits)
        @test approx_eq(Float64.(gpu_logits), cpu_logits; atol=1e-3)

        # streaming fires identically on device
        seen = Int[]
        gpu_ids2 = Gesso.generate(
            Gesso.Session(model, gpu_ts; backend=lava, context_length=128, eos_token_id=2),
            LAVA_PROMPT;
            max_new_tokens=3,
            on_token=id -> push!(seen, id),
        )
        @test seen == gpu_ids2[5:end]

        # page_size=4 crosses boundaries on device too — ids must still match
        gpu_ids3 = Gesso.generate(
            _lava_session(model, gpu_ts; backend=lava),
            LAVA_PROMPT;
            max_new_tokens=8,
        )
        @test gpu_ids3 == cpu_ids
    end

    @testset "Lava fork: prefix aliased on device, CoW leaves parent storage unmoved" begin
        gpu_ts = Gesso.to_device(lava, ts)
        s = _lava_session(model, gpu_ts; backend=lava)
        Gesso.prefill!(s, LAVA_PROMPT)
        c = Gesso.fork(s)

        # the prefix is aliased ON DEVICE: same page objects, shared flag set,
        # storage never copied to the host (§LXXX alias law holds on LavaArray)
        for layer in 1:2, kind in (:k, :v)
            ps = _pagesof_lava(s.mgr, layer, kind)
            pc = _pagesof_lava(c.mgr, layer, kind)
            @test length(pc) == length(ps) == 1
            @test pc[1] === ps[1]
            @test pc[1].storage isa Lava.LavaArray
            @test pc[1].shared == true
        end

        # child decodes: page 1 was FILLED by the prompt, so the child's first
        # decode takes the allocation branch (no CoW needed); the parent's
        # page must stay exactly as fork left it
        parent_before = [
            Array(_pagesof_lava(s.mgr, layer, kind)[1].storage) for
            layer in 1:2, kind in (:k, :v)
        ]
        child_ids = Int[]
        for _ in 1:4
            id = Gesso.decode!(c)
            push!(child_ids, id)
            id == c.eos_token_id && break
        end
        for layer in 1:2, kind in (:k, :v)
            ps = _pagesof_lava(s.mgr, layer, kind)[1]
            @test ps.shared == true                       # still shared, still frozen
            @test Array(ps.storage) == parent_before[layer, kind == :k ? 1 : 2]  # parent storage UNMOVED
            @test ps.filled == length(LAVA_PROMPT)        # parent page not appended to
        end

        # forked ids equal an INDEPENDENT device session's ids
        solo = _lava_session(model, gpu_ts; backend=lava)
        Gesso.prefill!(solo, LAVA_PROMPT)
        solo_ids = Int[]
        for _ in 1:4
            id = Gesso.decode!(solo)
            push!(solo_ids, id)
            id == solo.eos_token_id && break
        end
        @test child_ids == solo_ids

        # and the ids equal the CPU fork session's ids too (same oracle both sides)
        cpu_s = _lava_session(model, ts)
        Gesso.prefill!(cpu_s, LAVA_PROMPT)
        cpu_child = Gesso.fork(cpu_s)
        cpu_child_ids = Int[]
        for _ in 1:4
            id = Gesso.decode!(cpu_child)
            push!(cpu_child_ids, id)
            id == cpu_child.eos_token_id && break
        end
        @test child_ids == cpu_child_ids
    end

    @testset "Lava CoW: partial shared page is copied on child write, parent unmoved" begin
        # llama_micro with DEFAULT page_size=16: the 3-token prompt does NOT
        # fill page 1, so the child's first decode must CoW the partial page —
        # the exact path Phase 7 built, here on Vulkan storage
        dir = mktempdir()
        make_micro_checkpoint(dir)
        lm_model, lm_ts, lm_cfg = Gesso.load_llama(dir)
        ltokens = [0, 1, 2]
        lm_gpu_ts = Gesso.to_device(lava, lm_ts)

        lm_s = Gesso.Session(
            lm_model,
            lm_gpu_ts;
            backend=lava,
            context_length=128,
            eos_token_id=2,
            eps=lm_cfg.rms_norm_eps,
            theta=lm_cfg.rope_theta,
        )
        Gesso.prefill!(lm_s, ltokens)
        lm_c = Gesso.fork(lm_s)

        ps1 = _pagesof_lava(lm_s.mgr, 1, :k)[1]
        @test ps1.shared == true && ps1.filled == length(ltokens) < 16   # partial + shared
        parent_page_before = Array(ps1.storage)

        lm_child_ids = Int[]
        for _ in 1:3
            id = Gesso.decode!(lm_c)
            push!(lm_child_ids, id)
            id == lm_c.eos_token_id && break
        end

        pc1 = _pagesof_lava(lm_c.mgr, 1, :k)[1]
        @test pc1 !== ps1                                  # CoW: child got a NEW page
        @test pc1.shared == false                          # now privately owned
        @test pc1.filled == length(ltokens) + length(lm_child_ids)
        @test ps1.shared == true                           # parent still shared
        @test Array(ps1.storage) == parent_page_before     # parent storage UNMOVED
        # the copied page's prefix rows equal the parent's (values carried over;
        # page storage is (page_size, d_head, n_kv_heads) — slice whole rows)
        @test Array(pc1.storage)[1:length(ltokens), :, :] ==
              parent_page_before[1:length(ltokens), :, :]
        # and the forked ids equal an independent device session's ids
        lm_solo = Gesso.Session(
            lm_model,
            lm_gpu_ts;
            backend=lava,
            context_length=128,
            eos_token_id=2,
            eps=lm_cfg.rms_norm_eps,
            theta=lm_cfg.rope_theta,
        )
        lm_solo_ids = Gesso.generate(lm_solo, ltokens; max_new_tokens=length(lm_child_ids))
        @test lm_child_ids == lm_solo_ids[(length(ltokens)+1):end]
    end

    @testset "no-copy law: host Array tensors + Lava backend throws at construction" begin
        err = try
            Gesso.Session(model, ts; backend=lava, context_length=128, eos_token_id=2)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_INVALID_PLAN
        @test occursin("to_device", sprint(showerror, err))
        # the CPU tensors were not silently mutated by the attempt
        @test ts.embedding.storage isa Array{Float64}
    end
end
