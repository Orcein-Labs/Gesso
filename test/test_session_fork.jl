# Phase 7 item B tests (§LXXX): Session `fork` — DECLARED identity prefix
# share (Magenta §9.5 step 3).
#
# Gates (docs/goals/PHASE7_PREFIX_SHARE.md):
#   * fork before prefill! is ERR_INVALID_PLAN (sharing needs a ready prefix)
#   * after prefill!, the child aliases the prefix pages (`===`, shared=true)
#     and carries copied h/seqlen/ready with its own manager and sink
#   * parent + child interleaved decode ids equal INDEPENDENT sessions that
#     saw the same token sequences — a forked child is not a new oracle; the
#     oracle path is untouched
#   * a full shared page is never written: parent and child keep page 1
#     `===` forever, each allocating a private page past the boundary
#   * generate on a forked child still RESETS — and so DROPS the share
#   * fork emits NO receipt; the child's engine calls emit into the child's
#     own sink
#   * llama_micro: same id-equality vs an independent session
#   * CUDA: named skip without a device; with a device, forked ids equal an
#     independent CUDA session's ids and the prefix stays aliased on device

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const FORK_PROMPT = [1, 3, 4, 5]

# helper name avoids `_session` (test_session.jl / test_session_receipts.jl
# overwrite it at file scope). page_size=4 so the 4-token prompt fills page 1
# exactly — the decode side must allocate NEW pages while the full shared
# prefix page stays aliased forever (§LXXX).
_fork_session(model, ts; kw...) =
    Gesso.Session(model, ts; page_size=4, context_length=128, eos_token_id=2, kw...)

# the page table of one (layer, kind) cache
_pagesof(mgr, layer, kind) = Gesso.Inference.kv_cache(mgr, layer, kind).storage

@testset "Session fork (§LXXX item B, Magenta §9.5 step 3)" begin
    ts = toy2_tensors()
    model = toy2_modelir()

    @testset "declaration, not discovery: independent sessions never share" begin
        sa = _fork_session(model, ts)
        sb = _fork_session(model, ts)
        Gesso.prefill!(sa, FORK_PROMPT)
        Gesso.prefill!(sb, FORK_PROMPT)
        for layer in 1:2, kind in (:k, :v)
            pa = _pagesof(sa.mgr, layer, kind)
            pb = _pagesof(sb.mgr, layer, kind)
            @test length(pa) == length(pb) == 1
            @test pa[1].storage !== pb[1].storage
            @test pa[1].shared == false && pb[1].shared == false
        end
    end

    @testset "fork before prefill! is ERR_INVALID_PLAN" begin
        err = try
            Gesso.fork(_fork_session(model, ts))
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        @test occursin("not ready", sprint(showerror, err))
    end

    @testset "after prefill!: aliasing + copied state + own sink/manager" begin
        s = _fork_session(model, ts)
        Gesso.prefill!(s, FORK_PROMPT)
        c = Gesso.fork(s; sink=Gesso.InMemorySink())

        @test c.seqlen == s.seqlen == length(FORK_PROMPT)
        @test c.ready && s.ready
        @test c.model === s.model
        @test c.tensors === s.tensors
        @test c.backend == s.backend
        @test c.page_size == s.page_size == 4
        @test c.context_length == s.context_length
        @test c.eos_token_id == s.eos_token_id
        @test c.tokenizer === s.tokenizer
        @test c.eps == s.eps && c.theta == s.theta
        # own sink (injected through fork) and own manager; fork emitted
        # nothing into it (no receipt for a declaration)
        @test c.sink !== s.sink
        @test c.sink isa Gesso.InMemorySink
        @test isempty(c.sink.buf)
        @test c.mgr !== s.mgr
        # hidden state is COPIED, not aliased (§LXXX: no hidden-state CoW)
        @test c.h !== s.h
        @test c.h == s.h

        for layer in 1:2, kind in (:k, :v)
            ps = _pagesof(s.mgr, layer, kind)
            pc = _pagesof(c.mgr, layer, kind)
            @test length(pc) == length(ps) == 1
            @test pc[1] === ps[1]                      # same KVPage OBJECT
            @test pc[1].storage === ps[1].storage
            @test pc[1].shared == true
        end
    end

    @testset "parent+child interleaved decode == independent sessions" begin
        s = _fork_session(model, ts)
        Gesso.prefill!(s, FORK_PROMPT)
        c = Gesso.fork(s)

        parent_ids = Int[]
        child_ids = Int[]
        for _ in 1:8
            pid = Gesso.decode!(s)
            push!(parent_ids, pid)
            cid = Gesso.decode!(c)
            push!(child_ids, cid)
            (pid == s.eos_token_id || cid == c.eos_token_id) && break
        end

        solo_p = _fork_session(model, ts)
        Gesso.prefill!(solo_p, FORK_PROMPT)
        solo_p_ids = Int[]
        for _ in 1:8
            id = Gesso.decode!(solo_p)
            push!(solo_p_ids, id)
            id == solo_p.eos_token_id && break
        end

        solo_c = _fork_session(model, ts)
        Gesso.prefill!(solo_c, FORK_PROMPT)
        solo_c_ids = Int[]
        for _ in 1:8
            id = Gesso.decode!(solo_c)
            push!(solo_c_ids, id)
            id == solo_c.eos_token_id && break
        end

        @test parent_ids == solo_p_ids               # parent continues solo
        @test child_ids == solo_c_ids                # child == independent session
        @test solo_p_ids == solo_c_ids               # determinism pin

        # a FULL shared page is never written: page 1 (filled by the prompt,
        # page_size=4) is still the same object in parent and child, while
        # every page past the boundary is that side's private allocation
        for layer in 1:2, kind in (:k, :v)
            ps = _pagesof(s.mgr, layer, kind)
            pc = _pagesof(c.mgr, layer, kind)
            @test length(ps) == length(pc) == 1 + cld(length(parent_ids), 4)
            @test pc[1] === ps[1]
            @test pc[1].shared == true
            for i in 2:length(ps)
                @test pc[i] !== ps[i]
                @test pc[i].shared == false && ps[i].shared == false
            end
        end
    end

    @testset "generate on a forked child RESETS (drops the share)" begin
        sink_c = Gesso.InMemorySink()
        s = _fork_session(model, ts)
        Gesso.prefill!(s, FORK_PROMPT)
        c = Gesso.fork(s; sink=sink_c)

        gen_ids = Gesso.generate(c, FORK_PROMPT; max_new_tokens=8)
        @test gen_ids == Gesso.reference_generate(model, ts, FORK_PROMPT; max_new_tokens=8)
        # generate still emits exactly ONE receipt (Phase 5/6 laws unchanged)
        @test length(sink_c.buf) == 1
        @test sink_c.buf[end].task == :generate

        for layer in 1:2, kind in (:k, :v)
            ps = _pagesof(s.mgr, layer, kind)
            pc = _pagesof(c.mgr, layer, kind)
            @test pc[end].storage !== ps[end].storage   # fresh pages after reset+prefill
            @test pc[end].shared == false
            @test ps[end].shared == true                # original never un-shared
        end

        # Phase 5 laws intact on the child: empty prompt throws AND emits a
        # failure receipt into the child's sink
        err = try
            Gesso.generate(c, Int[]; max_new_tokens=2)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        @test length(sink_c.buf) == 2
        @test sink_c.buf[end].task == :generate
        @test sink_c.buf[end].failure !== nothing
    end

    @testset "fork emits NO receipt; child calls emit into the child's sink" begin
        fork_sink = Gesso.InMemorySink()
        s2 = _fork_session(model, ts; sink=fork_sink)
        Gesso.prefill!(s2, FORK_PROMPT)
        @test length(fork_sink.buf) == 1
        @test fork_sink.buf[end].task == :prefill

        c3 = Gesso.fork(s2; sink=fork_sink)
        @test length(fork_sink.buf) == 1        # fork is a declaration, not a step
        @test c3.sink === fork_sink

        Gesso.decode!(c3)
        @test length(fork_sink.buf) == 2        # child decode → the CHILD's sink
        @test fork_sink.buf[end].task == :decode
        @test fork_sink.buf[end].token_usage.new_tokens == 1

        # a fork WITHOUT sink= hands the child the process default sink —
        # never the parent's sink object
        c4 = Gesso.fork(s2)
        @test c4.sink !== s2.sink
        @test c4.sink === Gesso.default_receipt_sink()
    end

    @testset "empty prompt / context limit still throw" begin
        s5 = _fork_session(model, ts)
        err = try
            Gesso.prefill!(s5, Int[])
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN

        s6 = Gesso.Session(model, ts; page_size=4, context_length=3, eos_token_id=2)
        err2 = try
            Gesso.prefill!(s6, FORK_PROMPT)      # 4-token prompt > context_length 3
            nothing
        catch e
            e
        end
        @test err2 isa Gesso.GessoError && err2.code == Gesso.ERR_RESOURCE_LIMIT
        # a session whose prefill threw is not ready → fork refuses it too
        err3 = try
            Gesso.fork(s6)
            nothing
        catch e
            e
        end
        @test err3 isa Gesso.GessoError && err3.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "llama_micro: forked decode ids == an independent session" begin
        dir = mktempdir()
        make_micro_checkpoint(dir)
        lm, lts, lcfg = Gesso.load_llama(dir)
        ltokens = [0, 1, 2]

        _lm_session(; kw...) = Gesso.Session(
            lm,
            lts;
            context_length=32,
            eos_token_id=0,
            eps=lcfg.rms_norm_eps,
            theta=lcfg.rope_theta,
            kw...,
        )

        lm_s = _lm_session()
        Gesso.prefill!(lm_s, ltokens)
        lm_c = Gesso.fork(lm_s)
        # alias pin: every prefix page is the same object, marked shared
        for layer in 1:2, kind in (:k, :v)
            ps = _pagesof(lm_s.mgr, layer, kind)
            pc = _pagesof(lm_c.mgr, layer, kind)
            @test length(pc) == length(ps) == 1
            @test pc[1] === ps[1]
            @test pc[1].shared == true
        end

        lm_child_ids = Int[]
        for _ in 1:3
            id = Gesso.decode!(lm_c)
            push!(lm_child_ids, id)
            id == lm_c.eos_token_id && break
        end
        lm_solo = _lm_session()
        Gesso.prefill!(lm_solo, ltokens)
        lm_solo_ids = Int[]
        for _ in 1:3
            id = Gesso.decode!(lm_solo)
            push!(lm_solo_ids, id)
            id == lm_solo.eos_token_id && break
        end
        @test lm_child_ids == lm_solo_ids

        # the aliased prefix stayed BIT-IDENTICAL in the parent while the
        # child and the solo session kept decoding past it (solo is 3 rows
        # longer; compare the prefix rows only)
        @test isequal(
            Gesso.Inference.gather_kv(lm_s.mgr, 1, :k),
            Gesso.Inference.gather_kv(lm_solo.mgr, 1, :k; len=length(ltokens)),
        )
        # the prompt does NOT fill llama_micro's first page (default
        # page_size=16): the child's first decode COPIES that partial shared
        # page (CoW of the dirty page), so the child's page 1 is a NEW
        # unshared object; the parent's original is frozen, still shared
        ps1 = _pagesof(lm_s.mgr, 1, :k)[1]
        pc1 = _pagesof(lm_c.mgr, 1, :k)[1]
        @test pc1 !== ps1
        @test pc1.shared == false && ps1.shared == true
        @test pc1.filled == length(ltokens) + length(lm_child_ids)
        @test ps1.filled == length(ltokens)
        @test isequal(
            Array(Gesso.Inference.gather_kv(lm_s.mgr, 1, :k)),
            Array(Gesso.Inference.gather_kv(lm_c.mgr, 1, :k; len=length(ltokens))),
        )
    end

    @testset "CUDA: forked ids == independent session, prefix aliased on device" begin
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
                "no NVIDIA device (CUDA.functional() == false) — fork CUDA test skipped (§LXXVIII skip law)",
            )
        else
            cuda = Gesso.CUDABackend()
            gpu_ts = Gesso.to_device(cuda, ts)
            gs = Gesso.Session(
                model,
                gpu_ts;
                backend=cuda,
                page_size=4,
                context_length=128,
                eos_token_id=2,
            )
            Gesso.prefill!(gs, FORK_PROMPT)
            gc = Gesso.fork(gs)

            # the prefix is aliased ON DEVICE: same page objects, shared flag
            # set, storage never copied to the host (§LXXVII)
            for layer in 1:2, kind in (:k, :v)
                ps = _pagesof(gs.mgr, layer, kind)
                pc = _pagesof(gc.mgr, layer, kind)
                @test length(pc) == length(ps) == 1
                @test pc[1] === ps[1]
                @test pc[1].storage isa CUDA.CuArray
                @test pc[1].shared == true
            end

            gpu_child_ids = Int[]
            for _ in 1:4
                id = Gesso.decode!(gc)
                push!(gpu_child_ids, id)
                id == gc.eos_token_id && break
            end
            solo_gpu = Gesso.Session(
                model,
                gpu_ts;
                backend=cuda,
                page_size=4,
                context_length=128,
                eos_token_id=2,
            )
            Gesso.prefill!(solo_gpu, FORK_PROMPT)
            solo_gpu_ids = Int[]
            for _ in 1:4
                id = Gesso.decode!(solo_gpu)
                push!(solo_gpu_ids, id)
                id == solo_gpu.eos_token_id && break
            end
            @test gpu_child_ids == solo_gpu_ids

            # CoW allocated on device (similar off device storage): the child
            # has a private second page; the full prefix page stayed ===
            for layer in 1:2, kind in (:k, :v)
                ps = _pagesof(gs.mgr, layer, kind)
                pc = _pagesof(gc.mgr, layer, kind)
                @test length(ps) == 1
                @test length(pc) == 2
                @test pc[1] === ps[1]
                @test pc[2].shared == false
            end
        end
    end
end
