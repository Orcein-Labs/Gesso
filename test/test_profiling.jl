# Phase 6 item B tests (§LXXIX): Profiling — memory accounting is a function
# of the page table, reports are structurally stable, an empty sink is an
# explicit empty report. Profiling must NOT import CUDA (checked structurally:
# the module's namespace carries no CUDA binding).

using .GessoTestHelpers: approx_eq

const PROMPT = [1, 3, 4, 5]

# distinct name: test_session_receipts.jl defines its own helper at the same
# scope (include order — no method-overwrite warning)
_prof_session(model, ts; kw...) =
    Gesso.Session(model, ts; context_length=128, eos_token_id=2, kw...)

@testset "Profiling (§LXXIX item B)" begin
    ts = toy2_tensors()
    model = toy2_modelir()

    @testset "kv_bytes(mgr) equals the documented formula after N appends" begin
        mgr = Gesso.Inference.PagedKVManager(
            Float64[];
            n_layers=2,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        row = fill(1.0, 2, 4)
        for i in 1:9        # crosses two page boundaries: 3 pages per (layer, kind)
            Gesso.Inference.append_kv!(mgr, 1, :k, row)
            Gesso.Inference.append_kv!(mgr, 1, :v, row)
        end        # formula: n_pages * page_size * n_kv_heads * d_head * sizeof(eltype)
        # (page_count already spans K and V — no extra factor). This test
        # appends to LAYER 1 only: ceil(9/4)=3 pages × (K,V) × 1 layer = 6.
        n_pages = Gesso.Profiling.page_footprint(mgr)
        @test n_pages == 3 * 2 * 1            # 3 pages × (K,V) × 1 layer
        @test Gesso.Profiling.kv_footprint(mgr) == n_pages * 4 * 2 * 4 * sizeof(Float64)
        # and it equals an actual sizeof walk of the live page storages
        direct = sum(
            sizeof(p.storage) for pages in (mgr.k_pages, mgr.v_pages) for lp in pages
            for p in lp
        )
        @test Gesso.Profiling.kv_footprint(mgr) == direct
    end

    @testset "page_size=4 session: kv_bytes still matches sizeof of live pages" begin
        sink = Gesso.InMemorySink()
        s = _prof_session(model, ts; sink=sink, page_size=4)
        Gesso.generate(s, PROMPT; max_new_tokens=8)
        r = only(sink.buf)
        @test r.memory_usage.kv_bytes == Gesso.Profiling.kv_footprint(s.mgr)
        direct = sum(
            sizeof(p.storage) for pages in (s.mgr.k_pages, s.mgr.v_pages) for
            lp in pages for p in lp
        )
        @test r.memory_usage.kv_bytes == direct
    end

    @testset "engine_report from a toy2 generate receipt: machine-readable keys" begin
        sink = Gesso.InMemorySink()
        s = _prof_session(model, ts; sink=sink)
        Gesso.generate(s, PROMPT; max_new_tokens=8)
        reports = Gesso.Profiling.engine_report(sink)
        @test length(reports) == 1
        rep = reports[1]
        for k in (:prefill_ns, :decode_ns, :ttft_ns, :kv_bytes, :kv_len)
            @test haskey(rep, k)
        end
        @test rep.prefill_ns isa UInt64
        @test rep.decode_ns isa UInt64
        @test rep.ttft_ns isa UInt64
        @test rep.kv_bytes isa Int
        @test rep.kv_len isa Int
        @test rep.task == :generate
        @test rep.failed == false
        @test rep.failure_code === nothing
    end

    @testset "two identical generates after warmup: same report STRUCTURE" begin
        sink = Gesso.InMemorySink()
        s = _prof_session(model, ts; sink=sink)
        Gesso.generate(s, PROMPT; max_new_tokens=4)   # warmup (compile)
        Gesso.generate(s, PROMPT; max_new_tokens=4)
        Gesso.generate(s, PROMPT; max_new_tokens=4)
        reports = Gesso.Profiling.engine_report(sink)
        @test length(reports) == 3
        r1, r2 = reports[2], reports[3]
        @test Set(keys(r1)) == Set(keys(r2))
        for k in keys(r1)
            @test typeof(getfield(r1, k)) == typeof(getfield(r2, k))
        end
    end

    @testset "empty sink: explicit empty report, not a crash" begin
        sink = Gesso.InMemorySink()
        reports = Gesso.Profiling.engine_report(sink)
        @test reports isa Vector
        @test isempty(reports)
    end

    @testset "failed call: report carries failed=true and the code" begin
        sink = Gesso.InMemorySink()
        s = _prof_session(model, ts; sink=sink)
        try
            Gesso.generate(s, Int[]; max_new_tokens=2)
        catch
            # the throw propagated; the receipt is what we audit
        end
        rep = only(Gesso.Profiling.engine_report(sink))
        @test rep.failed == true
        @test rep.failure_code == Gesso.ERR_INVALID_PLAN
    end

    @testset "print_report: fixed-order key=value line" begin
        sink = Gesso.InMemorySink()
        s = _prof_session(model, ts; sink=sink)
        Gesso.generate(s, PROMPT; max_new_tokens=2)
        r = only(sink.buf)
        text = sprint(Gesso.Profiling.print_report, r)
        @test occursin("task=generate", text)
        @test occursin("prefill_ns=", text)
        @test occursin("kv_bytes=", text)
        @test occursin("failed=false", text)
    end

    @testset "Profiling does not import CUDA" begin
        @test !isdefined(Gesso.Profiling, :CUDA)
        @test !(:CUDA in string.(names(Gesso.Profiling)))
    end
end
