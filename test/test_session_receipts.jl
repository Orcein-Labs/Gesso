# Phase 6 item A tests (§LXXIX): the engine is auditable — every
# generate / prefill! / decode! leaves ONE receipt whose §XLII fields are
# present and internally consistent. Oracle ids are re-checked here: if ids
# drift once receipts are on the path, that is a packet-level escalation.
#
# §XXXIII hygiene: these tests assert STRUCTURE and CONSISTENCY (totals,
# sums, identities), never absolute ns values — clocks are not the
# fingerprint, and structure tests may include compile.

using .GessoTestHelpers: approx_eq

const PROMPT = [1, 3, 4, 5]

_session(model, ts; kw...) =
    Gesso.Session(model, ts; context_length=128, eos_token_id=2, kw...)

@testset "Session receipts (§LXXIX item A)" begin
    ts = toy2_tensors()
    model = toy2_modelir()

    @testset "generate ids still equal the oracle (receipts on the path)" begin
        sink = Gesso.InMemorySink()
        s = _session(model, ts; sink=sink)
        ids = Gesso.generate(s, PROMPT; max_new_tokens=8)
        @test ids == Gesso.reference_generate(model, ts, PROMPT; max_new_tokens=8)
        # exactly one receipt for the whole generate (impl calls are internal)
        @test length(sink.buf) == 1
    end

    @testset "receipt fields: timing consistency + token identity" begin
        sink = Gesso.InMemorySink()
        s = _session(model, ts; sink=sink)
        ids = Gesso.generate(s, PROMPT; max_new_tokens=8)
        r = only(sink.buf)
        @test r isa Gesso.Receipt
        @test r.task == :generate
        t = r.timing
        @test t.prefill_ns isa UInt64 && t.decode_ns isa UInt64
        @test t.total_ns isa UInt64 && t.ttft_ns isa UInt64
        # total ≈ prefill + decode (1% or 1e6 ns tolerance — clocks are not exact)
        slack = max(0.01 * t.total_ns, 1_000_000.0)
        @test abs(Float64(t.total_ns) - Float64(t.prefill_ns + t.decode_ns)) <= slack
        # TTFT = prefill + first decode step (new_tokens > 0 here)
        @test t.ttft_ns >= t.prefill_ns
        tu = r.token_usage
        @test tu.prompt_tokens == 4
        @test tu.new_tokens == length(ids) - 4
        @test tu.total_tokens == tu.prompt_tokens + tu.new_tokens
        ir = r.inference_request
        @test ir.backend == :cpu
        @test ir.page_size == 16
        @test ir.max_new_tokens == 8
        @test ir.eos_token_id == 2
    end

    @testset "receipt memory_usage: KV accounting from the page table" begin
        sink = Gesso.InMemorySink()
        s = _session(model, ts; sink=sink, page_size=4)
        ids = Gesso.generate(s, PROMPT; max_new_tokens=8)
        r = only(sink.buf)
        m = r.memory_usage
        kv_len = Gesso.Inference.kv_len(s.mgr)
        # EOS is returned but NOT consumed — a trailing EOS is not in the KV
        expected_len = length(ids) - (ids[end] == 2 ? 1 : 0)
        @test m.kv_len == kv_len == s.seqlen == expected_len
        @test m.context_length == 128
        @test m.context_remaining == 128 - kv_len
        # page_count spans K AND V, all layers
        @test m.page_count ==
              length(Gesso.Inference.kv_cache(s.mgr, 1, :k).storage) * 2 * 2
        # kv_bytes derived from the manager matches a direct sizeof walk
        direct = sum(
            sizeof(p.storage) for pages in (s.mgr.k_pages, s.mgr.v_pages) for lp in pages for
            p in lp
        )
        @test m.kv_bytes == direct
        # documented formula (no extra K+V factor — page_count already spans both)
        @test m.kv_bytes == m.page_count * 4 * 2 * 8 * sizeof(Float64)
    end

    @testset "prefill! and decode! each emit one receipt" begin
        sink = Gesso.InMemorySink()
        s = _session(model, ts; sink=sink)
        Gesso.prefill!(s, PROMPT)
        @test length(sink.buf) == 1
        @test sink.buf[end].task == :prefill
        id1 = Gesso.decode!(s)
        @test length(sink.buf) == 2
        @test sink.buf[end].task == :decode
        @test sink.buf[end].token_usage.new_tokens == 1
        @test sink.buf[end].timing.prefill_ns == UInt64(0)
    end

    @testset "empty prompt: failure receipt + the throw still happens" begin
        sink = Gesso.InMemorySink()
        s = _session(model, ts; sink=sink)
        err = try
            Gesso.generate(s, Int[]; max_new_tokens=2)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        @test length(sink.buf) == 1
        r = only(sink.buf)
        @test r.failure isa Gesso.GessoError
        @test r.failure.code == Gesso.ERR_INVALID_PLAN
        @test r.failure === err                      # the constructed error itself
    end

    @testset "context exhaustion: failure receipt + ERR_RESOURCE_LIMIT propagates" begin
        sink = Gesso.InMemorySink()
        s = Gesso.Session(model, ts; context_length=6, eos_token_id=2, sink=sink)
        err = try
            Gesso.generate(s, PROMPT; max_new_tokens=8)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_RESOURCE_LIMIT
        @test length(sink.buf) == 1
        r = only(sink.buf)
        @test r.failure isa Gesso.GessoError
        @test r.failure.code == Gesso.ERR_RESOURCE_LIMIT
    end

    @testset "two identical generates: same receipt STRUCTURE (keys/types)" begin
        sink = Gesso.InMemorySink()
        s = _session(model, ts; sink=sink)
        Gesso.generate(s, PROMPT; max_new_tokens=4)
        Gesso.generate(s, PROMPT; max_new_tokens=4)
        @test length(sink.buf) == 2
        r1, r2 = sink.buf
        @test Set(keys(r1.timing)) == Set(keys(r2.timing))
        @test Set(keys(r1.token_usage)) == Set(keys(r2.token_usage))
        @test Set(keys(r1.memory_usage)) == Set(keys(r2.memory_usage))
        @test typeof(r1.timing.prefill_ns) == typeof(r2.timing.prefill_ns)
        @test r1.task == r2.task == :generate
    end

    @testset "default sink is the process-level InMemorySink" begin
        s = _session(model, ts)
        @test s.sink === Gesso.default_receipt_sink()
        before = length(Gesso.default_receipt_sink().buf)
        Gesso.generate(s, PROMPT; max_new_tokens=2)
        @test length(Gesso.default_receipt_sink().buf) == before + 1
    end
end
