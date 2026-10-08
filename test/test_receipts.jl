# Receipt tests (§XLII): construction, schema stability, bounded ring, and
# the no-throw delivery guarantee (a receipt path must never become a second
# failure amplifier).
#
# Split from test_foundation.jl per the per-area harness convention
# (test/runtests.jl includes per-area files).

using Dates

@testset "receipts (§XLII): construction + schema stability" begin
    # Schema stability: the §XLII field vocabulary is a persisted schema
    # (RECEIPT_SCHEMA_VERSION). Changing its fields without bumping the
    # version is a §LXIX violation — this list is the tripwire.
    fields = sort(collect(fieldnames(Gesso.Receipt)))
    @test fields == sort([
        :agent,
        :cancellation,
        :context,
        :failure,
        :id,
        :inference_request,
        :materialization,
        :memory_usage,
        :model,
        :output_digest,
        :parent_dependency,
        :retry,
        :schema_version,
        :task,
        :timestamp,
        :timing,
        :token_usage,
        :tool_request,
        :tool_result,
    ])

    # Deterministic field presence: every §XLII slot exists and defaults to
    # nothing (nothing = "not applicable to this action"); ids are fresh and
    # the timestamp is UTC.
    before = Gesso.next_receipt_id()
    t0 = Dates.now(Dates.UTC)
    r = Gesso.new_receipt(task="audit me")
    t1 = Dates.now(Dates.UTC)
    @test r.id > before
    @test r.schema_version == Gesso.RECEIPT_SCHEMA_VERSION
    @test r.task == "audit me"
    # the t0/t1 window is against Dates.now(Dates.UTC): a wall-clock in any
    # other zone would fall outside it — this IS the UTC assertion (DateTime
    # itself is zone-less; there is no .timezone field to inspect)
    @test t0 <= r.timestamp <= t1
    for f in setdiff(fields, [:id, :timestamp, :schema_version, :task, :context])
        @test getproperty(r, f) === nothing
    end
    @test isempty(r.context)
end

@testset "receipts (§XLII): bounded ring saturation" begin
    sink = Gesso.InMemorySink(3)
    rs = [Gesso.new_receipt(task=i) for i in 1:7]
    foreach(r -> Gesso.emit!(sink, r), rs)

    # after saturation the sink is a window over the LAST capacity receipts,
    # oldest-first, and dropped counts exactly what overflowed
    @test length(sink.buf) == 3
    @test [x.task for x in sink.buf] == [5, 6, 7]
    @test sink.dropped == UInt64(4)

    # exactly-at-capacity and below: nothing dropped
    sink2 = Gesso.InMemorySink(2)
    foreach(r -> Gesso.emit!(sink2, r), rs[1:2])
    @test length(sink2.buf) == 2
    @test sink2.dropped == UInt64(0)
end

@testset "receipts (§XLII): no-throw delivery guarantee" begin
    # A telemetry path must never become a second failure amplifier: whatever
    # the sink or logging does, emit! returns the receipt it was given.

    # Partial/malformed EVENTS (hostile payloads in the permissive §XLII Any
    # fields) still land: emit! is payload-agnostic. Typed fields (id,
    # timestamp, schema_version) make malformed identity unrepresentable by
    # construction.
    sink = Gesso.InMemorySink(2)
    bad = Gesso.new_receipt(
        task=Dict(:nested => :junk),
        failure="not a GessoError",   # wrong type for the slot
        timing=123,                   # not a timing record
        context=Dict(:k => BigFloat(π)),
    )
    @test Gesso.emit!(sink, bad) === bad
    @test sink.buf[end] === bad

    # Nested failure: even with LOGGING ITSELF broken (config io replaced by
    # a throwing pseudo-IO), delivery stays no-throw and the receipt is still
    # stored. Overflow forces the WARN log, whose failure forces the ERROR
    # log — the full catch path is exercised.
    struct ExplodingIO <: IO end
    Base.write(io::ExplodingIO, args...) = error("log I/O is broken")

    sink = Gesso.InMemorySink(4)
    cfg = Gesso.current_config()
    old_io, old_level = cfg.io, cfg.min_level
    try
        cfg.io = ExplodingIO()
        cfg.min_level = Gesso.Log.LOG_DEBUG  # force the glog calls to format
        rs = [Gesso.new_receipt(task=i) for i in 1:6]
        @test Gesso.emit!.(Ref(sink), rs) == rs  # no throw despite broken logger
        @test length(sink.buf) == 4    # push-first ordering survived
        @test sink.dropped == UInt64(2)
    finally
        cfg.io, cfg.min_level = old_io, old_level
    end
end

@testset "receipts (§XLII): concurrency" begin
    # concurrent emitters cannot lose receipts or corrupt the ring; the
    # dropped counter is exact (stored == emitted - dropped)
    sink = Gesso.InMemorySink(256)
    per_thread = 2_000
    @sync begin
        for t in 1:4
            Threads.@spawn begin
                for i in 1:per_thread
                    Gesso.emit!(sink, Gesso.new_receipt(task="t$t-$i"))
                end
            end
        end
    end
    @test length(sink.buf) == 256
    @test sink.dropped == UInt64(4 * per_thread - 256)
    @test all(r -> r isa Gesso.Receipt, sink.buf)

    # ids are atomic: strictly monotonic across threads (duplicates would
    # break parent_dependency references — §XLII)
    ids = Vector{Vector{UInt64}}(undef, 4)
    @sync begin
        for t in 1:4
            Threads.@spawn begin
                local seen = UInt64[]
                for i in 1:1_000
                    push!(seen, Gesso.next_receipt_id())
                end
                ids[t] = seen
            end
        end
    end
    flat = sort(vcat(ids...))
    @test allunique(flat)
end

@testset "Bounded receipt storage stays stable across overflow" begin
    sink=Gesso.InMemorySink(4)
    before=pointer(sink.buf)
    records=[Gesso.new_receipt(; task=:overflow_probe) for _ in 1:20]
    for r in records
        Gesso.emit!(sink, r)
    end
    @test [r.id for r in sink.buf]==[r.id for r in records[(end-3):end]]
    @test sink.dropped==16
    @test pointer(sink.buf)==before
end
