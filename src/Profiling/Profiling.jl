# Profiling — performance observability (§XLIX, §L; Phase 6).
#
# Owns: stable, machine-readable attribution of what the engine already
# measured. §LXXIX exit for this sprint: prefill vs decode vs KV bytes are
# attributable per Session call, memory accounting is a FUNCTION OF THE PAGE
# TABLE (not an estimate), and reports are structurally stable across runs.
#
# This module does NOT diagnose "why it is slow" (§L: taxonomy is a label —
# the engine attaches context[:gap_class] on the receipt; the detective is
# Phase 9+). It does NOT import CUDA, does not time anything itself, and
# adds no dependencies. KV footprint helpers live next to the pages
# (Inference.kv_bytes / Inference.page_count — derived from the page table);
# this module renders them. §LXXX adds unique_kv_bytes: the DECLARED-share
# win metric — live storage counted once per distinct page array across a
# group of managers.

module Profiling

using ..Inference: PagedKVManager, kv_bytes, page_count
using ..Gesso: Receipt, ReceiptSink, InMemorySink, GessoError

export engine_report, print_report, kv_footprint, page_footprint, unique_kv_bytes

"""
    kv_footprint(mgr) -> Int

KV cache footprint in bytes — the page-table-derived number
(Inference.kv_bytes): sum of sizeof over every allocated page storage,
K and V, all layers, including unused rows of live pages. Trustworthy means
a test can reconstruct it from the manager; test_session_receipts.jl and
test_profiling.jl both do.
"""
kv_footprint(mgr::PagedKVManager) = kv_bytes(mgr)

"""
    page_footprint(mgr) -> Int

Allocated page count across all layers, K and V combined.
"""
page_footprint(mgr::PagedKVManager) = page_count(mgr)

"""
    unique_kv_bytes(mgrs::PagedKVManager...) -> Int

Live KV storage of a GROUP of managers, counted ONCE per distinct page
storage array (§LXXX item C — the §LIV win metric): `sizeof` summed over the
distinct `objectid(page.storage)` across every manager's pages, K and V, all
layers.

Per-session `kv_bytes` stays honest per-session accounting (§LXXX: shared
pages appear in BOTH sessions' receipts). The win is visible HERE: after
`prefill!` + `fork` (before any decode) `unique_kv_bytes(parent.mgr,
child.mgr) == kv_bytes(parent.mgr)` — the aliased prefix is one array, not
two — and the value stays below the per-session sum for as long as any page
remains aliased. Two managers built INDEPENDENTLY share no storage
(declaration, not discovery — Session `fork` is the only share constructor),
so the function degenerates to the sum of their `kv_bytes`.
"""
function unique_kv_bytes(mgrs::PagedKVManager...)
    seen = Set{UInt}()
    total = 0
    for mgr in mgrs
        for pages in (mgr.k_pages, mgr.v_pages), layer_pages in pages, p in layer_pages

            id = objectid(p.storage)
            id in seen && continue
            push!(seen, id)
            total += sizeof(p.storage)
        end
    end
    return total
end

# the machine-readable projection of an engine receipt: fixed keys, fixed
# types (UInt64 ns timings, Int counts) — two identical generates produce
# structurally identical reports; only the VALUES differ. `nothing` for a
# field the receipt does not carry is explicit, never a missing key.
function _project(r::Receipt)
    t = r.timing isa NamedTuple ? r.timing : NamedTuple()
    m = r.memory_usage isa NamedTuple ? r.memory_usage : NamedTuple()
    tu = r.token_usage isa NamedTuple ? r.token_usage : NamedTuple()
    return (
        task=r.task,
        prefill_ns=get(t, :prefill_ns, nothing),
        decode_ns=get(t, :decode_ns, nothing),
        ttft_ns=get(t, :ttft_ns, nothing),
        total_ns=get(t, :total_ns, nothing),
        ttft_ms=get(t, :ttft_ns, nothing) === nothing ? nothing :
                Float64(get(t, :ttft_ns, nothing)) / 1.0e6,
        prefill_ms=get(t, :prefill_ns, nothing) === nothing ? nothing :
                   Float64(get(t, :prefill_ns, nothing)) / 1.0e6,
        decode_ms=get(t, :decode_ns, nothing) === nothing ? nothing :
                  Float64(get(t, :decode_ns, nothing)) / 1.0e6,
        total_ms=get(t, :total_ns, nothing) === nothing ? nothing :
                 Float64(get(t, :total_ns, nothing)) / 1.0e6,
        prompt_tokens=get(tu, :prompt_tokens, nothing),
        new_tokens=get(tu, :new_tokens, nothing),
        decode_tokens_per_s=begin
            dns = get(t, :decode_ns, nothing)
            ntok = get(tu, :new_tokens, nothing)
            dns === nothing || ntok === nothing || dns == 0 || ntok == 0 ?
                nothing : Float64(ntok) * 1.0e9 / Float64(dns)
        end,
        decode_ms_per_token=begin
            dns = get(t, :decode_ns, nothing)
            ntok = get(tu, :new_tokens, nothing)
            dns === nothing || ntok === nothing || ntok == 0 ?
                nothing : Float64(dns) / 1.0e6 / Float64(ntok)
        end,
        total_tokens=get(tu, :total_tokens, nothing),
        kv_bytes=get(m, :kv_bytes, nothing),
        page_count=get(m, :page_count, nothing),
        kv_len=get(m, :kv_len, nothing),
        context_length=get(m, :context_length, nothing),
        context_remaining=get(m, :context_remaining, nothing),
        failed=r.failure !== nothing,
        failure_code=r.failure isa GessoError ? r.failure.code : nothing,
    )
end

"""
    engine_report(r::Receipt) -> NamedTuple

Machine-readable attribution for one engine receipt: fixed keys
(`prefill_ns`, `decode_ns`, `ttft_ns`, `kv_bytes`, `kv_len`, …), fixed
types. Structure is stable across runs; values are whatever the call
measured. A failed call reports `failed = true` and its code.
"""
engine_report(r::Receipt) = _project(r)

"""
    engine_report(sink) -> Vector{NamedTuple}

Reports for every receipt in a sink, in insertion order. An EMPTY sink
yields an empty vector — explicit, not a crash.
"""
function engine_report(sink::InMemorySink)
    return [_project(r) for r in sink.buf]
end

"""
    print_report(io, r::Receipt)

Stable text rendering of one engine receipt: one fixed-order line, one
`key=value` field per projection key. Nothing here is parsed by tests that
want numbers — they use `engine_report`; this is the human surface.
"""
function print_report(io::IO, r::Receipt)
    p = _project(r)
    print(io, "gesso.engine task=", p.task)
    print(io, " prefill_ns=", p.prefill_ns)
    print(io, " decode_ns=", p.decode_ns)
    print(io, " ttft_ns=", p.ttft_ns)
    print(io, " total_ns=", p.total_ns)
    print(io, " ttft_ms=", p.ttft_ms)
    print(io, " decode_tok_s=", p.decode_tokens_per_s)
    print(io, " decode_ms_per_token=", p.decode_ms_per_token)
    print(io, " prompt_tokens=", p.prompt_tokens)
    print(io, " new_tokens=", p.new_tokens)
    print(io, " kv_bytes=", p.kv_bytes)
    print(io, " pages=", p.page_count)
    print(io, " kv_len=", p.kv_len)
    print(io, "/", p.context_length)
    print(io, " failed=", p.failed)
    p.failure_code === nothing || print(io, " code=", p.failure_code)
    return nothing
end

print_report(r::Receipt) = print_report(stdout, r)

end
