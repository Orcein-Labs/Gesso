# Phase 5 (§LXXVIII item A): paged KV manager — Magenta §9.5 step 1, nothing
# further (docs/research/KV_MEMORY_PROGRAM.md; §XXXI: KV is a semantic object).
#
# Physical unit: a PAGE of `page_size` token-rows, shape
# (page_size, n_kv_heads, d_head), allocated with `similar` off the session's
# tensor storage (CPU Array{Float64}, CUDA CuArray{Float32} — §LXXVII
# discipline). The logical cache per (layer, kind) is an ordered list of pages;
# append writes the NEXT row into the last page (or allocates exactly one more
# page when it is full); the whole cache is never realloc-copied.
#
# OWNERSHIP OF TRUTH (gate-tested): the filled length of each cache is DERIVED
# from its pages (Σ page.filled) — the manager keeps no shadow counters, so a
# cache's reported shape can never drift from its storage. The SEQUENCE length
# (one counter for the whole token stream, §XXXI "sequence position") is the
# SESSION's concern: the engine advances every (layer, kind) in lockstep, one
# row per token, and reports progress itself. At the manager level `kv_len` is
# the derived filled length of the canonical cache (layer 1, K side) — under
# engine lockstep all caches agree with it; per-cache counts are available via
# `filled_len(mgr, layer, kind)`. Tests that append to a single kind/layer
# exercise the manager in isolation; they see exactly what they wrote.
#
# DESIGN DECISIONS (goal-mandated picks, pinned by test_kv_manager.jl):
#   * KVCache.shape is the LOGICAL FILLED shape (filled, n_kv_heads, d_head),
#     derived from the pages. The fixed-capacity alternative was rejected:
#     filled length is the quantity every consumer wants.
#   * K and V are independent logical caches (Magenta: "K and V separately");
#     the engine appends them as a pair per token via the paired `append_kv!`.
#   * Pages are the cache; reads GATHER into contiguous scratch. There is no
#     second contiguous buffer to keep coherent (locked decision, 2026-09-30).
#
# The manager owns storage and append. It does NOT own a kernel (§LXXVIII):
# attention this sprint reuses the existing score/softmax/value contraction
# over the gathered scratch.
#
# `KVCache` is the SEMANTIC IDENTITY of a layer's K or V (§CIX family type,
# unused until this phase). The page table lives in its `storage` field — the
# type gains NO new fields (a packet would be required; it is not). Each page
# records provenance, per Magenta §12.2, even though nothing consumes it yet:
#
#     layer::Int      1-based layer index
#     kind::Symbol    :k or :v
#     start_pos::Int  0-based first token-row this page covers
#     filled::Int     rows written in this page, 0 ≤ filled ≤ page_size
#
# That provenance is the hook for Magenta §9.5 steps 2–7 (CoW, prefix share,
# span classes, tiering, eviction) — NONE of which exist here. No span
# classes, no CoW, no eviction (Phase 7+).
#
# Exceeding `context_length` on a cache throws typed ERR_RESOURCE_LIMIT: no
# silent drop, no wraparound (§LXX).

mutable struct KVPage
    layer::Int
    kind::Symbol
    start_pos::Int
    filled::Int
    storage::Any                       # (page_size, n_kv_heads, d_head)
end

mutable struct PagedKVManager
    n_layers::Int
    n_kv_heads::Int
    d_head::Int
    page_size::Int
    context_length::Int
    k_pages::Vector{Vector{KVPage}}    # [layer] => ordered pages
    v_pages::Vector{Vector{KVPage}}
    prototype::Any                     # allocation source (similar), §LXXVII
end

_kind_str(kind::Symbol) = kind === :k ? "K" : kind === :v ? "V" : string(kind)

function _pages(mgr::PagedKVManager, layer::Int, kind::Symbol)
    1 <= layer <= mgr.n_layers || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "paged KV: layer $layer out of range 1:$(mgr.n_layers)";
            layer=layer,
        ),
    )
    kind === :k ||
        kind === :v ||
        throw(
            gesso_error(
                ERR_INVALID_PLAN,
                "paged KV: kind must be :k or :v (got $kind)";
                kind=kind,
            ),
        )
    return kind === :k ? mgr.k_pages[layer] : mgr.v_pages[layer]
end

# the one source of truth for a cache's filled length: its pages
_pages_filled(pages::Vector{KVPage}) =
    isempty(pages) ? 0 : pages[end].start_pos + pages[end].filled

"""
    PagedKVManager(prototype; n_layers, n_kv_heads, d_head, page_size=16, context_length)

Allocate an empty paged KV manager. `prototype` is an (empty is fine) array
whose storage kind + eltype page allocation mirrors via `similar` — pass the
session's tensor storage (CPU `Array{Float64}`, CUDA `CuArray{Float32}`,
§LXXVII). `page_size < 1` and non-positive extents throw `ERR_INVALID_PLAN`.
"""
function PagedKVManager(
    prototype;
    n_layers::Int,
    n_kv_heads::Int,
    d_head::Int,
    page_size::Int=16,
    context_length::Int,
)
    n_layers >= 1 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "PagedKVManager: n_layers must be ≥ 1";
            n_layers=n_layers,
        ),
    )
    n_kv_heads >= 1 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "PagedKVManager: n_kv_heads must be ≥ 1";
            n_kv_heads=n_kv_heads,
        ),
    )
    d_head >= 1 || throw(
        gesso_error(ERR_INVALID_PLAN, "PagedKVManager: d_head must be ≥ 1"; d_head=d_head),
    )
    page_size >= 1 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "PagedKVManager: page_size must be ≥ 1 (got $page_size)";
            page_size=page_size,
        ),
    )
    context_length >= 1 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "PagedKVManager: context_length must be ≥ 1";
            context_length=context_length,
        ),
    )
    return PagedKVManager(
        n_layers,
        n_kv_heads,
        d_head,
        page_size,
        context_length,
        [KVPage[] for _ in 1:n_layers],
        [KVPage[] for _ in 1:n_layers],
        prototype,
    )
end

"""
    filled_len(mgr, layer, kind) -> Int

Filled token-rows of ONE cache, derived from its pages (Σ page.filled) —
the single source of truth; nothing else can drift from it.
"""
filled_len(mgr::PagedKVManager, layer::Int, kind::Symbol) =
    _pages_filled(_pages(mgr, layer, kind))

"""
    kv_len(mgr) -> Int

Sequence length as the engine sees it: the derived filled length of the
canonical cache (layer 1, K side). The engine appends every (layer, kind)
in lockstep — one row per token — so all caches agree with this count; tests
that drive a single cache in isolation see exactly what they wrote.
"""
kv_len(mgr::PagedKVManager) = filled_len(mgr, 1, :k)

"""
    kv_cache(mgr, layer, kind) -> KVCache

The §CIX semantic identity of this layer's K or V: `shape` is the logical
FILLED shape `(filled_len, n_kv_heads, d_head)` (derived from the pages —
it cannot lie); `storage` holds the page table (`Vector{KVPage}`,
provenance per Magenta §12.2). Rebuilt per call — the KVCache struct is
immutable (§CIX); the page table CONTENT is what mutates (same discipline
as Activation in the interpreter).
"""
function kv_cache(mgr::PagedKVManager, layer::Int, kind::Symbol)
    pages = _pages(mgr, layer, kind)
    n = _pages_filled(pages)
    return KVCache(; shape=(n, mgr.n_kv_heads, mgr.d_head), storage=pages)
end

"""
    append_kv!(mgr, layer, kind, row)

Write one token-row — shape `(n_kv_heads, d_head)`, post-RoPE (the oracle
caches post-RoPE K/V, §LXXV) — into the next row of the cache's last page,
allocating exactly one new page when it is full. Appending past
`context_length` throws `ERR_RESOURCE_LIMIT`; the cache is never realloc-
copied and tokens are never silently dropped or wrapped (§LXX).

    append_kv!(mgr, layer, k_row, v_row)

Paired form: appends K and V for the SAME token (the engine's unit of work).
"""
function append_kv!(mgr::PagedKVManager, layer::Int, kind::Symbol, row)
    pages = _pages(mgr, layer, kind)
    size(row) == (mgr.n_kv_heads, mgr.d_head) || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "append_kv!: row must be ($(_kind_str(kind))) (n_kv_heads, d_head) = " *
            "($(mgr.n_kv_heads), $(mgr.d_head)), got $(size(row))";
            layer=layer,
            kind=kind,
        ),
    )
    filled = _pages_filled(pages)
    filled < mgr.context_length || throw(
        gesso_error(
            ERR_RESOURCE_LIMIT,
            "append_kv!: context_length $(mgr.context_length) exhausted — " *
            "refusing to append $(_kind_str(kind)) token-row $(filled + 1) " *
            "(no silent drop, no wraparound, §LXX)";
            context_length=mgr.context_length,
            filled=filled,
            layer=layer,
            kind=kind,
        ),
    )

    page = isempty(pages) || pages[end].filled >= mgr.page_size ? nothing : pages[end]
    if page === nothing
        # allocate ONE page off the prototype (CPU Array{Float64} or device
        # CuArray{Float32}, §LXXVII). start_pos is 0-based (§LXXV positions).
        page = KVPage(
            layer,
            kind,
            filled,
            0,
            similar(mgr.prototype, mgr.page_size, mgr.n_kv_heads, mgr.d_head),
        )
        push!(pages, page)
    end
    i = page.filled + 1
    @views page.storage[i, :, :] .= row
    page.filled = i
    return mgr
end

function append_kv!(mgr::PagedKVManager, layer::Int, k_row, v_row)
    append_kv!(mgr, layer, :k, k_row)
    append_kv!(mgr, layer, :v, v_row)
    return mgr
end

"""
    gather_kv!(dest, mgr, layer, kind; len=filled_len(mgr, layer, kind))

Copy the first `len` filled token-rows into the contiguous `dest`
(`(len, n_kv_heads, d_head)`, same storage kind). Pure row copies —
bit-identical on CPU by construction (§LXXVIII: the manager owns storage,
not a kernel; attention contracts run over the gathered scratch).
"""
function gather_kv!(dest, mgr::PagedKVManager, layer::Int, kind::Symbol; len::Int=-1)
    n = filled_len(mgr, layer, kind)
    len = len < 0 ? n : len
    0 <= len <= n ||
        throw(gesso_error(ERR_INVALID_PLAN, "gather_kv!: len $len outside 0:$n"; len=len))
    size(dest) == (len, mgr.n_kv_heads, mgr.d_head) || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "gather_kv!: dest must be ($len, $(mgr.n_kv_heads), $(mgr.d_head)), got $(size(dest))",
        ),
    )
    pages = _pages(mgr, layer, kind)
    remaining = len
    for page in pages
        remaining <= 0 && break
        take = min(page.filled, remaining)
        take == 0 && continue
        r1 = page.start_pos + 1          # 0-based page origin → 1-based rows
        @views dest[r1:(r1+take-1), :, :] .= page.storage[1:take, :, :]
        remaining -= take
    end
    return dest
end

"""
    gather_kv(mgr, layer, kind; len=filled_len(mgr, layer, kind)) -> contiguous scratch

Allocating form of `gather_kv!` — a fresh contiguous `(len, n_kv_heads,
d_head)` array of the prototype's storage kind.
"""
function gather_kv(mgr::PagedKVManager, layer::Int, kind::Symbol; len::Int=-1)
    n = len < 0 ? filled_len(mgr, layer, kind) : len
    dest = similar(mgr.prototype, n, mgr.n_kv_heads, mgr.d_head)
    return gather_kv!(dest, mgr, layer, kind; len=n)
end

export PagedKVManager, append_kv!, filled_len, gather_kv, gather_kv!, kv_cache, kv_len
