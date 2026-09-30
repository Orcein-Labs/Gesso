# Phase 5 item A tests (§LXXVIII): paged KV manager — Magenta §9.5 step 1.
# Phase 7 item A tests (§LXXX): CoW aliasing — Magenta §9.5 step 2.
#
# The gate is BIT-IDENTITY on CPU: gathering the logical view equals `vcat`
# of the written rows. Page-boundary appends (page_size=4 with 5+ rows) are
# mandatory, not optional. Exceeding context_length is a typed
# ERR_RESOURCE_LIMIT — never a silent drop or wraparound (§LXX). CoW:
# appending into a shared last page copies THAT page only; full shared pages
# stay `===` forever; independent managers never share storage.

using .GessoTestHelpers: approx_eq

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const KVManager = Gesso.Inference.PagedKVManager
const KVPageT = Gesso.Inference.KVPage

# deterministic row filler: row i (0-based) → i/10 in every cell
_row(n_kv_heads, d_head, i) = fill(Float64(i / 10), n_kv_heads, d_head)

# vcat of (h, d) rows stacked along dim 1 = the logical (n, h, d) view
_rows(n_kv_heads, d_head, is) = cat(
    [reshape(_row(n_kv_heads, d_head, i), 1, n_kv_heads, d_head) for i in is]...;
    dims=1,
)

function _append_rows!(mgr, layer, n; kind=:k)
    for i in 0:(n-1)
        Gesso.Inference.append_kv!(mgr, layer, kind, _row(mgr.n_kv_heads, mgr.d_head, i))
    end
    return mgr
end

@testset "paged KV manager (§LXXVIII item A, Magenta §9.5 step 1)" begin
    @testset "construction validation" begin
        @test_throws Gesso.GessoError KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=0,
            context_length=16,
        )
        @test_throws Gesso.GessoError KVManager(
            Float64[];
            n_layers=0,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        @test_throws Gesso.GessoError KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=0,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        @test_throws Gesso.GessoError KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=0,
            page_size=4,
            context_length=16,
        )
        @test_throws Gesso.GessoError KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=0,
        )

        err = try
            KVManager(
                Float64[];
                n_layers=1,
                n_kv_heads=2,
                d_head=4,
                page_size=0,
                context_length=16,
            )
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN

        mgr = KVManager(
            Float64[];
            n_layers=2,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        @test Gesso.Inference.kv_len(mgr) == 0
        @test Gesso.Inference.filled_len(mgr, 1, :k) == 0
        @test Gesso.Inference.filled_len(mgr, 1, :v) == 0
    end

    @testset "append 1 row: kv_len == 1, logical view shape (1, h, d)" begin
        mgr = KVManager(
            Float64[];
            n_layers=2,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(mgr, 1, 1)
        @test Gesso.Inference.kv_len(mgr) == 1
        @test Gesso.Inference.kv_cache(mgr, 1, :k).shape == (1, 2, 4)
        @test Gesso.Inference.kv_cache(mgr, 1, :k) isa Gesso.KVCache
    end

    @testset "page boundary: 5 rows at page_size=4 → second page exists, kv_len == 5" begin
        mgr = KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(mgr, 1, 5)
        @test Gesso.Inference.kv_len(mgr) == 5
        pages = Gesso.Inference.kv_cache(mgr, 1, :k).storage
        @test length(pages) == 2
        @test pages[1].filled == 4 && pages[2].filled == 1
        @test pages[1].start_pos == 0 && pages[2].start_pos == 4
    end

    @testset "gather == vcat of written rows (bit-identical, CPU)" begin
        mgr = KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(mgr, 1, 10)                     # crosses TWO boundaries
        @test Gesso.Inference.kv_len(mgr) == 10
        gathered = Gesso.Inference.gather_kv(mgr, 1, :k)
        expected = _rows(2, 4, 0:9)
        @test gathered isa Array{Float64, 3}
        @test isequal(gathered, expected)             # BIT-identical, not approx
        # partial gather: prefix rows only
        part = Gesso.Inference.gather_kv(mgr, 1, :k; len=6)
        @test isequal(part, _rows(2, 4, 0:5))
        # gathering at len 0 and full-page multiples works
        @test isempty(Gesso.Inference.gather_kv(mgr, 1, :k; len=0))
        @test isequal(Gesso.Inference.gather_kv(mgr, 1, :k; len=4), _rows(2, 4, 0:3))
    end

    @testset "K and V are independent logical caches" begin
        mgr = KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(mgr, 1, 3; kind=:k)
        _append_rows!(mgr, 1, 2; kind=:v)
        @test Gesso.Inference.filled_len(mgr, 1, :k) == 3
        @test Gesso.Inference.filled_len(mgr, 1, :v) == 2
        @test Gesso.Inference.kv_len(mgr) == 3         # canonical cache = layer 1 K
        # the paired form appends both for the same token
        Gesso.Inference.append_kv!(mgr, 1, _row(2, 4, 3), _row(2, 4, 0))
        @test Gesso.Inference.filled_len(mgr, 1, :k) == 4
        @test Gesso.Inference.filled_len(mgr, 1, :v) == 3
    end

    @testset "context exhaustion: typed ERR_RESOURCE_LIMIT, no drop, no wrap" begin
        mgr = KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=6,
        )
        _append_rows!(mgr, 1, 6)
        @test Gesso.Inference.kv_len(mgr) == 6
        err = try
            _append_rows!(mgr, 1, 1)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_RESOURCE_LIMIT
        @test occursin("context_length", sprint(showerror, err))
        @test Gesso.Inference.kv_len(mgr) == 6        # nothing written past the limit
        # gathered data is unchanged by the refused append
        gathered = Gesso.Inference.gather_kv(mgr, 1, :k)
        @test isequal(gathered, _rows(2, 4, 0:5))
    end

    @testset "page provenance (Magenta §12.2 hook) is consistent with kv_len" begin
        for ps in (4, 16)
            mgr = KVManager(
                Float64[];
                n_layers=2,
                n_kv_heads=2,
                d_head=4,
                page_size=ps,
                context_length=64,
            )
            _append_rows!(mgr, 1, 9)
            _append_rows!(mgr, 2, 5)                  # layers are independent too
            for (li, layer) in enumerate((1, 2))
                pages = Gesso.Inference.kv_cache(mgr, layer, :k).storage
                n = layer == 1 ? 9 : 5
                @test sum(p.filled for p in pages) == n
                @test pages[1].start_pos == 0
                for i in 2:length(pages)
                    @test pages[i].start_pos == pages[i-1].start_pos + pages[i-1].filled
                end
                @test all(p.layer == layer for p in pages)
                @test all(p.kind == :k for p in pages)
                @test all(0 <= p.filled <= ps for p in pages)
                @test Gesso.Inference.filled_len(mgr, layer, :k) == n
            end
        end
    end

    @testset "layers are independent page lists" begin
        mgr = KVManager(
            Float64[];
            n_layers=2,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(mgr, 1, 5)
        @test Gesso.Inference.filled_len(mgr, 1, :k) == 5
        @test isempty(Gesso.Inference.kv_cache(mgr, 2, :k).storage)
        @test Gesso.Inference.kv_cache(mgr, 2, :k).shape == (0, 2, 4)
    end

    @testset "gather_kv! destination validation" begin
        mgr = KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(mgr, 1, 3)
        @test_throws Gesso.GessoError Gesso.Inference.gather_kv!(zeros(2, 2, 4), mgr, 1, :k)
        @test_throws Gesso.GessoError Gesso.Inference.gather_kv(mgr, 1, :k; len=5)
        @test_throws Gesso.GessoError Gesso.Inference.gather_kv(mgr, 1, :x)
        @test_throws Gesso.GessoError Gesso.Inference.append_kv!(mgr, 3, :k, _row(2, 4, 0))
        @test_throws Gesso.GessoError Gesso.Inference.append_kv!(mgr, 1, :k, zeros(3, 4))
    end

    @testset "CoW aliasing (§LXXX item A, Magenta §9.5 step 2)" begin
        # helper: make B's (empty) page lists alias A's current pages
        function _alias!(b, a)
            Gesso.Inference._alias_pages!(b, a)
            return b
        end
        a = KVManager(
            Float64[];
            n_layers=2,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(a, 1, 6)                      # crosses one boundary: 2 pages
        _append_rows!(a, 1, 6; kind=:v)
        _append_rows!(a, 2, 3)                      # partial page on layer 2

        b = _alias!(
            KVManager(
                Float64[];
                n_layers=2,
                n_kv_heads=2,
                d_head=4,
                page_size=4,
                context_length=16,
            ),
            a,
        )
        @test Gesso.Inference.kv_len(b) == Gesso.Inference.kv_len(a) == 6

        # every aliased page: same object, same storage, marked shared
        for kind in (:k, :v), layer in 1:2
            pa = Gesso.Inference.kv_cache(a, layer, kind).storage
            pb = Gesso.Inference.kv_cache(b, layer, kind).storage
            @test length(pa) == length(pb)
            for i in eachindex(pa)
                @test pb[i] === pa[i]
                @test pb[i].storage === pa[i].storage
                @test pb[i].shared == true
            end
        end

        # append one row on B (partial last page): B's last page is a NEW
        # object with NEW storage; A's page object-id, shared flag, and
        # filled bytes are untouched
        last_a = Gesso.Inference.kv_cache(a, 1, :k).storage[end]
        bytes_a = copy(Gesso.Inference.gather_kv(a, 1, :k))
        Gesso.Inference.append_kv!(b, 1, :k, _row(2, 4, 6))
        last_b = Gesso.Inference.kv_cache(b, 1, :k).storage[end]
        @test last_b !== last_a
        @test last_b.storage !== last_a.storage
        @test last_b.shared == false
        @test last_b.filled == last_a.filled + 1 == 3
        @test last_b.start_pos == last_a.start_pos == 4
        @test last_a.shared == true                 # original stays shared
        @test Gesso.Inference.kv_len(a) == 6        # A did not grow
        @test Gesso.Inference.kv_len(b) == 7        # B did
        @test isequal(Gesso.Inference.gather_kv(a, 1, :k), bytes_a)
        @test isequal(Gesso.Inference.gather_kv(b, 1, :k), _rows(2, 4, 0:6))

        # the FULL shared page behind it is still identical in B
        @test Gesso.Inference.kv_cache(b, 1, :k).storage[1] ===
              Gesso.Inference.kv_cache(a, 1, :k).storage[1]

        # filling B's CoW'd page to full and appending again: the NEXT page
        # is fresh+unshared in B, and the full shared page in A stays ===
        Gesso.Inference.append_kv!(b, 1, :k, _row(2, 4, 7))
        @test Gesso.Inference.kv_cache(b, 1, :k).storage[end] === last_b
        Gesso.Inference.append_kv!(b, 1, :k, _row(2, 4, 8))     # boundary: new page
        pages_b = Gesso.Inference.kv_cache(b, 1, :k).storage
        @test length(pages_b) == 3
        @test pages_b[2] === last_b && pages_b[2].filled == 4
        @test pages_b[3].filled == 1 && pages_b[3].shared == false
        @test pages_b[3].start_pos == 8
        @test Gesso.Inference.kv_cache(a, 1, :k).storage[1] === pages_b[1]
        @test Gesso.Inference.kv_cache(a, 1, :k).storage[end] === last_a
        @test Gesso.Inference.kv_len(a) == 6

        # filled length still derives from pages; context exhaustion intact
        @test Gesso.Inference.filled_len(b, 1, :k) ==
              sum(p.filled for p in Gesso.Inference.kv_cache(b, 1, :k).storage)
        full_ctx = KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=6,
        )
        _append_rows!(full_ctx, 1, 6)
        shared_ctx = _alias!(
            KVManager(
                Float64[];
                n_layers=1,
                n_kv_heads=2,
                d_head=4,
                page_size=4,
                context_length=6,
            ),
            full_ctx,
        )
        err = try
            Gesso.Inference.append_kv!(shared_ctx, 1, :k, _row(2, 4, 6))
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_RESOURCE_LIMIT

        # alias validation: geometry mismatch and non-empty dst are typed errors
        @test_throws Gesso.GessoError Gesso.Inference._alias_pages!(
            KVManager(
                Float64[];
                n_layers=2,
                n_kv_heads=2,
                d_head=4,
                page_size=4,
                context_length=8,
            ),
            a,
        )
        @test_throws Gesso.GessoError Gesso.Inference._alias_pages!(b, a)   # b has pages

        # two managers filled independently NEVER share (declaration, not discovery)
        c = KVManager(
            Float64[];
            n_layers=2,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        _append_rows!(c, 1, 6)
        _append_rows!(c, 1, 6; kind=:v)
        _append_rows!(c, 2, 3)
        for kind in (:k, :v), layer in 1:2
            pa = Gesso.Inference.kv_cache(a, layer, kind).storage
            pc = Gesso.Inference.kv_cache(c, layer, kind).storage
            @test all(pa[i].storage !== pc[i].storage for i in eachindex(pa))
            @test all(p.shared == false for p in pc)
        end

        # alias of an EMPTY manager is a no-op (fresh Session fork path)
        e1 = KVManager(
            Float64[];
            n_layers=1,
            n_kv_heads=2,
            d_head=4,
            page_size=4,
            context_length=16,
        )
        e2 = _alias!(
            KVManager(
                Float64[];
                n_layers=1,
                n_kv_heads=2,
                d_head=4,
                page_size=4,
                context_length=16,
            ),
            e1,
        )
        @test isempty(Gesso.Inference.kv_cache(e2, 1, :k).storage)
    end

    @testset "CUDA: append/gather on CuArray{Float32}" begin
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
                "no NVIDIA device (CUDA.functional() == false) — paged-KV CUDA tests skipped (§LXXVIII skip law)",
            )
        else
            proto = CUDA.zeros(Float32, 0)          # device prototype
            mgr = KVManager(
                proto;
                n_layers=1,
                n_kv_heads=2,
                d_head=4,
                page_size=4,
                context_length=16,
            )
            for i in 0:5                             # crosses one boundary
                row = CUDA.CuArray(fill(Float32(i / 10), 2, 4))
                Gesso.Inference.append_kv!(mgr, 1, :k, row)
            end
            @test Gesso.Inference.kv_len(mgr) == 6
            pages = Gesso.Inference.kv_cache(mgr, 1, :k).storage
            @test length(pages) == 2
            @test pages[1].storage isa CUDA.CuArray
            gathered = Array(Gesso.Inference.gather_kv(mgr, 1, :k))   # host copy is explicit
            @test isequal(
                gathered,
                cat([fill(Float32(i / 10), 1, 2, 4) for i in 0:5]...; dims=1),
            )
        end
    end
end
