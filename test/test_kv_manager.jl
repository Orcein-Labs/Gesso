# Phase 5 item A tests (§LXXVIII): paged KV manager — Magenta §9.5 step 1.
#
# The gate is BIT-IDENTITY on CPU: gathering the logical view equals `vcat`
# of the written rows. Page-boundary appends (page_size=4 with 5+ rows) are
# mandatory, not optional. Exceeding context_length is a typed
# ERR_RESOURCE_LIMIT — never a silent drop or wraparound (§LXX).

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
