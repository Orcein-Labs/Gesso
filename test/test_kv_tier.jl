# Exact attention split over KV row ranges (src/Inference/kv_tier.jl).
#
# Each range's partial is merged by log-sum-exp and compared with the reference
# single-pass decode over all rows. Both sides read the same cache, so a
# quantized format is compared on its own dequantized values: the split adds
# only float rounding, and the bound stays at the f32 level for every format.

using Random: MersenneTwister

const _KVT = Gesso.Inference

_kvt_relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

function _kvt_cache(fmt::Symbol, x::Array{Float32,3})
    rows, n_kv, d = size(x)
    fmt === :f32 && return copy(x)
    fmt === :f16 && return Float16.(x)
    c = _KVT._kv_alloc(fmt, zeros(Float32, 0, 0, 0), rows, n_kv, d)
    _KVT._kv_write_rows!(c, x, 1)
    return c
end

# q4h caches hold rotated rows, so the query takes the same rotation the decode
# step applies before attention.
_kvt_query(fmt::Symbol, q) = fmt === :q4h ? _KVT._kv_rotate!(copy(q), true) : q

function _kvt_reference(q, kc, vc, n_heads, d, group, K)
    ref = zeros(Float32, 1, n_heads, d)
    _KVT._attention_decode_cpu!(ref, q, kc, vc, zeros(Float32, K), n_heads, d, group, K)
    return ref
end

# A random split of 1:K into n ranges. Empty and single-row ranges are allowed.
function _kvt_split(rng, K::Int, n::Int)
    cuts = sort(rand(rng, 0:K, n - 1))
    bounds = vcat(0, cuts, K)
    return [(bounds[i] + 1):bounds[i + 1] for i in 1:n]
end

function _kvt_merged(q, kc, vc, ranges, n_heads, d, group; int_scores::Bool=false)
    total = _KVT._kv_partial(n_heads, d)
    tmp = _KVT._kv_partial(n_heads, d)
    for r in ranges
        _KVT._attention_partial_cpu!(tmp, q, kc, vc, r, n_heads, d, group; int_scores)
        _KVT._partial_merge!(total, tmp)
    end
    out = zeros(Float32, 1, n_heads, d)
    _KVT._partial_finish!(out, total)
    return out
end

const _KVT_FORMATS = (:f32, :f16, :q8, :q4h)

@testset "kv tier: row split equals the reference attention for every format" begin
    n_heads, n_kv, d, K = 16, 2, 256, 1000
    group = div(n_heads, n_kv)
    for (fi, fmt) in enumerate(_KVT_FORMATS)
        rng = MersenneTwister(0x6b70 + fi)
        q = randn(rng, Float32, 1, n_heads, d)
        kc = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
        vc = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
        qq = _kvt_query(fmt, q)
        ref = _kvt_reference(qq, kc, vc, n_heads, d, group, K)

        @test _kvt_relerr(_kvt_merged(qq, kc, vc, [1:K], n_heads, d, group), ref) <= 1e-5
        @test _kvt_relerr(_kvt_merged(qq, kc, vc, [1:0, 1:K], n_heads, d, group), ref) <= 1e-5
        @test _kvt_relerr(_kvt_merged(qq, kc, vc, [1:K, (K + 1):K], n_heads, d, group), ref) <= 1e-5
        @test _kvt_relerr(_kvt_merged(qq, kc, vc, [1:1, 2:K], n_heads, d, group), ref) <= 1e-5

        worst2 = 0.0
        worst3 = 0.0
        for _ in 1:20
            r2 = _kvt_split(rng, K, 2)
            worst2 = max(worst2, _kvt_relerr(_kvt_merged(qq, kc, vc, r2, n_heads, d, group), ref))
            r3 = _kvt_split(rng, K, 3)
            worst3 = max(worst3, _kvt_relerr(_kvt_merged(qq, kc, vc, r3, n_heads, d, group), ref))
        end
        a = rand(rng, 2:(K - 1))
        single = [1:a, (a + 1):(a + 1), (a + 2):K]
        worst3 = max(worst3, _kvt_relerr(_kvt_merged(qq, kc, vc, single, n_heads, d, group), ref))
        @test worst2 <= 1e-5
        @test worst3 <= 1e-5
        @info "kv tier split" fmt worst_relerr_2range=worst2 worst_relerr_3range=worst3
    end
end

@testset "kv tier: large scores do not overflow" begin
    n_heads, n_kv, d, K = 16, 2, 256, 1000
    group = div(n_heads, n_kv)
    for (fi, fmt) in enumerate(_KVT_FORMATS)
        rng = MersenneTwister(0x6b80 + fi)
        q = 50 .* randn(rng, Float32, 1, n_heads, d)
        kc = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
        vc = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
        qq = _kvt_query(fmt, q)
        ref = _kvt_reference(qq, kc, vc, n_heads, d, group, K)
        split = _kvt_merged(qq, kc, vc, [1:333, 334:700, 701:K], n_heads, d, group)
        @test all(isfinite, split)
        # At this score scale the Float32 reference is itself about 2e-5 from an
        # exact Float64 attention, so 1e-5 would measure reference noise. This
        # bound still catches a dropped rescale, which is off by O(1).
        @test _kvt_relerr(split, ref) <= 1e-4
    end
end

@testset "kv tier: threaded block driver is deterministic and matches the serial merge" begin
    n_heads, n_kv, d, K = 16, 2, 64, 5000
    group = div(n_heads, n_kv)
    rng = MersenneTwister(0x6b90)
    q = randn(rng, Float32, 1, n_heads, d)
    kc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    vc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    block = 512
    threaded = zeros(Float32, 1, n_heads, d)
    _KVT._attention_split_cpu!(threaded, q, kc, vc, K, n_heads, d, group; block_rows=block)
    ranges = [((b - 1) * block + 1):min(K, b * block) for b in 1:cld(K, block)]
    serial = _kvt_merged(q, kc, vc, ranges, n_heads, d, group)
    @test threaded == serial
    ref = _kvt_reference(q, kc, vc, n_heads, d, group, K)
    @test _kvt_relerr(threaded, ref) <= 1e-5
end

@testset "kv tier: quantized caches stay near the f32 attention" begin
    # Against the unquantized cache, q8 is within half a step of amax/127 per
    # element; the attention output then moves by a few percent at most.
    n_heads, n_kv, d, K = 16, 2, 256, 1000
    group = div(n_heads, n_kv)
    rng = MersenneTwister(0x6ba0)
    q = randn(rng, Float32, 1, n_heads, d)
    kf = randn(rng, Float32, K, n_kv, d)
    vf = randn(rng, Float32, K, n_kv, d)
    f32 = _kvt_reference(q, kf, vf, n_heads, d, group, K)
    for fmt in (:q8, :q4h)
        out = _kvt_merged(_kvt_query(fmt, q), _kvt_cache(fmt, kf), _kvt_cache(fmt, vf),
            [1:400, 401:K], n_heads, d, group)
        fmt === :q4h && _KVT._kv_rotate!(out, false)
        @test _kvt_relerr(out, f32) <= (fmt === :q8 ? 0.05 : 0.25)
    end
end

@testset "kv tier: empty partial is the merge identity and cannot finish" begin
    n_heads, d = 4, 8
    rng = MersenneTwister(0x6bb0)
    q = randn(rng, Float32, 1, n_heads, d)
    kc = randn(rng, Float32, 30, 2, d)
    vc = randn(rng, Float32, 30, 2, d)
    full = _kvt_merged(q, kc, vc, [1:30], n_heads, d, 2)
    @test _kvt_merged(q, kc, vc, [1:0, 1:15, 16:16, 17:17, 18:30], n_heads, d, 2) ≈ full atol = 1e-6
    empty = _KVT._kv_partial(n_heads, d)
    @test_throws Gesso.GessoError _KVT._partial_finish!(zeros(Float32, 1, n_heads, d), empty)
end

# Forwards to a cache's getindex without being a QuantKV, so the generic methods
# run over the same codes and scales the q8 fast path reads.
struct _KVTGeneric{C<:AbstractArray} <: AbstractArray{Float32,3}
    x::C
end
Base.size(w::_KVTGeneric) = size(w.x)
Base.getindex(w::_KVTGeneric, i::Int, j::Int, k::Int) = w.x[i, j, k]

@testset "kv tier: q8 fast path matches the generic getindex path on the same cache" begin
    n_heads, n_kv, d, K = 16, 2, 256, 1000
    group = div(n_heads, n_kv)
    rng = MersenneTwister(0x6bc0)
    q = randn(rng, Float32, 1, n_heads, d)
    kc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    vc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    ranges = [1:400, 401:K]
    fast = _kvt_merged(q, kc, vc, ranges, n_heads, d, group)
    generic = _kvt_merged(q, _KVTGeneric(kc), _KVTGeneric(vc), ranges, n_heads, d, group)
    @test _kvt_relerr(fast, generic) <= 1e-5
    fresh = zeros(Float32, 1, n_heads, d)
    _KVT._attention_split_cpu!(fresh, q, kc, vc, K, n_heads, d, group)
    ws = _KVT.KVTierWorkspace(K, n_heads, d, group)
    reused = zeros(Float32, 1, n_heads, d)
    for _ in 1:2
        _KVT._attention_split_cpu!(reused, q, kc, vc, K, n_heads, d, group; ws=ws)
    end
    @test reused == fresh
end

# Attention in Float64 over the dequantized cache, so the only error left is the
# one the integer score path adds.
function _kvt_exact64(q, kc, vc, n_heads, d, group, K)
    out = zeros(Float64, 1, n_heads, d)
    for hh in 1:n_heads
        hk = div(hh - 1, group) + 1
        sc = zeros(Float64, K)
        for u in 1:K, j in 1:d
            sc[u] += Float64(q[1, hh, j]) * Float64(kc[u, hk, j])
        end
        sc ./= sqrt(d)
        w = exp.(sc .- maximum(sc))
        w ./= sum(w)
        for u in 1:K, j in 1:d
            out[1, hh, j] += w[u] * Float64(vc[u, hk, j])
        end
    end
    return out
end

@testset "kv tier: integer q8 scores match Float64 attention" begin
    n_heads, n_kv, d, K = 16, 2, 256, 1000
    group = div(n_heads, n_kv)
    for (si, scale) in enumerate((1.0f0, 50.0f0))
        rng = MersenneTwister(0x6bd0 + si)
        q = scale .* randn(rng, Float32, 1, n_heads, d)
        kc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
        vc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
        ref = _kvt_exact64(q, kc, vc, n_heads, d, group, K)
        out = _kvt_merged(q, kc, vc, [1:333, 334:700, 701:K], n_heads, d, group; int_scores=true)
        err = maximum(abs.(out .- ref)) / maximum(abs.(ref))
        @test all(isfinite, out)
        @test err <= 5e-3
        @info "kv tier integer scores" scale worst_relerr_vs_f64=err
    end
end

@testset "kv tier: integer scores keep the needle row on top and dominate the output" begin
    n_heads, n_kv, d, K = 16, 2, 256, 4096
    group = div(n_heads, n_kv)
    rng = MersenneTwister(0x6be0)
    kf = randn(rng, Float32, K, n_kv, d)
    u = randn(rng, Float32, n_kv, d)
    needle = 4096
    for hk in 1:n_kv
        kf[needle, hk, :] .= 3.0f0 .* u[hk, :]
    end
    q = randn(rng, Float32, 1, n_heads, d)
    for hh in 1:n_heads
        hk = div(hh - 1, group) + 1
        q[1, hh, :] .+= 0.5f0 .* u[hk, :]
    end
    kc = _kvt_cache(:q8, kf)
    vf = zeros(Float32, K, n_kv, d)
    vf[needle, :, :] .= 1.0f0
    vc = _kvt_cache(:q8, vf)

    blk_f = _KVT._kv_tier_block(K, n_heads, d, group)
    blk_i = _KVT._kv_tier_block(K, n_heads, d, group)
    _KVT._scale_query!(blk_f.qs, q, d)
    _KVT._scale_query!(blk_i.qs, q, d)
    _KVT._quantize_query!(blk_i, d)
    for hk in 1:n_kv
        _KVT._score_block!(blk_f, kc, hk, 0, K, d, group)
        _KVT._score_block_int!(blk_i, kc, hk, 0, K, d, group)
        for g in 1:group
            @test argmax(blk_i.s[1:K, g]) == needle
            @test argmax(blk_f.s[1:K, g]) == argmax(blk_i.s[1:K, g])
        end
    end

    out = _kvt_merged(q, kc, vc, [1:K], n_heads, d, group; int_scores=true)
    @test minimum(out) > 0.9
    @test maximum(out) < 1.1
end

@testset "kv tier: integer partials merge to the single integer pass" begin
    n_heads, n_kv, d, K = 16, 2, 256, 2000
    group = div(n_heads, n_kv)
    rng = MersenneTwister(0x6bf0)
    q = randn(rng, Float32, 1, n_heads, d)
    kc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    vc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    single = _kvt_merged(q, kc, vc, [1:K], n_heads, d, group; int_scores=true)
    split = _kvt_merged(q, kc, vc, [1:1, 2:700, 701:1400, 1401:K], n_heads, d, group; int_scores=true)
    @test _kvt_relerr(split, single) <= 1e-5
    fresh = zeros(Float32, 1, n_heads, d)
    _KVT._attention_split_cpu!(fresh, q, kc, vc, K, n_heads, d, group; block_rows=512, int_scores=true)
    @test _kvt_relerr(fresh, single) <= 1e-5
end

@testset "kv tier: the integer-score gate can fail" begin
    # One 32-value scale block of every key row is doubled. The corrupted cache
    # must then miss the Float64 attention of the true cache by more than the gate.
    n_heads, n_kv, d, K = 16, 2, 256, 1000
    group = div(n_heads, n_kv)
    rng = MersenneTwister(0x6c00)
    q = randn(rng, Float32, 1, n_heads, d)
    kc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    vc = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
    ref = _kvt_exact64(q, kc, vc, n_heads, d, group, K)
    bad = _KVT.QuantKV{:q8}(kc.codes, copy(kc.scales), kc.d)
    bad.scales[:, :, 1] .*= Float16(2)
    out = _kvt_merged(q, bad, vc, [1:333, 334:700, 701:K], n_heads, d, group; int_scores=true)
    corrupt_err = _kvt_relerr(out, ref)
    @info "kv tier corrupted-scale control" relerr=corrupt_err
    @test corrupt_err > 5e-3
end
