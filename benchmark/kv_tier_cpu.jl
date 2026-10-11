# CPU half of the tiered KV attention: the row-split partial and merge over a
# host-resident cache (src/Inference/kv_tier.jl), timed per layer and per token.
#
#   julia --project=. -t 8 benchmark/kv_tier_cpu.jl
#
# Environment (all optional):
#   KV_TIER_ROWS     cache rows attended per layer          (65536)
#   KV_TIER_LAYERS   layers, each with its own K and V cache (10)
#   KV_TIER_ITERS    timed passes over all layers            (20)
#   KV_TIER_FORMAT   f32 | f16 | q8 | q4h                    (q8)
#   KV_TIER_HEADS    query heads                             (16)
#   KV_TIER_KV_HEADS KV heads                                (2)
#   KV_TIER_D        head dim                                (256)
#
# Each pass runs once untimed first, so compilation is never in a sample. The
# per-layer figure is the median over all (layer, pass) samples. The per-token
# figure is the median over passes of the sum across layers. GB/s counts the
# bytes of K and V (codes and scales) read by one layer.

using Gesso
using Random: MersenneTwister
using Printf: @printf

const _KVI = Gesso.Inference

_kvb_env(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))

const KVB_ROWS = _kvb_env("KV_TIER_ROWS", 65536)
const KVB_LAYERS = _kvb_env("KV_TIER_LAYERS", 10)
const KVB_ITERS = _kvb_env("KV_TIER_ITERS", 20)
const KVB_HEADS = _kvb_env("KV_TIER_HEADS", 16)
const KVB_KV_HEADS = _kvb_env("KV_TIER_KV_HEADS", 2)
const KVB_D = _kvb_env("KV_TIER_D", 256)
const KVB_FORMAT = Symbol(lowercase(get(ENV, "KV_TIER_FORMAT", "q8")))

_median(x) = (s = sort(x); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

_kvb_bytes(x::_KVI.QuantKV) = sizeof(x.codes) + sizeof(x.scales)
_kvb_bytes(x::AbstractArray) = sizeof(x)

function _kvb_cache(fmt::Symbol, x::Array{Float32,3})
    rows, n_kv, d = size(x)
    fmt === :f32 && return copy(x)
    fmt === :f16 && return Float16.(x)
    c = _KVI._kv_alloc(fmt, zeros(Float32, 0, 0, 0), rows, n_kv, d)
    _KVI._kv_write_rows!(c, x, 1)
    return c
end

function _kvb_pass!(attn, q, kcs, vcs, times, it)
    for l in eachindex(kcs)
        t0 = time_ns()
        _KVI._attention_split_cpu!(attn, q, kcs[l], vcs[l], KVB_ROWS, KVB_HEADS, KVB_D,
            div(KVB_HEADS, KVB_KV_HEADS))
        times[l, it] = (time_ns() - t0) / 1.0e6
    end
    return times
end

function main()
    fmt = KVB_FORMAT
    fmt in _KVI.KV_FORMATS || error("KV_TIER_FORMAT=$fmt: expected one of $(_KVI.KV_FORMATS)")
    rng = MersenneTwister(0x6b5e)
    kf = randn(rng, Float32, KVB_ROWS, KVB_KV_HEADS, KVB_D)
    vf = randn(rng, Float32, KVB_ROWS, KVB_KV_HEADS, KVB_D)
    q = randn(rng, Float32, 1, KVB_HEADS, KVB_D)
    fmt === :q4h && (q = _KVI._kv_rotate!(q, true))
    kcs = [_kvb_cache(fmt, kf) for _ in 1:KVB_LAYERS]
    vcs = [_kvb_cache(fmt, vf) for _ in 1:KVB_LAYERS]
    attn = zeros(Float32, 1, KVB_HEADS, KVB_D)
    bytes_per_layer = _kvb_bytes(kcs[1]) + _kvb_bytes(vcs[1])

    times = zeros(Float64, KVB_LAYERS, KVB_ITERS)
    _kvb_pass!(attn, q, kcs, vcs, zeros(Float64, KVB_LAYERS, 1), 1)
    for it in 1:KVB_ITERS
        _kvb_pass!(attn, q, kcs, vcs, times, it)
    end

    layer_median = _median(vec(times))
    token_median = _median(sum(times; dims=1)[:])
    gbps = bytes_per_layer / (layer_median * 1.0e-3) / 1.0e9
    @printf("kv_tier_cpu format=%s rows=%d layers=%d iters=%d heads=%d kv_heads=%d d=%d threads=%d\n",
        fmt, KVB_ROWS, KVB_LAYERS, KVB_ITERS, KVB_HEADS, KVB_KV_HEADS, KVB_D, Threads.nthreads())
    @printf("kv_tier_cpu median_ms_per_layer=%.4f median_ms_per_token=%.4f kv_bytes_per_layer=%d gb_per_s=%.3f\n",
        layer_median, token_median, bytes_per_layer, gbps)
    return nothing
end

main()
