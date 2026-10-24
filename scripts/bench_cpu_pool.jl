# CPU persistent pool vs Threads.@threads for the native quantized GEMV row kernels.
#
# One process per thread count, so the Julia thread count is the thread count:
#   flock /tmp/gesso-cpu-bench.lock nice -n 5 julia --project=. -t 8  scripts/bench_cpu_pool.jl
#   flock /tmp/gesso-cpu-bench.lock nice -n 5 julia --project=. -t 16 scripts/bench_cpu_pool.jl
#
# Both paths run the same row kernel (_qgemv_q8k_rows!) over the same even row
# split into nthreads() chunks. The baseline is _qgemv_q8k_threaded! (Threads.@threads);
# the candidate is pool_run! with one chunk per thread. Each shape is checked bit-exact
# once before timing. GB/s is weight bytes streamed per second (decimal GB).
#
# Weights are synthetic IQ4_XS rows with finite f16 scales. Copies rotate so each
# timed working set is about 256 MiB, which keeps the matrix out of the last-level cache.
# Each figure is the median of 3 samples; each sample averages enough calls to take
# about 0.25 s. Calls are warmed up first.

using Gesso

const I = Gesso.Inference
const RT = Gesso.Runtime
const BENCH_SAMPLES = 3
const BENCH_TARGET_S = 0.25
const WS_BYTES = 256 * 2^20
const FMT = I.GGML_TYPE_IQ4_XS
const SHAPES = (
    ("expert 2048x512", 2048, 512),
    ("expert 512x2048", 512, 2048),
    ("dense 5120x13824", 5120, 13824),
)

struct GemvChunks{X}
    y::Vector{Float32}
    xq::X
    W::Vector{UInt8}
    ne0::Int
    ne1::Int
    nc::Int
    simd::Bool
end
function (g::GemvChunks)(c::Int)
    j0 = div((c - 1) * g.ne1, g.nc) + 1
    j1 = div(c * g.ne1, g.nc)
    j0 <= j1 && I._qgemv_q8k_rows!(g.y, g.xq, g.W, FMT, g.ne0, j0, j1, 1.0f0, 0.0f0, g.simd)
    return nothing
end

function bench_median(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
end

# Median seconds per call. Each sample runs enough calls to take about BENCH_TARGET_S.
function bench_seconds(f)
    f()
    t1 = @elapsed f()
    reps = max(1, ceil(Int, BENCH_TARGET_S / max(t1, 1.0e-9)))
    samples = Float64[]
    for _ in 1:BENCH_SAMPLES
        t = @elapsed for _ in 1:reps
            f()
        end
        push!(samples, t / reps)
    end
    return bench_median(samples)
end

function bench_weights(ne0::Int, ne1::Int)
    bsz = I._iq_block_bytes(FMT)
    nb = div(ne0, I._q8k_block_values(FMT))
    doff = I._q8k_d_offset(FMT)
    w = rand(UInt8, ne1 * nb * bsz)
    for blk in 0:(ne1*nb-1)
        bits = reinterpret(UInt16, Float16(0.01f0 + 0.02f0 * rand(Float32)))
        w[blk*bsz+doff+1] = UInt8(bits & 0xff)
        w[blk*bsz+doff+2] = UInt8(bits >> 8)
    end
    return w
end

function rotated_weights(ne0::Int, ne1::Int)
    nbytes = ne1 * I._qrow_bytes(FMT, ne0)
    return [bench_weights(ne0, ne1) for _ in 1:max(1, cld(WS_BYTES, nbytes))]
end

function bench_shape(name::String, ne0::Int, ne1::Int, nt::Int)
    Ws = rotated_weights(ne0, ne1)
    nbytes = ne1 * I._qrow_bytes(FMT, ne0)
    k = Ref(0)
    next_w() = (k[]=mod(k[], length(Ws)) + 1; Ws[k[]])
    x = rand(Float32, ne0) .- 0.5f0
    xq = I._q8k_quantize_act(FMT, x)
    simd = I.q8k_simd_enabled()

    # Bit-exactness: the pool result equals the @threads result on the same weights.
    yt = zeros(Float32, ne1)
    yp = zeros(Float32, ne1)
    I._qgemv_q8k_threaded!(yt, xq, Ws[1], FMT, ne0, ne1; nchunks=nt)
    RT.pool_run!(GemvChunks(yp, xq, Ws[1], ne0, ne1, nt, simd), nt)
    exact = yp == yt

    y1 = zeros(Float32, ne1)
    t_thr = bench_seconds(
        () -> I._qgemv_q8k_threaded!(y1, xq, next_w(), FMT, ne0, ne1; nchunks=nt),
    )
    y2 = zeros(Float32, ne1)
    t_pool = bench_seconds(
        () -> RT.pool_run!(GemvChunks(y2, xq, next_w(), ne0, ne1, nt, simd), nt),
    )
    # Second, interleaved pass: removes order effects from the first.
    t_pool2 = bench_seconds(
        () -> RT.pool_run!(GemvChunks(y2, xq, next_w(), ne0, ne1, nt, simd), nt),
    )
    t_thr2 = bench_seconds(
        () -> I._qgemv_q8k_threaded!(y1, xq, next_w(), FMT, ne0, ne1; nchunks=nt),
    )

    thr_gbs = nbytes / min(t_thr, t_thr2) / 1.0e9
    pool_gbs = nbytes / min(t_pool, t_pool2) / 1.0e9
    thr_med = nbytes / bench_median([t_thr, t_thr2]) / 1.0e9
    pool_med = nbytes / bench_median([t_pool, t_pool2]) / 1.0e9
    println(
        rpad(name, 18),
        " threads=",
        lpad(nt, 2),
        "  @threads GB/s=",
        lpad(round(thr_med; digits=2), 6),
        "  pool GB/s=",
        lpad(round(pool_med; digits=2), 6),
        "  gain=",
        lpad(round(100 * (pool_med - thr_med) / thr_med; digits=1), 6),
        "%",
        "  call_us thr/pool=",
        round(t_thr * 1.0e6; digits=1),
        "/",
        round(t_pool * 1.0e6; digits=1),
        "  bitexact=",
        exact,
    )
    flush(stdout)
    return exact
end

function main()
    nt = Threads.nthreads()
    RT._cpu_pool_configure!(nt)
    println(
        "mode: IQ4_XS, threads=",
        nt,
        ", pool participants=",
        RT._cpu_pool_participants(),
        ", samples=",
        BENCH_SAMPLES,
        ", target_s=",
        BENCH_TARGET_S,
    )
    ok = true
    for (name, ne0, ne1) in SHAPES
        ok &= bench_shape(name, ne0, ne1, nt)
    end
    println(ok ? "all shapes bit-exact" : "BIT-EXACTNESS FAILED")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    main() || exit(1)
end
