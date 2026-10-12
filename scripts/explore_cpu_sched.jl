# Control: does the @threads schedule explain the gap between Gesso's GEMV and
# the persistent pool? Stdlib only. Compares, in one process and on the same
# weight copies:
#   prod     Gesso's _qgemv_q8k_threaded! (default @threads schedule, nthreads chunks)
#   static   the same row split with @threads :static (one chunk per thread)
# Run one process per thread count, under the CPU bench lock:
#
#   EXP_SAMPLES=9 flock /tmp/gesso-cpu-bench.lock nice -n 5 julia --project=. -t 16 scripts/explore_cpu_sched.jl

include(joinpath(@__DIR__, "explore_cpu_threading.jl"))

function static_gemv!(y, xq, q, ty, ne0, ne1, nc)
    @threads :static for c in 1:nc
        j0 = div((c - 1) * ne1, nc) + 1
        j1 = div(c * ne1, nc)
        I._qgemv_q8k_rows!(y, xq, q, ty, ne0, j0, j1, 1.0f0, 0.0f0, true)
    end
    return y
end

function sched_row(name::String, ne0::Int, ne1::Int)
    nt = nthreads()
    nbytes = ne1 * I._qrow_bytes(FMT, ne0)
    nextw = rotator(rotated_weights(ne0, ne1))
    x = rand(Float32, ne0) .- 0.5f0
    xq = I._q8k_quantize_act(FMT, x)
    y = zeros(Float32, ne1)
    t_prod = time_samples(() -> I._qgemv_q8k_threaded!(y, xq, nextw(), FMT, ne0, ne1))
    t_stat = time_samples(() -> static_gemv!(y, xq, nextw(), FMT, ne0, ne1, nt))
    sp = summarize(nbytes ./ t_prod ./ 1.0e9)
    ss = summarize(nbytes ./ t_stat ./ 1.0e9)
    println(rpad(name, 18), " threads=", nt, "  prod GB/s med ", round(sp.med; digits=2), " sd ", round(sp.sd; digits=2),
        "  | static GB/s med ", round(ss.med; digits=2), " sd ", round(ss.sd; digits=2))
    return nothing
end

function main()
    println("mode=sched threads=", nthreads(), " samples=", SAMPLES)
    println("uptime: ", strip(read(`uptime`, String)))
    for (name, ne0, ne1) in SHAPES
        sched_row(name, ne0, ne1)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
