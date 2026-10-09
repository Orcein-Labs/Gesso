# CPU quantized GEMV throughput: native Q8_K path vs the existing F32 dequant
# path, plus the host memory-read ceiling.
#
#   julia --project=. -t 18 scripts/bench_cpu_quant.jl
#   julia --project=. -t 1  scripts/bench_cpu_quant.jl
#
# Weights are synthetic random blocks (finite f16 super-scales). GB/s is weight
# bytes streamed per second, decimal GB. Each figure is the median of 3 timed
# samples, and each sample averages enough calls to take about 0.25 s. Calls are
# warmed up first, so compilation is never timed. The Q8_K rows include the
# activation quantization, which is O(ne0) and small next to the weight stream.

using Gesso

const I = Gesso.Inference
const BENCH_SAMPLES = 3
const BENCH_TARGET_S = 0.25

function _bench_median(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# Median per-call seconds over BENCH_SAMPLES samples.
function _bench_seconds(f)
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
    return _bench_median(samples)
end

function _bench_weights(ggml_type::Int32, ne0::Int, ne1::Int)
    bsz = I._iq_block_bytes(ggml_type)
    nb = div(ne0, I.QK_K)
    w = rand(UInt8, ne1 * nb * bsz)
    for blk in 0:(ne1 * nb - 1)
        d = Float16(0.01f0 + 0.02f0 * rand(Float32))
        bits = reinterpret(UInt16, d)
        w[blk * bsz + 1] = UInt8(bits & 0xff)
        w[blk * bsz + 2] = UInt8(bits >> 8)
    end
    return w
end

# Median GB/s of the host memory-read ceiling: a sum over `A`, split across
# `nt` tasks.
function _bench_host_read_gbps(A::Vector{UInt64}, nt::Int)
    n = length(A)
    out = zeros(UInt64, nt)
    f = () -> begin
        Threads.@threads for c in 1:nt
            j0 = div((c - 1) * n, nt) + 1
            j1 = div(c * n, nt)
            out[c] = sum(view(A, j0:j1))
        end
        return sum(out)
    end
    t = _bench_seconds(f)
    return (8 * n) / t / 1.0e9
end

# Weight copies rotate so each timed working set is at least BENCH_WS_BYTES.
# A single 37 MB matrix re-read every call would be served from the last-level
# cache and overstate DRAM streaming.
const BENCH_WS_BYTES = 256 * 2^20

function _bench_format(name::String, ggml_type::Int32, ne0::Int, ne1::Int, nt::Int)
    nbytes = ne1 * div(ne0, I.QK_K) * I._iq_block_bytes(ggml_type)
    ncopies = max(1, cld(BENCH_WS_BYTES, nbytes))
    Ws = [_bench_weights(ggml_type, ne0, ne1) for _ in 1:ncopies]
    x = rand(Float32, ne0) .- 0.5f0
    y = zeros(Float32, ne1)
    rows = Tuple{String,Float64}[]
    k = Ref(0)
    next_w = () -> begin
        k[] = mod(k[], ncopies) + 1
        return Ws[k[]]
    end

    t = _bench_seconds(() -> I._qgemv_bytes!(y, x, next_w(), ggml_type, ne0, ne1, 1.0f0, 0.0f0))
    push!(rows, ("F32 dequant _qgemv_bytes! (1T)", nbytes / t / 1.0e9))

    t = _bench_seconds(() -> I._qgemv_bytes_threaded!(y, x, next_w(), ggml_type, ne0, ne1, 1.0f0, 0.0f0))
    push!(rows, ("F32 dequant _qgemv_bytes_threaded! (MT)", nbytes / t / 1.0e9))

    t = _bench_seconds(() -> I._qgemv_q8k_f32!(y, x, next_w(), ggml_type, ne0, ne1; simd=false, threaded=false))
    push!(rows, ("Q8_K scalar reference (1T)", nbytes / t / 1.0e9))

    t = _bench_seconds(() -> I._qgemv_q8k_f32!(y, x, next_w(), ggml_type, ne0, ne1; simd=true, threaded=false))
    push!(rows, ("Q8_K AVX2 (1T)", nbytes / t / 1.0e9))

    t = _bench_seconds(() -> I._qgemv_q8k_f32!(y, x, next_w(), ggml_type, ne0, ne1; simd=true, threaded=true))
    push!(rows, ("Q8_K AVX2 threaded (MT)", nbytes / t / 1.0e9))

    return (; name, ne0, ne1, nbytes, rows)
end

function main()
    nt = Threads.nthreads()
    println("Gesso CPU quant GEMV benchmark")
    println("threads = ", nt, "   cpu = ", Sys.cpu_info()[1].model)
    println("avx2 kernels in use = ", I.q8k_simd_enabled())
    println("samples per figure = ", BENCH_SAMPLES, " (median), target ", BENCH_TARGET_S, " s per sample")
    println()

    shapes = (("expert 2048x512", 2048, 512), ("expert 512x2048", 512, 2048), ("dense 5120x13824", 5120, 13824))
    formats = (("IQ4_XS", I.GGML_TYPE_IQ4_XS), ("IQ2_XS", I.GGML_TYPE_IQ2_XS), ("IQ3_XXS", I.GGML_TYPE_IQ3_XXS))

    println(rpad("format", 9), rpad("shape (ne0 x ne1)", 22), rpad("weights MB", 12), rpad("path", 42), "GB/s")
    for (fname, ty) in formats, (sname, ne0, ne1) in shapes
        r = _bench_format(fname, ty, ne0, ne1, nt)
        for (path, gbps) in r.rows
            println(
                rpad(fname, 9),
                rpad(sname, 22),
                rpad(string(round(r.nbytes / 1.0e6; digits=2)), 12),
                rpad(path, 42),
                round(gbps; digits=2),
            )
        end
    end

    println()
    A = Vector{UInt64}(undef, 2^28)   # 2 GiB
    fill!(A, UInt64(0x0123456789abcdef))
    println("host memory read ceiling (2 GiB, sum):")
    println("  1T  GB/s = ", round(_bench_host_read_gbps(A, 1); digits=2))
    println("  MT  GB/s = ", round(_bench_host_read_gbps(A, nt); digits=2), "  (", nt, " tasks)")
    return nothing
end

main()
