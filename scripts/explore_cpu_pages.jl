# Exploration: page size and TLB effects on the IQ4_XS GEMV, using real tensors
# from a GGUF file. Stdlib only.
#
#   EXP_GGUF=/path/to/model.gguf flock /tmp/gesso-cpu-bench.lock nice -n 5 \
#       julia --project=. -t 18 scripts/explore_cpu_pages.jl
#
# Three placements of the same IQ4_XS bytes, all read by the production
# threaded GEMV (Gesso.Inference._qgemv_q8k_threaded!):
#   mmap          the GGUF file mapping (page cache, 4 KiB pages), as Gesso loads it
#   anon_default  a Vector{UInt8} copy, default THP policy (this host: "always")
#   anon_madv     the same copy, madvise(MADV_HUGEPAGE) before first touch
#   anon_4k       the same copy with THP disabled for this process only
#                 (prctl PR_SET_THP_DISABLE), the 4 KiB baseline
#
# The working set is at least 512 MiB across rotated real tensors, so the
# matrices are not served from the last-level cache. Per-process settings only;
# nothing system-wide is changed.

using Gesso
using Mmap
using Base.Threads

const I = Gesso.Inference
const FMT = I.GGML_TYPE_IQ4_XS
const SAMPLES = parse(Int, get(ENV, "EXP_SAMPLES", "9"))
const TARGET_S = 0.25
const WS_MIN = 512 * 2^20
const GGUF = get(ENV, "EXP_GGUF", joinpath(homedir(), ".cache", "gesso",
    "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf"))

function time_samples(f)
    f()
    t1 = @elapsed f()
    reps = clamp(ceil(Int, TARGET_S / max(t1, 1.0e-9)), 1, 10_000_000)
    out = Float64[]
    for _ in 1:SAMPLES
        t = @elapsed for _ in 1:reps
            f()
        end
        push!(out, t / reps)
    end
    return out
end

function stats(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    m = sum(v) / n
    med = isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
    sd = sqrt(sum((v .- m) .^ 2) / (n - 1))
    return (; med, mean=m, sd, min=s[1], max=s[end])
end

# AnonHugePages for this process, kB, from smaps_rollup.
function anon_huge_kb()::Int
    for line in eachline("/proc/self/smaps_rollup")
        startswith(line, "AnonHugePages:") && return parse(Int, split(line)[2])
    end
    return -1
end

# Pick IQ4_XS 2-D tensors with the most common ne0, until WS_MIN is covered.
function pick_tensors(hdr)
    cands = [t for t in hdr.tensors if t.ggml_type == FMT && length(t.dims) == 2]
    isempty(cands) && error("no IQ4_XS 2-D tensors in $GGUF")
    counts = Dict{Int,Int}()
    for t in cands
        counts[t.dims[1]] = get(counts, t.dims[1], 0) + 1
    end
    ne0 = argmax(counts)
    sel = [t for t in cands if t.dims[1] == ne0]
    out = []
    total = 0
    for t in sel
        push!(out, t)
        total += I._qrow_bytes(FMT, t.dims[1]) * t.dims[2]
        total >= WS_MIN && break
    end
    return ne0, out
end

# Rotating GEMV over (bytes-view, ne1) pairs that share ne0.
function rotating_gemv_samples(views::Vector, ne0::Int, ne1s::Vector{Int})
    xq = I._q8k_quantize_act(FMT, rand(Float32, ne0) .- 0.5f0)
    ys = [zeros(Float32, n) for n in ne1s]
    k = Ref(0)
    n = length(views)
    f = () -> begin
        k[] = mod(k[], n) + 1
        i = k[]
        I._qgemv_q8k_threaded!(ys[i], xq, views[i], FMT, ne0, ne1s[i])
        return nothing
    end
    return time_samples(f)
end

function report(label::String, ts::Vector{Float64}, nbytes_total::Int, ne1s::Vector{Int}, ne0::Int)
    # Bytes per call vary by matrix; rate is the per-call bytes over per-call time, median over samples.
    # Per-sample times are for one call, averaged over the rotation.
    mean_bytes = nbytes_total / length(ne1s)
    s = stats(mean_bytes ./ ts ./ 1.0e9)
    println(rpad(label, 16), " GB/s median ", round(s.med; digits=2), "  mean ", round(s.mean; digits=2),
        "  sd ", round(s.sd; digits=2), "  min ", round(s.min; digits=2), "  max ", round(s.max; digits=2),
        "  call_ms_med ", round(stats(ts).med * 1.0e3; digits=3), "  anon_huge_kB ", anon_huge_kb())
    return s
end

function main()
    println("gguf = ", basename(GGUF))
    println("threads = ", nthreads(), "   thp enabled = ", strip(read("/sys/kernel/mm/transparent_hugepage/enabled", String)))
    println("anon_huge_kB at start = ", anon_huge_kb())
    hdr = I.gguf_header(GGUF)
    ne0, tensors = pick_tensors(hdr)
    ne1s = [t.dims[2] for t in tensors]
    sizes = [I._qrow_bytes(FMT, ne0) * n for n in ne1s]
    total = sum(sizes)
    println("tensors = ", length(tensors), "  ne0 = ", ne0, "  working set MiB = ", round(total / 2^20; digits=1))
    offs = [Int(hdr.data_start) + Int(t.offset) for t in tensors]

    # mmap: views into the file mapping, pre-touched so the timing excludes disk reads.
    mm_views = [view(hdr.data, o + 1:o + s) for (o, s) in zip(offs, sizes)]
    for v in mm_views
        sum(view(v, 1:64:length(v)))  # touch one byte per 64 B line
    end
    ts = rotating_gemv_samples(mm_views, ne0, ne1s)
    report("mmap", ts, total, ne1s, ne0)

    # Anonymous copies. Each variant allocates, optionally sets the policy, then
    # the first touch (the copy) faults the pages in.
    function anon_variant(label::String)
        buf = Vector{UInt8}(undef, total)
        if label == "anon_madv"
            p0 = pointer(buf)
            a = (UInt(p0) + 4095) & ~UInt(4095)
            b = (UInt(p0) + total) & ~UInt(4095)
            ret = ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), Ptr{Cvoid}(a), Csize_t(b - a), 14)
            println("  madvise(MADV_HUGEPAGE) on anon buffer returned ", ret)
        end
        off = 0
        for (o, s) in zip(offs, sizes)
            copyto!(buf, off + 1, hdr.data, o + 1, s)
            off += s
        end
        # Verify byte identity with the mapping on the first matrix.
        ok = view(buf, 1:sizes[1]) == view(hdr.data, offs[1] + 1:offs[1] + sizes[1])
        println("  ", label, " bytes match mapping (first matrix): ", ok)
        views = Vector{SubArray{UInt8,1,Vector{UInt8},Tuple{UnitRange{Int}},true}}()
        o2 = 0
        for s in sizes
            push!(views, view(buf, o2 + 1:o2 + s))
            o2 += s
        end
        return views, buf
    end

    keep = Any[]
    v_default, b1 = anon_variant("anon_default")
    push!(keep, b1)
    ts = rotating_gemv_samples(v_default, ne0, ne1s)
    report("anon_default", ts, total, ne1s, ne0)

    v_madv, b2 = anon_variant("anon_madv")
    push!(keep, b2)
    ts = rotating_gemv_samples(v_madv, ne0, ne1s)
    report("anon_madv", ts, total, ne1s, ne0)

    ret = ccall(:prctl, Cint, (Cint, Culong, Culong, Culong, Culong), 41, 1, 0, 0, 0)
    println("prctl(PR_SET_THP_DISABLE) returned ", ret)
    v_4k, b3 = anon_variant("anon_4k")
    push!(keep, b3)
    ts = rotating_gemv_samples(v_4k, ne0, ne1s)
    report("anon_4k", ts, total, ne1s, ne0)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
