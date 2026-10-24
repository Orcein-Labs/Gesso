# Exploration: software prefetch, non-temporal hints, and multi-row kernels for
# the IQ4_XS AVX2 GEMV. Stdlib only.
#
#   flock /tmp/gesso-cpu-bench.lock nice -n 5 julia --project=. -t 18 scripts/explore_cpu_prefetch.jl
#   flock /tmp/gesso-cpu-bench.lock nice -n 5 julia --project=. -t 1  scripts/explore_cpu_prefetch.jl
#
# Each variant is a copy of Gesso.Inference._dot_iq4_xs_q8k_avx2 with the same
# float operations in the same order, so every variant must reproduce the
# baseline bit for bit. The variants differ only in:
#   pf_t0 / pf_nta  an llvm.prefetch of the block D blocks ahead (locality 3 or 0)
#   row2 / row4     2 or 4 weight rows per call, sharing the activation loads
# Weights are synthetic (random bytes, finite f16 scales), dense 5120 x 13824
# shape, rotated over a 256 MiB working set.

using Gesso
using Base.Threads

const I = Gesso.Inference
const FMT = I.GGML_TYPE_IQ4_XS
const SAMPLES = parse(Int, get(ENV, "EXP_SAMPLES", "9"))
const TARGET_S = 0.25
const WS_BYTES = 256 * 2^20
const NE0 = 5120
const NE1 = 13824
const RB = (NE0 ÷ 256) * 136

# llvm.prefetch(ptr, rw=0 read, locality, cache=1 data).
@inline function pf_t0(p::Ptr{UInt8})
    Base.llvmcall(
        (
            """
 declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
 define void @pf_t0_body(ptr %p) alwaysinline {
   call void @llvm.prefetch.p0(ptr %p, i32 0, i32 3, i32 1)
   ret void
 }""",
            "pf_t0_body",
        ),
        Cvoid,
        Tuple{Ptr{UInt8}},
        p,
    )
end

@inline function pf_nta(p::Ptr{UInt8})
    Base.llvmcall(
        (
            """
 declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
 define void @pf_nta_body(ptr %p) alwaysinline {
   call void @llvm.prefetch.p0(ptr %p, i32 0, i32 0, i32 1)
   ret void
 }""",
            "pf_nta_body",
        ),
        Cvoid,
        Tuple{Ptr{UInt8}},
        p,
    )
end

# Baseline body, copied from _dot_iq4_xs_q8k_avx2 with a prefetch hook.
# hint: 0 none, 3 prefetch t0, 4 prefetch nta. dist: blocks ahead.
@noinline function dot_iq4_pf(
    w::AbstractVector{UInt8},
    woff::Int,
    x::I.Q8KVec,
    dist::Int,
    hint::Int,
)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    GC.@preserve w x begin
        pq = pointer(x.qs)
        pw = pointer(w)
        @inbounds for ib in 0:(nb-1)
            if dist > 0 && hint == 3
                pf_t0(pw + (woff + (ib + dist) * 136))
            elseif dist > 0 && hint == 4
                pf_nta(pw + (woff + (ib + dist) * 136))
            end
            b = woff + ib * 136
            d4d8 = I._ld_f16(w, b) * x.d[ib+1]
            h = I._ld_u16(w, b + 2)
            ao = ib * I.QK_K
            for s in 0:7
                sl = I._ld_u8(w, b + 4 + (s >> 1))
                ls =
                    (
                        Int32((sl >> (4 * (s & 1))) & 0x0f) |
                        (Int32((h >> (2 * s)) & 0x0003) << 4)
                    ) - Int32(32)
                dl = d4d8 * Float32(ls)
                t = I._iq4_sub_avx2(
                    I._Q8K_KV_IQ4NL_X2,
                    pointer(w, b + 9 + 16 * s),
                    pq + (ao + 32 * s),
                )
                sumf += dl * Float32(t)
            end
        end
    end
    return sumf
end

# Two rows per call. Row a at woff_a, row b at woff_b; each row's accumulation
# order is the baseline's, so results are bit-identical to two baseline calls.
@noinline function dot_iq4_2row(
    w::AbstractVector{UInt8},
    woff_a::Int,
    woff_b::Int,
    x::I.Q8KVec,
)
    nb = length(x.d)
    sa = 0.0f0
    sb = 0.0f0
    GC.@preserve w x begin
        pq = pointer(x.qs)
        @inbounds for ib in 0:(nb-1)
            ba = woff_a + ib * 136
            bb = woff_b + ib * 136
            da = I._ld_f16(w, ba) * x.d[ib+1]
            db = I._ld_f16(w, bb) * x.d[ib+1]
            ha = I._ld_u16(w, ba + 2)
            hb = I._ld_u16(w, bb + 2)
            ao = ib * I.QK_K
            for s in 0:7
                sla = I._ld_u8(w, ba + 4 + (s >> 1))
                slb = I._ld_u8(w, bb + 4 + (s >> 1))
                lsa =
                    (
                        Int32((sla >> (4 * (s & 1))) & 0x0f) |
                        (Int32((ha >> (2 * s)) & 0x0003) << 4)
                    ) - Int32(32)
                lsb =
                    (
                        Int32((slb >> (4 * (s & 1))) & 0x0f) |
                        (Int32((hb >> (2 * s)) & 0x0003) << 4)
                    ) - Int32(32)
                dla = da * Float32(lsa)
                dlb = db * Float32(lsb)
                ta = I._iq4_sub_avx2(
                    I._Q8K_KV_IQ4NL_X2,
                    pointer(w, ba + 9 + 16 * s),
                    pq + (ao + 32 * s),
                )
                tb = I._iq4_sub_avx2(
                    I._Q8K_KV_IQ4NL_X2,
                    pointer(w, bb + 9 + 16 * s),
                    pq + (ao + 32 * s),
                )
                sa += dla * Float32(ta)
                sb += dlb * Float32(tb)
            end
        end
    end
    return sa, sb
end

# Four rows per call, same scheme.
@noinline function dot_iq4_4row(
    w::AbstractVector{UInt8},
    o1::Int,
    o2::Int,
    o3::Int,
    o4::Int,
    x::I.Q8KVec,
)
    nb = length(x.d)
    s1 = 0.0f0
    s2 = 0.0f0
    s3 = 0.0f0
    s4 = 0.0f0
    GC.@preserve w x begin
        pq = pointer(x.qs)
        @inbounds for ib in 0:(nb-1)
            b1 = o1 + ib * 136
            b2 = o2 + ib * 136
            b3 = o3 + ib * 136
            b4 = o4 + ib * 136
            d1 = I._ld_f16(w, b1) * x.d[ib+1]
            d2 = I._ld_f16(w, b2) * x.d[ib+1]
            d3 = I._ld_f16(w, b3) * x.d[ib+1]
            d4 = I._ld_f16(w, b4) * x.d[ib+1]
            h1 = I._ld_u16(w, b1 + 2)
            h2 = I._ld_u16(w, b2 + 2)
            h3 = I._ld_u16(w, b3 + 2)
            h4 = I._ld_u16(w, b4 + 2)
            ao = ib * I.QK_K
            for s in 0:7
                q8 = pq + (ao + 32 * s)
                sh = 4 * (s & 1)
                hs = 2 * s
                l1 =
                    (
                        Int32((I._ld_u8(w, b1 + 4 + (s >> 1)) >> sh) & 0x0f) |
                        (Int32((h1 >> hs) & 0x0003) << 4)
                    ) - Int32(32)
                l2 =
                    (
                        Int32((I._ld_u8(w, b2 + 4 + (s >> 1)) >> sh) & 0x0f) |
                        (Int32((h2 >> hs) & 0x0003) << 4)
                    ) - Int32(32)
                l3 =
                    (
                        Int32((I._ld_u8(w, b3 + 4 + (s >> 1)) >> sh) & 0x0f) |
                        (Int32((h3 >> hs) & 0x0003) << 4)
                    ) - Int32(32)
                l4 =
                    (
                        Int32((I._ld_u8(w, b4 + 4 + (s >> 1)) >> sh) & 0x0f) |
                        (Int32((h4 >> hs) & 0x0003) << 4)
                    ) - Int32(32)
                t1 = I._iq4_sub_avx2(I._Q8K_KV_IQ4NL_X2, pointer(w, b1 + 9 + 16 * s), q8)
                t2 = I._iq4_sub_avx2(I._Q8K_KV_IQ4NL_X2, pointer(w, b2 + 9 + 16 * s), q8)
                t3 = I._iq4_sub_avx2(I._Q8K_KV_IQ4NL_X2, pointer(w, b3 + 9 + 16 * s), q8)
                t4 = I._iq4_sub_avx2(I._Q8K_KV_IQ4NL_X2, pointer(w, b4 + 9 + 16 * s), q8)
                s1 += (d1 * Float32(l1)) * Float32(t1)
                s2 += (d2 * Float32(l2)) * Float32(t2)
                s3 += (d3 * Float32(l3)) * Float32(t3)
                s4 += (d4 * Float32(l4)) * Float32(t4)
            end
        end
    end
    return s1, s2, s3, s4
end

# GEMV over rows with a per-row kernel. kind: :base, :pf_t0, :pf_nta, :row2, :row4.
function gemv_kind!(
    y::Vector{Float32},
    w::AbstractVector{UInt8},
    x::I.Q8KVec,
    kind::Symbol,
    dist::Int,
    nchunks::Int,
)
    ne1 = NE1
    if kind in (:base, :pf_t0, :pf_nta)
        hint = kind == :pf_t0 ? 3 : (kind == :pf_nta ? 4 : 0)
        d = kind == :base ? 0 : dist
        @threads :static for c in 1:nchunks
            j0 = div((c - 1) * ne1, nchunks) + 1
            j1 = div(c * ne1, nchunks)
            @inbounds for j in j0:j1
                y[j] = dot_iq4_pf(w, (j - 1) * RB, x, d, hint)
            end
        end
    elseif kind == :row2
        @threads :static for c in 1:nchunks
            j0 = div((c - 1) * ne1, nchunks) + 1
            j1 = div(c * ne1, nchunks)
            @inbounds for j in j0:2:(j1-1)
                a, b = dot_iq4_2row(w, (j - 1) * RB, j * RB, x)
                y[j] = a
                y[j+1] = b
            end
        end
    else
        @threads :static for c in 1:nchunks
            j0 = div((c - 1) * ne1, nchunks) + 1
            j1 = div(c * ne1, nchunks)
            @inbounds for j in j0:4:(j1-3)
                a, b, cc, dd =
                    dot_iq4_4row(w, (j - 1) * RB, j * RB, (j + 1) * RB, (j + 2) * RB, x)
                y[j] = a
                y[j+1] = b
                y[j+2] = cc
                y[j+3] = dd
            end
        end
    end
    return y
end

function bench_weights()
    bsz = I._iq_block_bytes(FMT)
    nb = div(NE0, I._q8k_block_values(FMT))
    doff = I._q8k_d_offset(FMT)
    w = rand(UInt8, NE1 * nb * bsz)
    for blk in 0:(NE1*nb-1)
        bits = reinterpret(UInt16, Float16(0.01f0 + 0.02f0 * rand(Float32)))
        w[blk*bsz+doff+1] = UInt8(bits & 0xff)
        w[blk*bsz+doff+2] = UInt8(bits >> 8)
    end
    return w
end

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

function summarize(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    m = sum(v) / n
    med = isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
    return (; med, mean=m, sd=sqrt(sum((v .- m) .^ 2) / (n - 1)), min=s[1], max=s[end])
end

function run_variant(label, kind, dist, nchunks, Ws, xq, yref)
    nbytes = NE1 * RB
    ncopies = length(Ws)
    k = Ref(0)
    y = zeros(Float32, NE1)
    nextw = () -> begin
        k[] = mod(k[], ncopies) + 1
        return Ws[k[]]
    end
    gemv_kind!(y, Ws[1], xq, kind, dist, nchunks)
    exact = y == yref
    ts = time_samples(() -> gemv_kind!(y, nextw(), xq, kind, dist, nchunks))
    s = summarize(nbytes ./ ts ./ 1.0e9)
    println(
        rpad(label, 22),
        " threads=",
        nchunks,
        "  GB/s median ",
        round(s.med; digits=2),
        "  mean ",
        round(s.mean; digits=2),
        "  sd ",
        round(s.sd; digits=2),
        "  min ",
        round(s.min; digits=2),
        "  max ",
        round(s.max; digits=2),
        "  bitexact ",
        exact,
    )
    return s
end

function main()
    nt = nthreads()
    println(
        "threads=",
        nt,
        "  samples=",
        SAMPLES,
        "  shape=",
        NE1,
        "x",
        NE0,
        "  working set MiB=",
        WS_BYTES ÷ 2^20,
    )
    ncopies = max(1, cld(WS_BYTES, NE1 * RB))
    Ws = [bench_weights() for _ in 1:ncopies]
    x = rand(Float32, NE0) .- 0.5f0
    xq = I._q8k_quantize_act(FMT, x)
    yref = zeros(Float32, NE1)
    gemv_kind!(yref, Ws[1], xq, :base, 0, 1)
    nchunks = nt
    run_variant("baseline", :base, 0, nchunks, Ws, xq, yref)
    run_variant("pf_t0 dist=4", :pf_t0, 4, nchunks, Ws, xq, yref)
    run_variant("pf_t0 dist=16", :pf_t0, 16, nchunks, Ws, xq, yref)
    run_variant("pf_nta dist=8", :pf_nta, 8, nchunks, Ws, xq, yref)
    run_variant("row2", :row2, 0, nchunks, Ws, xq, yref)
    run_variant("row4", :row4, 0, nchunks, Ws, xq, yref)
    run_variant("baseline (repeat)", :base, 0, nchunks, Ws, xq, yref)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
