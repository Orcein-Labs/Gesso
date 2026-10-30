# CPU prefill GEMM microbench: _quant_matmul_multi! on random valid quantized
# weights, warm, median of 5 runs, with GFLOPS (2 * rows * ne0 * seq / time).
#
#   julia -t N --project=. scripts/bench_prefill_gemm.jl [seq ...]
#
# Env: GESSO_BENCH_GGUF (colon-separated GGUF paths) adds the first ffn_gate
# shape of each file, and for MoE files its per-expert shape, at that tensor's
# own type. Without it only the built-in shapes run.
using Gesso
using Printf
using Random: Xoshiro, rand

const INF = Gesso.Inference

function scale_offsets(t)
    t == INF.GGML_TYPE_Q6_K && return (208,)
    t in (INF.GGML_TYPE_Q4_K, INF.GGML_TYPE_Q5_K) && return (0, 2)
    return (0,)
end

# Arbitrary bytes with finite f16 super-scales (random f16 bits are often NaN).
function random_matrix(rng, ty::Int32, ne0::Int, nrows::Int)
    rb = INF._qrow_bytes(ty, ne0)
    bsz = INF._iq_block_bytes(ty)
    data = rand(rng, UInt8, nrows * rb)
    for blk in 0:(length(data)÷bsz-1), off in scale_offsets(ty)
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk*bsz+off+1] = UInt8(bits & 0xff)
        data[blk*bsz+off+2] = UInt8(bits >> 8)
    end
    return INF.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(INF.QK_K))
end

function bench(name, ty, ne0, nrows, seq; reps=5)
    rng = Xoshiro(7)
    q = random_matrix(rng, ty, ne0, nrows)
    x = randn(rng, Float32, seq, ne0)
    dst = zeros(Float32, seq, nrows)
    INF._quant_matmul_multi!(dst, x, q)
    ts = Float64[]
    for _ in 1:reps
        push!(ts, @elapsed INF._quant_matmul_multi!(dst, x, q))
    end
    t = sort(ts)[div(reps + 1, 2)]
    gf = 2.0 * nrows * ne0 * seq / t / 1e9
    @printf("PGEMM %-22s ne0=%-5d rows=%-6d seq=%-4d ms=%9.2f GFLOPS=%7.1f\n", name, ne0, nrows, seq, t * 1e3, gf)
    return gf
end

# (name, type, ne0, rows) of the first dense ffn_gate and, for MoE files, the
# per-expert shape.
function gguf_shapes(path)
    out = Tuple{String, Int32, Int, Int}[]
    isfile(path) || return out
    h = INF.gguf_header(path)
    base = replace(basename(path), r"\.gguf$" => "")
    for t in h.tensors
        occursin(r"^blk\.\d+\.ffn_gate(_exps)?\.weight$", t.name) || continue
        length(t.dims) >= 2 || continue
        push!(out, ("$(first(base, 18)):$(t.name)", t.ggml_type, t.dims[1], t.dims[2]))
        break
    end
    return out
end

seqs = isempty(ARGS) ? [64, 512] : parse.(Int, ARGS)
shapes = [
    ("qwen2b-ffn_gate-Q4_K", INF.GGML_TYPE_Q4_K, 2048, 6144),
    ("qwen2b-ffn_gate-Q6_K", INF.GGML_TYPE_Q6_K, 2048, 6144),
    ("lfm-expert-Q4_K", INF.GGML_TYPE_Q4_K, 2048, 1792),
    ("lfm-expert-Q5_K", INF.GGML_TYPE_Q5_K, 2048, 1792),
]
for p in split(get(ENV, "GESSO_BENCH_GGUF", ""), ':'; keepempty=false)
    append!(shapes, gguf_shapes(p))
end
println("PGEMM threads=$(Threads.nthreads()) simd=$(INF.q8k_simd_enabled())")
for (name, ty, ne0, nrows) in shapes, seq in seqs
    bench(name, ty, ne0, nrows, seq)
end
