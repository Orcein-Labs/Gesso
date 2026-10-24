# Ornith-1.5-35B-A3B decode stage profile (GESSO_PROFILE stage profiler).
#
#   GESSO_ORNITH35_GGUF=/path/to/Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   GESSO_BACKEND=lava|cuda|cpu \
#   julia --project=benchmark -t 16 scripts/profile_35b.jl
#
# GESSO_BACKEND defaults to lava; CUDA is loaded only when it is asked for.
#
# Optional env: GESSO_PROFILE_TOKENS (decode tokens to measure, default 8),
# GESSO_WALL_TOKENS (unprofiled decode tokens timed, default 16),
# GESSO_PROFILE_CTX (prompt length, default 1; longer prompts are real text
# from GESSO_PROFILE_TEXT, repeated, so decode runs against a filled KV cache).
# The engine switches are read as usual: GESSO_GPU_WEIGHT_DTYPE (f16 default),
# GESSO_MOE_HOST_EXPERTS (0 default).
#
# Three phases, one process:
#   1. JIT warm: two tokens, profiler off (compiles the decode path).
#   2. Cold token log: N+1 tokens profiled right after the JIT warm, so the
#      first touches of the mmap'd expert pages happen inside the log.
#   3. Warm table: profiled 1-token run and profiled (N+1)-token run; the
#      difference is exactly N decode tokens (prefill and the head-only
#      token cancel). Prefill is never inside a token window anyway.
#
# Prints a RECEIPT line, the cold token log, and the warm per-token table.
# Exit 1 if the greedy ids change under profiling, or if the one-token prompt
# drifts from the reference sequence (teacher-forced, see ForcedIds).

using Gesso

const GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")
isfile(GGUF) || error("profile_35b: set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF file")
const GGML_DIR = get(ENV, "GESSO_GGML_DIR", "")
isdir(GGML_DIR) ||
    error("profile_35b: set GESSO_GGML_DIR to the llama.cpp build/bin directory")
const EXPECTED = [9419, 11, 271, 40, 2688, 4313, 310, 958, 279]
const N = parse(Int, get(ENV, "GESSO_PROFILE_TOKENS", "8"))
N >= 1 || error("profile_35b: GESSO_PROFILE_TOKENS must be >= 1")
const WALL = parse(Int, get(ENV, "GESSO_WALL_TOKENS", "16"))
const CTX = parse(Int, get(ENV, "GESSO_PROFILE_CTX", "1"))
CTX >= 1 || error("profile_35b: GESSO_PROFILE_CTX must be >= 1")
const TEXT = get(
    ENV,
    "GESSO_PROFILE_TEXT",
    joinpath(@__DIR__, "..", "benchmark", "corpus-pride-and-prejudice.txt"),
)

which = get(ENV, "GESSO_BACKEND", "lava")
if which == "lava"
    using Lava
    backend = Gesso.LavaBackend()
elseif which == "cuda"
    using CUDA
    backend = Gesso.CUDABackend()
elseif which == "cpu"
    backend = Gesso.CPUBackend()
else
    error("profile_35b: GESSO_BACKEND must be lava, cuda or cpu, got '$which'")
end

const INF = Gesso.Inference

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
    dev = which == "cpu" ? tensors : Gesso.to_device(backend, tensors)
end

# EXPECTED holds for the one-token prompt only.
prompt = if CTX == 1
    [9419]
else
    corpus = INF.encode(INF.load_gguf_tokenizer(GGUF), read(TEXT, String); parse_special=false)
    [corpus[mod1(i, length(corpus))] for i in 1:CTX]
end

gen(n) = Gesso.reference_generate(
    model,
    dev,
    prompt;
    max_new_tokens=n,
    eps=spec.rms_norm_eps,
    backend=backend,
)

dtype = get(ENV, "GESSO_GPU_WEIGHT_DTYPE", "f16")
moe_host = get(ENV, "GESSO_MOE_HOST_EXPERTS", "0")

# The reference sequence is checked teacher-forced, not by greedy ids: step 2
# is a near-tie (271 over 353 by 0.016 logits on the byte-wise Q8_0 kernel),
# and an equally exact kernel with another summation order flips it, which
# shifts every later id. Feeding the reference token at each step and
# requiring it to sit within IDS_TOL of the top logit passes reordering noise
# (0.0006) and fails a Q8_0 block-scale indexing bug (gaps up to 3.7). It
# checks the top token, not calibration: every Q8_0 scale off by 3% stays
# under 0.006 and passes.
const IDS_TOL = parse(Float64, get(ENV, "GESSO_IDS_TOL", "0.05"))
mutable struct ForcedIds
    ids::Vector{Int}
    k::Int
    gaps::Vector{Float32}
end
function INF._sample_id(s::ForcedIds, logits_row)
    row = vec(Array(logits_row))
    s.k += 1
    id = s.ids[s.k]
    push!(s.gaps, maximum(row) - row[id+1])
    return id
end

# Inside a function so the runs share one world. On Julia 1.12 each new
# top-level global bumps the world counter, and Lava drops every launch plan
# compiled in an older world, so a global assigned between two runs made the
# second one recompile all of its kernels.
function measure()
    INF.profile_disable!()
    gen(2)

    INF.profile_enable!()
    ids_cold = gen(N + 1)
    cold = INF.profile_snapshot()

    INF.profile_enable!()
    gen(1)
    a = INF.profile_snapshot()
    INF.profile_reset!()
    hits0 = INF.expert_hit_summary(dev.blocks)
    ids_warm = gen(N + 1)
    hits1 = INF.expert_hit_summary(dev.blocks)
    b = INF.profile_snapshot()
    INF.profile_disable!()
    # The profiled table syncs the device around every stage, so it cannot
    # show work the backend overlaps; this unprofiled pair can. The first pass
    # reaches positions the runs above never did, so it is not timed.
    gen(WALL + 1)
    t1 = @elapsed gen(1)
    ids_wall = nothing
    tw = @elapsed (ids_wall = gen(WALL + 1))
    wall_ms = 1e3 * (tw - t1) / max(1, length(ids_wall) - 1)
    forced = ForcedIds(EXPECTED[2:end], 0, Float32[])
    CTX == 1 && Gesso.reference_generate(
        model,
        dev,
        prompt;
        max_new_tokens=length(EXPECTED) - 1,
        eps=spec.rms_norm_eps,
        backend=backend,
        sampler=forced,
    )
    return (;
        forced,
        ids_cold,
        cold,
        a,
        hits0,
        ids_warm,
        hits1,
        b,
        wall_ms,
        nwall=length(ids_wall) - 1,
    )
end
(; forced, ids_cold, cold, a, hits0, ids_warm, hits1, b, wall_ms, nwall) = measure()

# per-layer residency and hit counts for the warm run only
let H = 0, M = 0
    for (x0, x1) in zip(hits0, hits1)
        h = x1.hits - x0.hits
        m = x1.misses - x0.misses
        H += h
        M += m
        println(
            "expert layer=$(x1.layer) slots=$(x1.slots) hits=$h misses=$m rate=$(round(h / max(1, h + m); digits=3))",
        )
    end
    println(
        "RECEIPT expert_hits warm_hits=$H warm_misses=$M rate=$(round(H / max(1, H + M); digits=4))",
    )
end

ids_warm == ids_cold ||
    error("profile_35b: ids differ between the cold and warm profiled runs")
checked = !isempty(forced.gaps)
ok = !checked || maximum(forced.gaps) <= IDS_TOL
ids_ok = checked ? string(ok) : "unchecked"
greedy =
    length(ids_cold) >= length(EXPECTED) ? ids_cold[1:length(EXPECTED)] == EXPECTED :
    missing

d = INF.profile_delta(b, a)
d.tokens == N ||
    error("profile_35b: measured $(d.tokens) decode tokens, expected $N (EOS reached?)")

println(
    "RECEIPT profile_35b backend=$which threads=$(Threads.nthreads()) ",
    "dtype=$dtype moe_host_experts=$moe_host ctx=$CTX decode_tokens=$N ",
    "ids_ok=$ids_ok greedy_match=$greedy load_s=$(round(t_load; digits=1))",
)
println(
    "RECEIPT unprofiled decode_tokens=$nwall ms_per_token=$(round(wall_ms; digits=2)) ",
    "tok_s=$(round(1e3 / wall_ms; digits=2))",
)
println()
println("== cold token log (JIT warm, then first profiled run in this process) ==")
INF.print_token_log(stdout, cold)
println()
println("== warm per-token table (difference of profiled runs, $N decode tokens) ==")
INF.print_decode_profile(stdout, d; label="backend=$which dtype=$dtype moe_host=$moe_host")
checked && println(
    "RECEIPT forced_ids tol=$IDS_TOL max_gap=$(maximum(forced.gaps)) ",
    "gaps=$(round.(forced.gaps; digits=4))",
)
ok || (println("ids got  $ids_cold\nids want $EXPECTED"); exit(1))
