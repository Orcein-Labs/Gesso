# Gesso benchmark harness (Gesso_Stack.md §LXXIII; §XXXIII integrity rules).
#
# Conventions:
#
#   §XXXIII  No timing is published from a run that included compilation
#            without saying so. Suites warm up before measuring and report
#            samples, not single runs. This harness compiles each entry
#            explicitly before the BenchmarkTools run (which additionally
#            tunes), and never gates on absolute numbers — CI only checks
#            the harness runs and persists.
#
#   §XLIX    Gesso must explain performance. Results print structured facts —
#            name, samples, median, allocations — and persist them with
#            environment metadata (host, arch, threads, commit, dirty state).
#
#   North Star §24: results persist to benchmark/results/ so a regression
#   corpus accrues from the very first sprint. You cannot retroactively
#   baseline. Result files are append-only and may mix row schemas — group
#   by the `schema` column; never silently reinterpret old rows (§LXIX).
#
#   §LXXII   No performance claims beyond what a run in this repo produced.
#            The harness measures; it does not editorialize. The only
#            standing claims are recorded per-benchmark as `bench_note`s.
#
# This is infrastructure, not a performance claim: today it establishes that
# the measurement machinery works and accrues a corpus for future phases.
#
# Run with:
#   make bench   (or: julia --project=benchmark benchmark/runbenchmarks.jl)

using Gesso
using BenchmarkTools
using Dates
using Sockets

# The schema tag is IMPORTED, not hardcoded: the persisted format and the
# schema constant in src/versions.jl cannot drift apart silently (§LXIX).
const BENCH_RESULT_SCHEMA = string(Gesso.BENCH_RESULT_SCHEMA_VERSION)

const RESULTS = []

function record(name; samples, median_ns, mean_ns, min_ns, allocs, bytes, note)
    push!(
        RESULTS,
        (
            name=name,
            samples=samples,
            median_ns=median_ns,
            mean_ns=mean_ns,
            min_ns=min_ns,
            allocs=allocs,
            bytes=bytes,
            note=note,
        ),
    )
    println(
        rpad(name, 34),
        lpad(string(round(median_ns / 1_000; digits=2)), 10),
        " µs",
        lpad(string(allocs), 6),
        " allocs   (",
        samples,
        " samples)",
        note == "" ? "" : "   # " * note,
    )
end

function git_meta()
    try
        head = strip(read(`git log -1 --format=%h`, String))
        dirty = !isempty(strip(read(`git status --porcelain`, String)))
        return head, dirty ? "true" : "false"
    catch
        return "unknown", "unknown"   # e.g. bench from a tarball; say so
    end
end

suite = Dict{String, Tuple{Function, String}}()   # name => (f, bench_note)

# Constructing backend tags must be free — planners probe constantly (§XX).
suite["backend_tag_construction"] =
    (() -> Gesso.CPUBackend(), "§XX: tag construction must stay free")

suite["capability_probe"] = (
    () -> Gesso.supports(Gesso.CPUBackend(), :some_capability),
    "§XX: capability probing is always safe, never throws",
)

# Structured event emission (§XLII) must be cheap enough that code never
# avoids logging to save time.
suite["structured_log_event"] = (
    () -> Gesso.glog(devnull, Gesso.Log.LOG_INFO, :bench_event; op=:noop, backend=:cpu),
    "§XLII: logging must never be the reason to skip logging",
)

# Receipt emission (§XLII) is on the hot path of every audited action; it
# must stay cheap and must never throw.
suite["receipt_emit_inmemory"] = (
    () -> begin
        sink = Gesso.InMemorySink(64)
        Gesso.emit!(sink, Gesso.new_receipt(task=:bench))
    end,
    "§XLII: audited actions pay for receipts; keep it negligible",
)

# §LXX: the explicit-failure path must be cheap enough to be used everywhere.
suite["lowering_not_implemented_throw"] = (
    () -> begin
        try
            Gesso.rmsnorm!(Gesso.CPUBackend(), nothing)
            error("expected LoweringNotImplemented")
        catch e
            e isa Gesso.LoweringNotImplemented || rethrow()
        end
    end,
    "§LXX: explicit failure must be affordable",
)

# --- Phase 4/5 shared micro model (public vocabulary, deterministic filler) --
# config_to_model + materialize_llama with formula filler — the bench env has
# no JSON dependency, and a formula fixture is bit-reproducible across runs
# (the regression corpus must be comparable, §XLIX). Correctness parity for
# this model shape is the test suite's job; the rows below are measurements,
# not claims (§LXXII).
_bench_cfg = (
    model_type="llama",
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=64,
    vocab_size=32,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    tie_word_embeddings=true,
)
_bench_model = Gesso.config_to_model(_bench_cfg)
_filler(shape) = reshape([(i % 13 - 6) * 0.01 for i in 1:prod(shape)], shape)
_bench_tensors = Dict{String, Array{Float64}}(
    "model.embed_tokens.weight" =>
        _filler((_bench_cfg.vocab_size, _bench_cfg.hidden_size)),
    "model.norm.weight" => _filler((_bench_cfg.hidden_size,)),
)
for i in 1:_bench_cfg.num_hidden_layers
    _bench_tensors["model.layers.$i.self_attn.q_proj.weight"] =
        _filler((_bench_cfg.hidden_size, _bench_cfg.hidden_size))
    _bench_tensors["model.layers.$i.self_attn.k_proj.weight"] =
        _filler((_bench_cfg.num_key_value_heads * 8, _bench_cfg.hidden_size))
    _bench_tensors["model.layers.$i.self_attn.v_proj.weight"] =
        _filler((_bench_cfg.num_key_value_heads * 8, _bench_cfg.hidden_size))
    _bench_tensors["model.layers.$i.self_attn.o_proj.weight"] =
        _filler((_bench_cfg.hidden_size, _bench_cfg.hidden_size))
    _bench_tensors["model.layers.$i.mlp.gate_proj.weight"] =
        _filler((_bench_cfg.intermediate_size, _bench_cfg.hidden_size))
    _bench_tensors["model.layers.$i.mlp.up_proj.weight"] =
        _filler((_bench_cfg.intermediate_size, _bench_cfg.hidden_size))
    _bench_tensors["model.layers.$i.mlp.down_proj.weight"] =
        _filler((_bench_cfg.hidden_size, _bench_cfg.intermediate_size))
    _bench_tensors["model.layers.$i.input_layernorm.weight"] =
        _filler((_bench_cfg.hidden_size,))
    _bench_tensors["model.layers.$i.post_attention_layernorm.weight"] =
        _filler((_bench_cfg.hidden_size,))
end
_bench_ts = Gesso.materialize_llama(_bench_model, _bench_tensors, _bench_cfg)
_bench_tokens = [0, 1, 2]

# --- Phase 5 (§LXXVIII item D): llama_micro Session generate row (CPU) ------
# Measures the ENGINE: Session + paged KV over CPU, tokens [0,1,2] + 3 greedy
# steps. page_size=4 so the row exercises page-boundary appends.
_bench_session5() = Gesso.Session(
    _bench_model,
    _bench_ts;
    context_length=16,
    eos_token_id=0,
    page_size=4,
    eps=_bench_cfg.rms_norm_eps,
    theta=_bench_cfg.rope_theta,
)
suite["llama_micro_session_generate_cpu"] = (
    () -> Gesso.generate(_bench_session5(), _bench_tokens; max_new_tokens=3),
    "§LXXVIII item D: CPU Session generate over the paged KV manager (page_size=4), tokens [0,1,2] + 3 greedy steps",
)

# --- Phase 6 (§LXXIX item C): engine TTFT / decode attribution rows (CPU) ---
# Same model, same prompt, ONE warmup generate (compile) before any timed
# sample — the harness then samples; every row here is post-warmup (§XXXIII).
# TTFT and decode are DISTINCT rows so neither subsumes the other; read the
# two together, never averaged.
_bench_ttft_session6() = Gesso.Session(
    _bench_model,
    _bench_ts;
    context_length=16,
    eos_token_id=0,
    page_size=4,
    eps=_bench_cfg.rms_norm_eps,
    theta=_bench_cfg.rope_theta,
)
# warmup: one full generate OUTSIDE any timed region (compiles every path)
Gesso.generate(_bench_ttft_session6(), _bench_tokens; max_new_tokens=3)

# TTFT row: prefill + exactly one decode step (time-to-first-token, engine
# receipt's ttft_ns definition)
_bench_ttft_probe() = Gesso.generate(
    Gesso.Session(
        _bench_model,
        _bench_ts;
        context_length=16,
        eos_token_id=0,
        page_size=4,
        eps=_bench_cfg.rms_norm_eps,
        theta=_bench_cfg.rope_theta,
    ),
    _bench_tokens;
    max_new_tokens=1,
)
suite["session_ttft_1tok_cpu"] = (
    _bench_ttft_probe,
    "§LXXIX item C: post-warmup — CPU Session prefill + 1 greedy step (time-to-first-token)",
)

# per-generate row: full 8-step decode on the same model/prompt (median over
# the whole generate — decode marginal cost = this row minus the TTFT row,
# both post-warmup, same measurement discipline)
_bench_gen8_probe() = Gesso.generate(
    Gesso.Session(
        _bench_model,
        _bench_ts;
        context_length=16,
        eos_token_id=0,
        page_size=4,
        eps=_bench_cfg.rms_norm_eps,
        theta=_bench_cfg.rope_theta,
    ),
    _bench_tokens;
    max_new_tokens=8,
)
suite["session_generate_8tok_cpu"] = (
    _bench_gen8_probe,
    "§LXXIX item C: post-warmup — CPU Session generate, 8 greedy steps over the paged KV manager",
)

# --- Phase 7 (§LXXX item C): unique-vs-summed KV byte rows (CPU) -------------
# The §LIV win as a NUMBER OF BYTES, not a slogan: Profiling.unique_kv_bytes
# counts live page storage once per distinct array, so a declared fork alias
# (prefill! + fork, §LXXX) costs one session's KV, not two. Byte rows are
# deterministic post-warmup — there is no sample to warm, only the engine
# paths to compile (§XXXIII): one untimed prefill here, fresh sessions inside
# each probe below. No CUDA row: the CPU pair proves the win (goal §C).
_bench_share_session() = Gesso.Session(
    _bench_model,
    _bench_ts;
    context_length=16,
    eos_token_id=0,
    page_size=4,
    eps=_bench_cfg.rms_norm_eps,
    theta=_bench_cfg.rope_theta,
)
Gesso.prefill!(_bench_share_session(), _bench_tokens)   # warmup: compile (§XXXIII)

# --- Phase 4 (§LXXVII item D): gated micro-llama CUDA prefill probe ---------
# benchmark/Project.toml declares CUDA (the bench env, never core §VII) so
# the GessoCUDAExt extension can trigger here. Without a functional device
# the entry is never added — CPU-only machines accrue no row and CI never
# requires a device (§LXXVII skip law; the run output names the skip).
const CUDA_BENCH = let
    ok = true
    try
        @eval Main using CUDA
        ok = CUDA.functional()
    catch
        ok = false
    end
    ok
end

if CUDA_BENCH
    @eval Main using CUDA   # in scope for the probe body

    _bench_cuda = Gesso.CUDABackend()
    _bench_gpu_ts = Gesso.to_device(_bench_cuda, _bench_ts)
    _bench_probe() = Array(
        Gesso.reference_prefill(
            _bench_model,
            _bench_gpu_ts,
            _bench_tokens;
            backend=_bench_cuda,
            eps=_bench_cfg.rms_norm_eps,
            theta=_bench_cfg.rope_theta,
        ),
    )   # Array() reads back to host (implicit synchronize, §LXXVII)
    suite["micro_llama_cuda_prefill_012"] = (
        _bench_probe,
        "§LXXVII item D: CUDA F32 prefill + host readback, tokens [0,1,2], llama_micro shape",
    )

    # --- Phase 6 (§LXXIX item C): CUDA Session generate row (gated) ----------
    # Session over device pages; one warmup generate, then the harness
    # samples (post-warmup, §XXXIII). Array() readback implies CUDA.
    # synchronize before ids reach host (§LXXVII).
    _bench_gpu_session6() = Gesso.Session(
        _bench_model,
        _bench_gpu_ts;
        backend=_bench_cuda,
        context_length=16,
        eos_token_id=0,
        page_size=4,
        eps=_bench_cfg.rms_norm_eps,
        theta=_bench_cfg.rope_theta,
    )
    Gesso.generate(_bench_gpu_session6(), _bench_tokens; max_new_tokens=3)   # warmup
    suite["session_generate_3tok_cuda"] = (
        () -> Gesso.generate(_bench_gpu_session6(), _bench_tokens; max_new_tokens=3),
        "§LXXIX item C: post-warmup — CUDA Session generate, 3 greedy steps over device pages",
    )

    # --- Phase 9 (§LXXXII item C): autotuned matmul! row (gated) -------------
    # The §LXXXII exit as ONE corpus row: the CUDA matmul! op consults
    # Autotune, so this prefill executes the (K, N)-regime search on its
    # first call per regime and dispatches to the cached winner thereafter.
    # The row is the autotuned-path prefill (post-warmup, §XXXIII); the note
    # names the winner and makes no speed claim vs any backend (§LXXXII).
    _bench_autotuned_probe() = Array(
        Gesso.reference_prefill(
            _bench_model,
            _bench_gpu_ts,
            _bench_tokens;
            backend=_bench_cuda,
            eps=_bench_cfg.rms_norm_eps,
            theta=_bench_cfg.rope_theta,
        ),
    )
    Gesso.Autotune.invalidate_all!()
    _bench_autotuned_probe()   # warmup: compiles the consult + runs the search once (MISS, §XXXIII)
    _at_tune = Gesso.Autotune.cached_result(:matmul!, :cuda, :llama_micro)
    _at_winner = _at_tune === nothing ? :none : _at_tune.winner
    suite["micro_llama_cuda_prefill_012_autotuned"] = (
        _bench_autotuned_probe,
        "§LXXXII item C: post-warmup — CUDA F32 prefill through the Autotune-consulted matmul!; autotune winner for :llama_micro = :$(_at_winner) (selection receipt, no speed claim)",
    )
else
    println(
        "skipping micro-llama CUDA prefill probes: no NVIDIA device ",
        "(CUDA.functional() == false) — no rows accrued (§LXXVII; §LXXXII autotune row also gated)",
    )
end

# --- Phase 8 (§LXXXI item C): gated micro-llama Lava prefill probe ----------
# benchmark/Project.toml declares Lava (the bench env, never core §VII) so
# the GessoLavaExt extension can trigger here. Without a usable Vulkan
# device the entry is never added — device-less machines accrue no row and
# CI never requires a device (§LXXXI skip law; the run output names it).
# ONE corpus row this sprint (§LXXXI): prefill; no CUDA/llama.cpp comparison
# is claimed anywhere (§LXXXI — tune is Phase 9).
const LAVA_BENCH = let
    ok = true
    try
        @eval Main using Lava
        Lava.vk_context()
    catch
        ok = false
    end
    ok
end

if LAVA_BENCH
    _bench_lava = Gesso.LavaBackend()
    _bench_lava_ts = Gesso.to_device(_bench_lava, _bench_ts)
    _bench_lava_probe() = Array(
        Gesso.reference_prefill(
            _bench_model,
            _bench_lava_ts,
            _bench_tokens;
            backend=_bench_lava,
            eps=_bench_cfg.rms_norm_eps,
            theta=_bench_cfg.rope_theta,
        ),
    )   # Array() reads back to host after the op-boundary synchronize (§LXXXI)
    suite["micro_llama_lava_prefill_012"] = (
        _bench_lava_probe,
        "§LXXXI item C: Vulkan F32 prefill + host readback, tokens [0,1,2], llama_micro shape — portable seam, no speed claim vs CUDA (tune is Phase 9)",
    )
else
    println(
        "skipping micro-llama Lava prefill probe: no usable Vulkan device ",
        "(Lava.vk_context() failed) — no row accrued (§LXXXI)",
    )
end

# --- environment metadata (recorded once; persisted per row) ----------------
host = try
    gethostname()
catch
    "unknown"
end
commit, dirty = git_meta()

println("Gesso benchmark suite (foundation cost)")
println("=" ^ 72)
println(
    "date: ",
    Dates.format(now(UTC), dateformat"yyyy-mm-dd\ THH:MM:SS\Z"),
    "   schema: ",
    BENCH_RESULT_SCHEMA,
    "   julia: ",
    VERSION,
)
println(
    "host: ",
    host,
    "   arch: ",
    Sys.ARCH,
    "   threads: ",
    Threads.nthreads(),
    "   commit: ",
    commit,
    "   dirty: ",
    dirty,
)

for name in sort(collect(keys(suite)))
    f, note = suite[name]
    # §XXXIII: explicit compile before measurement (BenchmarkTools then
    # tunes and samples); allocation numbers are per single evaluation.
    f()
    b = @benchmark $f()
    record(
        name;
        samples=length(b.times),
        median_ns=median(b.times),
        mean_ns=mean(b.times),
        min_ns=minimum(b.times),
        allocs=b.allocs,
        bytes=b.memory,
        note=note,
    )
end

# Byte rows: recorded directly (not through @benchmark — these measure
# BYTES from the page tables, not nanoseconds; ns fields are 0 and
# samples=1 by construction, §XXXIII: no timing is claimed here).
function _bench_share_unique_bytes(forked::Bool)
    if forked
        parent = _bench_share_session()
        Gesso.prefill!(parent, _bench_tokens)
        child = Gesso.fork(parent)
        return Gesso.Profiling.unique_kv_bytes(parent.mgr, child.mgr)
    end
    a = _bench_share_session()
    Gesso.prefill!(a, _bench_tokens)
    b = _bench_share_session()
    Gesso.prefill!(b, _bench_tokens)
    return Gesso.Profiling.unique_kv_bytes(a.mgr, b.mgr)
end
_isolated_bytes = _bench_share_unique_bytes(false)
_forked_bytes = _bench_share_unique_bytes(true)
record(
    "kv_bytes_two_isolated_prefill_cpu";
    samples=1,
    median_ns=0,
    mean_ns=0,
    min_ns=0,
    allocs=0,
    bytes=_isolated_bytes,
    note="§LXXX item C: post-warmup BYTE row (not timing) — unique_kv_bytes over two INDEPENDENT llama_micro prefill sessions = 2× one session's kv_bytes (declaration, not discovery: no fork, no sharing)",
)
record(
    "kv_bytes_prefill_fork_cpu";
    samples=1,
    median_ns=0,
    mean_ns=0,
    min_ns=0,
    allocs=0,
    bytes=_forked_bytes,
    note="§LXXX item C: post-warmup BYTE row (not timing) — unique_kv_bytes(parent, child) after prefill!+fork, pre-decode: the declared alias costs ONE session's kv_bytes; saving vs the isolated-pair row is the difference of the two rows",
)

println("=" ^ 72)
println(length(RESULTS), " benchmarks recorded.")

# --- persistence (North Star §24: accrue a regression corpus) ---------------
results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
tsv = joinpath(results_dir, Dates.format(now(UTC), dateformat"yyyy-mm-dd") * ".tsv")
header_needed = !isfile(tsv)
stamp = Dates.format(now(UTC), dateformat"yyyy-mm-dd\ THH:MM:SS\Z")
open(tsv, "a") do io
    if header_needed
        println(
            io,
            join(
                [
                    "schema",
                    "date_utc",
                    "julia_version",
                    "host",
                    "arch",
                    "nthreads",
                    "commit",
                    "dirty",
                    "benchmark",
                    "samples",
                    "median_ns",
                    "mean_ns",
                    "min_ns",
                    "allocs",
                    "bytes",
                    "bench_note",
                ],
                "\t",
            ),
        )
    end
    for r in RESULTS
        println(
            io,
            join(
                [
                    BENCH_RESULT_SCHEMA,
                    stamp,
                    string(VERSION),
                    string(host),
                    string(Sys.ARCH),
                    Threads.nthreads(),
                    commit,
                    dirty,
                    r.name,
                    r.samples,
                    r.median_ns,
                    r.mean_ns,
                    r.min_ns,
                    r.allocs,
                    r.bytes,
                    r.note,
                ],
                "\t",
            ),
        )
    end
end
println(
    "results appended to ",
    relpath(tsv, dirname(@__DIR__)),
    " (schema ",
    BENCH_RESULT_SCHEMA,
    ")",
)
