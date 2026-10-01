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

    # --- Phase 10B (item C): llama_micro CUDA generate on the fast path ------
    # The path we HAVE, measured: Session.generate on llama_micro AFTER the
    # Phase 10 fast paths (:attn_gemm device-GEMM attention over the gathered
    # scratch + :argmax device greedy — one Int D2H per token). FIRST-TOKEN
    # row = one wall-clock generate on a fresh Session (compile inside,
    # §XXXIII); WARMED row = untimed warmup then the standard suite loop.
    # ids gate BEFORE any row is recorded: CUDA ids must equal the CPU
    # Session's on the same prompt/length or the run errors with no rows
    # (§LXX fail closed). These are FIXTURE rows — llama_micro is NOT the
    # board factor; SmolLM2 G2 rows live in their own gated block.
    _p10_lm_session(; ts=_bench_gpu_ts) = Gesso.Session(
        _bench_model,
        ts;
        backend=_bench_cuda,
        context_length=16,
        eos_token_id=0,
        page_size=4,
        eps=_bench_cfg.rms_norm_eps,
        theta=_bench_cfg.rope_theta,
    )
    _p10_lm_t0 = time_ns()
    _p10_lm_ids = Gesso.generate(_p10_lm_session(), _bench_tokens; max_new_tokens=3)
    _p10_lm_first_s = (time_ns() - _p10_lm_t0) / 1e9
    _p10_lm_cpu_ids = Gesso.generate(
        Gesso.Session(                       # CPU Session on the host tensors
            _bench_model,
            _bench_ts;
            context_length=16,
            eos_token_id=0,
            page_size=4,
            eps=_bench_cfg.rms_norm_eps,
            theta=_bench_cfg.rope_theta,
        ),
        _bench_tokens;
        max_new_tokens=3,
    )
    _p10_lm_ids == _p10_lm_cpu_ids || error(
        "Phase 10B bench: CUDA Session ids diverged from the CPU Session on llama_micro — no rows published (§LXX fail closed)",
    )
    record(
        "llama_micro_gesso_cuda_first_token";
        samples=1,
        median_ns=round(Int, _p10_lm_first_s * 1e9),
        mean_ns=round(Int, _p10_lm_first_s * 1e9),
        min_ns=round(Int, _p10_lm_first_s * 1e9),
        allocs=0,
        bytes=0,
        note="§LXXXIII gates G3 (Phase 10B item C), FIRST-TOKEN clock: llama_micro CUDA Session.generate 3 greedy steps on a fresh Session, one-shot compile INSIDE the number; fast paths :attn_gemm (device GEMM over gathered scratch) + :argmax (one Int D2H per token); ids == CPU Session (asserted before this row); FIXTURE measurement, not the board factor",
    )
    _p10_lm_probe() = Gesso.generate(_p10_lm_session(), _bench_tokens; max_new_tokens=3)
    _p10_lm_probe()   # untimed warmup (§XXXIII)
    suite["llama_micro_gesso_cuda_warmed"] = (
        _p10_lm_probe,
        "§LXXXIII gates G3 (Phase 10B item C), WARMED clock: llama_micro CUDA Session.generate 3 greedy steps post-warmup; fast paths :attn_gemm + :argmax; ids == CPU Session (asserted before any row); FIXTURE measurement, not the board factor",
    )
else
    println(
        "skipping micro-llama CUDA prefill probes: no NVIDIA device ",
        "(CUDA.functional() == false) — no rows accrued (§LXXVII; §LXXXII autotune row also gated)",
    )
end

# --- Phase 10 (G2): named-model first-token + warmed rows vs eager PyTorch --
# The §LXXXIII gate's first implementation: HuggingFaceTB/SmolLM2-135M on the
# Gesso CUDA Session vs HF LlamaForCausalLM eager generate (no torch.compile,
# batch 1, greedy) — SAME checkpoint (GESSO_SMOLLM2_DIR), same prompt
# ("Hello"), same max_new_tokens (8), two clocks on each side. All sub-gates
# must hold: CUDA.functional() (the enclosing CUDA_BENCH), a local snapshot
# (never downloads, §LXXVI), and an external python3 with torch +
# transformers (probe by --help dry run — PyTorch is an EXTERNAL binary,
# never a Project.toml dep, §VII). Any missing gate: named skip, no rows —
# the harness still exists either way.
#
# Clock discipline (§XXXIII, applied to BOTH sides):
#   * FIRST-TOKEN row: one wall-clock generate on a fresh Session, one-shot
#     kernel compile INSIDE the number (that is what first-token means).
#   * WARMED row: untimed warmup, then BenchmarkTools samples. The eager
#     side's warmed number comes from benchmark/compare_eager.py's own
#     in-process warmup + median (python/torch startup amortized OUTSIDE its
#     timed samples) — it is recorded DIRECTLY below, never re-run through
#     @benchmark, which would time a process spawn per sample and lie.
#   * ids gate BEFORE any row is recorded: the CUDA Session's generate must
#     equal the CPU oracle on the named model, or the run errors with NO
#     rows published (§LXX: fail closed — no factor without exact ids).
#   * factor row is a DECLARATION (0-ns byte-row style): computed AFTER the
#     suite loop from THIS run's recorded medians, stated in the note only.
const _P10_PY = "python3"
const _P10_SCRIPT = joinpath(@__DIR__, "compare_eager.py")
const _P10_PROMPT = "Hello"
const _P10_MAXNEW = 8
const _P10_GESSO_DTYPE = "F32"          # CUDA Session compute dtype (§LXXVII)
const P10_TORCH_STAMP = Ref("unknown")  # eager compute dtype (filled in-block)
const P10_EAGER_WARMED_S = Ref(NaN)     # eager warmed seconds (factor post-loop)

function _p10_run_eager(mode::AbstractString)
    cmd = `$_P10_PY $_P10_SCRIPT --mode $mode --max-new-tokens $_P10_MAXNEW --prompt $_P10_PROMPT`
    out = IOBuffer()
    str = ""
    local p
    try
        p = run(pipeline(cmd; stdout=out, stderr=devnull))
        str = String(take!(out))
    catch e
        return (code=-1, out=sprint(showerror, e))
    end
    p.exitcode == 0 || return (code=p.exitcode, out=str)
    kv = Dict{String, Float64}()
    for line in split(str, '\n')
        m = match(r"^([a-z_]+):\s*(.+)$", strip(line))
        m === nothing && continue
        v = tryparse(Float64, m.captures[2])
        v === nothing || (kv[m.captures[1]] = v)
    end
    return (code=0, kv=kv, out=str)
end

function _p10_torch_ok()
    # REAL dry-import probe (Phase 10B item A): `--probe` runs
    # `import torch` + `import transformers` and exits 0 only when both
    # import. --help is NOT a torch probe — it proves the interpreter
    # parses the script, not that PyTorch is installed; probing with it
    # let a snapshot+missing-venv box pass the gate and error() mid-suite
    # (that hole is closed here).
    cmd = `$_P10_PY $_P10_SCRIPT --probe`
    out = IOBuffer()
    try
        ok =
            run(pipeline(cmd; stdout=out, stderr=devnull); wait=false) |>
            wait |>
            p -> p.exitcode == 0
        if ok
            println(
                "G2 torch probe: ",
                strip(String(take!(out))),
                " — eager reference available",
            )
        end
        return ok
    catch
        return false                            # python3 itself missing
    end
end

function _p10_snapshot_ok(dir::AbstractString)
    # isdir is not enough (Phase 10B item A): a junk/wrong dir must yield a
    # NAMED SKIP, not a load_llama explosion mid-suite. The three files the
    # loader actually needs (same contract as test_session_smollm2.jl's gate).
    return isdir(dir) &&
           isfile(joinpath(dir, "config.json")) &&
           isfile(joinpath(dir, "model.safetensors")) &&
           isfile(joinpath(dir, "tokenizer.json"))
end

if CUDA_BENCH &&
   haskey(ENV, "GESSO_SMOLLM2_DIR") &&
   _p10_snapshot_ok(ENV["GESSO_SMOLLM2_DIR"]) &&
   _p10_torch_ok()
    _p10_dir = ENV["GESSO_SMOLLM2_DIR"]
    _p10_model, _p10_cpu_ts, _p10_cfg = Gesso.load_llama(_p10_dir)
    _p10_tk = Gesso.load_gpt2_tokenizer(_p10_dir)
    _p10_ids = Gesso.encode(_p10_tk, _P10_PROMPT)
    _p10_gpu_ts = Gesso.to_device(_bench_cuda, _p10_cpu_ts)
    _p10_session() = Gesso.Session(
        _p10_model,
        _p10_gpu_ts;
        backend=_bench_cuda,
        context_length=128,
        eos_token_id=0,                  # the released checkpoint's EOS (§LXXVI)
        tokenizer=_p10_tk,
        eps=_p10_cfg.rms_norm_eps,
        theta=_p10_cfg.rope_theta,
    )

    # eager side first (both clocks); a failed eager reference aborts the
    # block BEFORE any Gesso row exists — never half a comparison
    _p10_ef = _p10_run_eager("first")
    _p10_ew = _p10_run_eager("warmed")
    (_p10_ef.code == 0 && _p10_ew.code == 0) || error(
        "G2 bench: eager reference failed (codes $(_p10_ef.code)/$(_p10_ew.code)) — no rows published (§LXX)",
    )
    _p10_stamp_txt = let
        m = match(r"compute_dtype:\s*(\S+)", _p10_ew.out)
        m === nothing ? "unknown" : String(m.captures[1])
    end
    P10_TORCH_STAMP[] = _p10_stamp_txt
    P10_EAGER_WARMED_S[] = _p10_ew.kv["seconds"]
    _p10_ef_s = _p10_ef.kv["seconds"]

    # Gesso FIRST-TOKEN clock: one generate on a fresh CUDA Session, compile
    # inside (§XXXIII). This same call's ids are the GATE: no exact ids, no
    # rows — the measurement is discarded with the error.
    _p10_t0 = time_ns()
    _p10_gate_ids = Gesso.generate(_p10_session(), _p10_ids; max_new_tokens=_P10_MAXNEW)
    _p10_g_first_s = (time_ns() - _p10_t0) / 1e9
    _p10_cpu_ids = Gesso.generate(
        Gesso.Session(
            _p10_model,
            _p10_cpu_ts;
            context_length=128,
            eos_token_id=0,
            tokenizer=_p10_tk,
            eps=_p10_cfg.rms_norm_eps,
            theta=_p10_cfg.rope_theta,
        ),
        _p10_ids;
        max_new_tokens=_P10_MAXNEW,
    )
    _p10_gate_ids == _p10_cpu_ids || error(
        "G2 bench: CUDA Session ids diverged from the CPU oracle on the named model — no rows published (§LXX fail closed)",
    )

    # gates passed — publish the rows
    record(
        "smollm2_eager_pytorch_first_token";
        samples=1,
        median_ns=round(Int, _p10_ef_s * 1e9),
        mean_ns=round(Int, _p10_ef_s * 1e9),
        min_ns=round(Int, _p10_ef_s * 1e9),
        allocs=0,
        bytes=0,
        note="§LXXXIII gate G2, FIRST-TOKEN clock: eager PyTorch single-shot generate, kernel compile INSIDE the number (LlamaForCausalLM.generate, do_sample=False, batch 1, NO torch.compile); one wall-clock run measured by benchmark/compare_eager.py; allocs/bytes not applicable across the process boundary",
    )
    record(
        "smollm2_eager_pytorch_warmed";
        samples=Int(_p10_ew.kv["samples"]),
        median_ns=round(Int, _p10_ew.kv["seconds"] * 1e9),
        mean_ns=round(Int, _p10_ew.kv["seconds"] * 1e9),
        min_ns=round(Int, _p10_ew.kv["seconds"] * 1e9),
        allocs=0,
        bytes=0,
        note="§LXXXIII gate G2, WARMED clock: eager PyTorch generate, median of the script's in-process samples AFTER its own untimed warmup (§XXXIII); measured by benchmark/compare_eager.py, recorded here verbatim; compute dtype $_p10_stamp_txt (from_pretrained defaults) vs Gesso $(_P10_GESSO_DTYPE) — both stamps travel with the numbers, no cross-dtype claim",
    )
    record(
        "smollm2_gesso_cuda_first_token";
        samples=1,
        median_ns=round(Int, _p10_g_first_s * 1e9),
        mean_ns=round(Int, _p10_g_first_s * 1e9),
        min_ns=round(Int, _p10_g_first_s * 1e9),
        allocs=0,
        bytes=0,
        note="§LXXXIII gate G2, FIRST-TOKEN clock: SmolLM2-135M CUDA Session generate on a fresh Session (greedy, batch 1), one-shot compile INSIDE the number; ids == CPU oracle (asserted before this row was recorded); compute dtype $(_P10_GESSO_DTYPE)",
    )

    # WARMED Gesso row through the standard suite (BenchmarkTools, post-warmup)
    _p10_gesso_probe() =
        Gesso.generate(_p10_session(), _p10_ids; max_new_tokens=_P10_MAXNEW)
    _p10_gesso_probe()                       # untimed warmup (§XXXIII)
    suite["smollm2_gesso_cuda_warmed"] = (
        _p10_gesso_probe,
        "§LXXXIII gate G2, WARMED clock: SmolLM2-135M CUDA Session generate post-warmup (greedy, batch 1); ids == CPU oracle (asserted before any row exists); compute dtype $(_P10_GESSO_DTYPE); eager warmed compute dtype $_p10_stamp_txt — same checkpoint, prompt \"$_P10_PROMPT\", max_new_tokens $_P10_MAXNEW on both sides",
    )
elseif CUDA_BENCH
    haskey(ENV, "GESSO_SMOLLM2_DIR") || println(
        "skipping G2 SmolLM2 rows: GESSO_SMOLLM2_DIR unset — no local snapshot, ",
        "never downloads (§LXXVI); factor stays NOT MEASURED until ops places weights",
    )
    haskey(ENV, "GESSO_SMOLLM2_DIR") &&
        !_p10_snapshot_ok(ENV["GESSO_SMOLLM2_DIR"]) &&
        println(
            "skipping G2 SmolLM2 rows: GESSO_SMOLLM2_DIR=$(ENV["GESSO_SMOLLM2_DIR"]) is set but is not a usable snapshot ",
            "(needs config.json + model.safetensors + tokenizer.json) — named skip, never downloads (§LXXVI)",
        )
    haskey(ENV, "GESSO_SMOLLM2_DIR") &&
        _p10_snapshot_ok(ENV["GESSO_SMOLLM2_DIR"]) &&
        !_p10_torch_ok() &&
        println(
            "skipping G2 SmolLM2 rows: snapshot present but the dry-import probe failed — ",
            "python3 needs torch AND transformers importable; named skip, no rows (§LXXVI; §LXXXIII)",
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

# --- Phase 10 (G2): the FACTOR row — a DECLARATION, not a measurement -------
# Computed from THIS run's recorded medians (eager warmed median came from
# the external script; the Gesso warmed median from the suite loop above).
# If the G2 block never ran (no CUDA / no snapshot / no torch), the factor
# names its own absence — the harness exists either way.
if !isnan(P10_EAGER_WARMED_S[])
    _p10_g_warmed = begin
        idx = findfirst(r -> r.name == "smollm2_gesso_cuda_warmed", RESULTS)
        idx === nothing ? NaN : RESULTS[idx].median_ns / 1e9
    end
    _p10_factor = P10_EAGER_WARMED_S[] / _p10_g_warmed
    record(
        "smollm2_g2_factor_eager_over_gesso";
        samples=1,
        median_ns=0,
        mean_ns=0,
        min_ns=0,
        allocs=0,
        bytes=0,
        note="§LXXXIII gate G2 FACTOR (declaration, not a timing): eager_warmed_seconds / gesso_warmed_median_seconds = $(round(_p10_factor; digits=3)) × from THIS run's rows (smollm2_eager_pytorch_warmed = $(round(P10_EAGER_WARMED_S[]; digits=6)) s, smollm2_gesso_cuda_warmed = $(round(_p10_g_warmed; digits=6)) s); gesso dtype $(_P10_GESSO_DTYPE), eager dtype $(P10_TORCH_STAMP[]) — both stamps travel with the numbers, no cross-dtype claim; greedy, batch 1, NO torch.compile, NO vLLM (later additional rows)",
    )
else
    record(
        "smollm2_g2_factor_eager_over_gesso";
        samples=1,
        median_ns=0,
        mean_ns=0,
        min_ns=0,
        allocs=0,
        bytes=0,
        note="§LXXXIII gate G2 FACTOR: NOT MEASURED on this run — the G2 block did not execute (needs CUDA + GESSO_SMOLLM2_DIR local snapshot + external python3 with torch/transformers; never downloads, §LXXVI). The harness (benchmark/compare_eager.py) exists either way; the gate number the board cares about is SmolLM2, and llama_micro is NOT that number",
    )
end

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
