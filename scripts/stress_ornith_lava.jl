#!/usr/bin/env julia
# Set GESSO_L1_JSONL to a new file. Roles: oracle writes GESSO_L1_ORACLE_OUT on CPU,
# validate reads GESSO_L1_ORACLE, selfcheck tests only the strict ID gate.
# Configure GESSO_ORNITH35_GGUF, GESSO_REPO, GESSO_GGML_DIR, GESSO_BACKEND=cpu|lava.
# GESSO_L1_* controls cases (default p64,hello), tokens, repeats, state modes, seed,
# variant length, cancellation/recovery, and GC cadence.

using Dates, Random, Gesso, JSON
if get(ENV, "GESSO_L1_ROLE", "validate") != "selfcheck" &&
   get(
    ENV,
    "GESSO_BACKEND",
    get(ENV, "GESSO_L1_ROLE", "validate") == "oracle" ? "cpu" : "lava",
) == "lava"
    @eval using Lava
end
const PREFIXES = (1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256)
struct CallbackProbeError <: Exception end
Base.showerror(io::IO, ::CallbackProbeError) = print(io, "intentional callback exception")

emit(io, x) = (println(io, JSON.json(x)); flush(io))
function envint(k, d, lo=0)
    x = tryparse(Int, get(ENV, k, string(d)))
    (x === nothing || x < lo) && error("invalid $k")
    return x
end
function id_gate(actual, expected, baseline, anchor)
    exact = expected === nothing ? nothing : actual == expected
    stable = baseline === nothing ? nothing : actual == baseline
    anchor_ok = anchor === nothing ? nothing : (!isempty(actual) && first(actual) == anchor)
    return (;
        exact,
        stable,
        anchor_ok,
        passed=(exact !== false && stable !== false && anchor_ok !== false),
    )
end
function callback_first_id_matches(seen, expected_full, prompt_len)
    prompt_len >= 0 || return false
    length(seen) == 1 || return false
    length(expected_full) > prompt_len || return false
    return first(seen) == expected_full[prompt_len+1]
end
function reuse_gate(prompt, before, after, full, info)
    valid_info =
        info isa NamedTuple &&
        haskey(info, :reused) &&
        info.reused isa Int &&
        0 <= info.reused <= length(prompt)
    expected_reused =
        before === nothing ? 0 :
        length(before) <= length(prompt) && view(prompt, 1:length(before)) == before ?
        length(before) : nothing
    reused = valid_info ? info.reused : nothing
    reuse_ok = valid_info && (expected_reused === nothing || reused == expected_reused)
    # The last returned token is sampled but not consumed by the interpreter.
    cache_ok =
        after === nothing ? nothing :
        full !== nothing && length(full) > length(prompt) && after == full[1:(end-1)]
    return (;
        reused,
        expected_reused,
        reuse_ok,
        cache_ok,
        passed=reuse_ok && cache_ok !== false,
    )
end
function selfcheck(io)
    ok = id_gate([7, 8], [7, 8], nothing, nothing).passed
    bad = id_gate([7, 9], [7, 8], nothing, nothing).passed
    unstable = id_gate([7, 9], nothing, [7, 8], nothing).passed
    callback_ok = callback_first_id_matches([7], [1, 7, 8], 1)
    callback_bad = callback_first_id_matches([9], [1, 7, 8], 1)
    callback_wrong_count = callback_first_id_matches([7, 8], [1, 7, 8], 1)
    reuse_ok = reuse_gate([1, 2], [1], [1, 2], [1, 2, 7], (; reused=1)).passed
    missed_reuse = reuse_gate([1, 2], [1], [1, 2], [1, 2, 7], (; reused=0)).passed
    wrong_cache = reuse_gate([1, 2], [1], [1], [1, 2, 7], (; reused=1)).passed
    implicit_ok = reuse_gate([1, 2], nothing, nothing, [1, 2, 7], (; reused=0)).passed
    invalid_reuse = reuse_gate([1, 2], [1], [1, 2], [1, 2, 7], (; reused=3)).passed
    pass =
        ok &&
        !bad &&
        !unstable &&
        callback_ok &&
        !callback_bad &&
        !callback_wrong_count &&
        reuse_ok &&
        !missed_reuse &&
        !wrong_cache &&
        implicit_ok &&
        !invalid_reuse
    emit(
        io,
        (;
            kind="selfcheck",
            exact_match_passes=ok,
            oracle_mismatch_fails=(!bad),
            repeat_mismatch_fails=(!unstable),
            callback_first_match_passes=callback_ok,
            callback_first_mismatch_fails=(!callback_bad),
            callback_wrong_count_fails=(!callback_wrong_count),
            prefix_reuse_passes=reuse_ok,
            silently_missed_reuse_fails=(!missed_reuse),
            incomplete_cached_ids_fail=(!wrong_cache),
            implicit_zero_reuse_passes=implicit_ok,
            out_of_range_reuse_fails=(!invalid_reuse),
            passed=pass,
        ),
    )
    return pass ? 0 : 1
end
gitread(repo, args...) = strip(read(Cmd(["git", "-C", repo, args...]), String))

function make_cases(corpus)
    names = String.(strip.(split(get(ENV, "GESSO_L1_CASES", "p64,hello"), ',')))
    any(isempty, names) && error("empty GESSO_L1_CASES item")
    seed = envint("GESSO_L1_SEED", 20261014)
    vlen = envint("GESSO_L1_VARIANT_LEN", 64, 1)
    rng = MersenneTwister(seed)
    cases = NamedTuple[]
    for name in names
        if name == "hello"
            push!(
                cases,
                (; name, prompt_ids=[9419], source="fixed_hello_t1", corpus_start=0),
            )
        elseif name == "varied"
            vlen <= length(corpus) || error("variant length exceeds corpus")
            start = rand(rng, 1:(length(corpus)-vlen+1))
            push!(
                cases,
                (;
                    name,
                    prompt_ids=copy(corpus[start:(start+vlen-1)]),
                    source="seeded_corpus_segment",
                    corpus_start=start,
                ),
            )
        else
            n = startswith(name, "p") ? tryparse(Int, name[2:end]) : nothing
            n in PREFIXES || error("unsupported case '$name'")
            n <= length(corpus) || error("case $name exceeds corpus length")
            push!(
                cases,
                (;
                    name,
                    prompt_ids=copy(corpus[1:n]),
                    source="corpus_prefix",
                    corpus_start=1,
                ),
            )
        end
    end
    isempty(cases) && error("no cases selected")
    return cases, seed, vlen
end
function tickgc!(io, count, every)
    count[] += 1
    if every > 0 && count[] % every == 0
        t = time_ns()
        GC.gc(true)
        emit(io, (; kind="gc", after_request=count[], elapsed_ns=time_ns()-t))
    end
end

function request!(
    io,
    failures,
    count,
    every,
    stability,
    cfg;
    model,
    tensors,
    backend,
    eps,
    case,
    prompt=case.prompt_ids,
    n,
    state,
    mode,
    phase,
    repeat,
    expected,
    expected_count=n,
    callback=nothing,
    callback_ids=nothing,
)
    t0 = time_ns()
    full = nothing
    err = nothing
    state_before = state === nothing ? nothing : copy(state.ids)
    info = Ref{Any}(nothing)
    try
        full = Int.(
            Gesso.reference_generate(
                model,
                tensors,
                prompt;
                backend,
                max_new_tokens=n,
                state,
                eps,
                on_token=callback,
                eos_ids=(),
                info,
            ),
        )
    catch e
        err = (; type=string(typeof(e)), message=sprint(showerror, e))
    end
    elapsed = time_ns()-t0
    prompt_ok =
        full !== nothing &&
        length(full) >= length(prompt) &&
        full[1:length(prompt)] == prompt
    generated = prompt_ok ? full[(length(prompt)+1):end] : Int[]
    expected_gen = expected === nothing ? nothing : expected[(length(prompt)+1):end]
    anchor =
        phase in ("oracle", "normal", "cancel") && prompt == case.prompt_ids ?
        (case.name == "p64" ? 82202 : case.name == "hello" ? 11 : nothing) : nothing
    key = (case.name, Tuple(prompt), n, mode, phase)
    baseline = get(stability, key, nothing)
    gate = id_gate(generated, expected_gen, baseline, anchor)
    state_after = state === nothing ? nothing : state.ids
    reuse = reuse_gate(prompt, state_before, state_after, full, info[])
    cbok = callback_ids === nothing ? nothing : length(callback_ids) == expected_count
    passed =
        err === nothing &&
        prompt_ok &&
        length(generated) == expected_count &&
        cbok !== false &&
        gate.passed &&
        reuse.passed
    emit(
        io,
        (;
            kind="request",
            time=string(Dates.now(Dates.UTC)),
            role=cfg.role,
            backend=cfg.backend,
            case=case.name,
            phase,
            repeat,
            state_mode=mode,
            prompt_ids=prompt,
            full_ids=full,
            generated_ids=generated,
            expected_generated_ids=expected_gen,
            exact_ids_match=gate.exact,
            same_backend_stability_ok=gate.stable,
            same_backend_baseline_ids=baseline,
            known_anchor_ok=gate.anchor_ok,
            callback_ids,
            callback_count_ok=cbok,
            reused_rows=reuse.reused,
            expected_prefix_reused_rows=reuse.expected_reused,
            reuse_info_ok=reuse.reuse_ok,
            state_rows_before=state_before === nothing ? nothing : length(state_before),
            state_rows_after=state_after === nothing ? nothing : length(state_after),
            state_cache_ids_match=reuse.cache_ok,
            error=err,
            elapsed_ns=elapsed,
            passed,
        ),
    )
    baseline === nothing &&
        full !== nothing &&
        prompt_ok &&
        (stability[key] = copy(generated))
    failures[] += passed ? 0 : 1
    tickgc!(io, count, every)
    return (; full, generated, passed)
end
function read_oracle(path, model_path, model_bytes, cases, lengths)
    doc = JSON.parsefile(path)
    doc["format"] == "gesso-ornith-l1-oracle-v2" || error("oracle format mismatch")
    doc["backend"] == "cpu" || error("oracle backend must be CPU")
    doc["model_path"] == model_path || error("oracle model path mismatch")
    Int(doc["model_bytes"]) == model_bytes || error("oracle model size mismatch")
    rows = Dict{Tuple{String, Int}, Any}()
    for r in doc["records"]
        key = (String(r["case"]), Int(r["max_new_tokens"]))
        haskey(rows, key) && error("duplicate oracle record $key")
        rows[key] = r
    end
    expected = Dict{Tuple{String, Int}, Vector{Int}}()
    for c in cases, n in lengths
        r = get(rows, (c.name, n), nothing)
        r === nothing && error("oracle lacks $(c.name) / $n tokens")
        Int.(r["prompt_ids"]) == c.prompt_ids ||
            error("oracle prompt mismatch for $(c.name)")
        expected[(c.name, n)] = Int.(r["full_ids"])
    end
    return expected
end

function write_oracle(path, gguf, cases, lengths, expected, seed, vlen, head, dirty)
    ispath(path) && error("oracle output already exists: $path")
    rows = [
        (;
            case=c.name,
            prompt_ids=c.prompt_ids,
            max_new_tokens=n,
            full_ids=expected[(c.name, n)],
        ) for c in cases for n in lengths
    ]
    doc = (;
        format="gesso-ornith-l1-oracle-v2",
        backend="cpu",
        oracle_kind="Gesso CPU reference_generate; not independent Hugging Face",
        model_path=gguf,
        model_bytes=filesize(gguf),
        git_head=head,
        source_dirty=dirty,
        seed,
        variant_len=vlen,
        records=rows,
    )
    open(path, "w") do f
        println(f, JSON.json(doc))
    end
end

function callback_recovery!(
    io,
    failures,
    count,
    every,
    stability,
    cfg;
    model,
    tensors,
    backend,
    eps,
    case,
    expected,
    ctx,
)
    state = Gesso.GenerateState(ctx)
    seen = Int[]
    t0 = time_ns()
    caught = nothing
    try
        Gesso.reference_generate(
            model,
            tensors,
            case.prompt_ids;
            backend,
            max_new_tokens=1,
            state,
            eps,
            eos_ids=(),
            on_token=id -> (push!(seen, id); throw(CallbackProbeError())),
        )
    catch e
        caught = e
    end
    elapsed = time_ns()-t0
    invalidated = isempty(state.ids)
    anchor = case.name == "p64" ? 82202 : case.name == "hello" ? 11 : nothing
    exact_ids_match = callback_first_id_matches(seen, expected, length(case.prompt_ids))
    expected_first =
        length(expected) > length(case.prompt_ids) ? expected[length(case.prompt_ids)+1] :
        nothing
    ok =
        caught isa CallbackProbeError &&
        length(seen)==1 &&
        invalidated &&
        exact_ids_match &&
        (anchor === nothing || first(seen)==anchor)
    emit(
        io,
        (;
            kind="request",
            role=cfg.role,
            backend=cfg.backend,
            case=case.name,
            phase="callback_exception",
            state_mode="explicit",
            prompt_ids=case.prompt_ids,
            full_ids=nothing,
            generated_ids=seen,
            expected_full_ids=expected,
            expected_first_generated_id=expected_first,
            exact_ids_match,
            error=caught === nothing ? nothing :
                  (; type=string(typeof(caught)), message=sprint(showerror, caught)),
            expected_callback_exception=true,
            state_invalidated=invalidated,
            known_anchor_ok=anchor === nothing ? nothing :
                            !isempty(seen) && first(seen)==anchor,
            elapsed_ns=elapsed,
            passed=ok,
        ),
    )
    failures[] += ok ? 0 : 1
    tickgc!(io, count, every)
    request!(
        io,
        failures,
        count,
        every,
        stability,
        cfg;
        model,
        tensors,
        backend,
        eps,
        case,
        n=cfg.maxnew,
        state,
        mode="retry_same_state",
        phase="retry",
        repeat=1,
        expected,
    )
    close(state)
    GC.gc(true)
    fresh=Gesso.GenerateState(ctx)
    result=request!(
        io,
        failures,
        count,
        every,
        stability,
        cfg;
        model,
        tensors,
        backend,
        eps,
        case,
        n=cfg.maxnew,
        state=fresh,
        mode="closed_then_new_state",
        phase="newstate_recovery",
        repeat=1,
        expected,
    )
    close(fresh)
    GC.gc(true)
    return result
end

function cancel_recover!(
    io,
    failures,
    count,
    every,
    stability,
    cfg;
    model,
    tensors,
    backend,
    eps,
    case,
    expected,
    ctx,
    cancel_after,
    recovery_tokens,
)
    state=Gesso.GenerateState(ctx)
    seen=Int[]
    total=cancel_after+recovery_tokens
    partial=request!(
        io,
        failures,
        count,
        every,
        stability,
        cfg;
        model,
        tensors,
        backend,
        eps,
        case,
        n=cancel_after+1,
        state,
        mode="explicit_cancel_state",
        phase="cancel",
        repeat=1,
        expected=expected[(case.name, cancel_after)],
        expected_count=cancel_after,
        callback_ids=seen,
        callback=id -> (push!(seen, id); length(seen)<cancel_after),
    )
    if partial.full !== nothing
        request!(
            io,
            failures,
            count,
            every,
            stability,
            cfg;
            model,
            tensors,
            backend,
            eps,
            case,
            prompt=partial.full,
            n=recovery_tokens,
            state,
            mode="explicit_cancel_state",
            phase="recovery",
            repeat=1,
            expected=expected[(case.name, total)],
            expected_count=recovery_tokens,
        )
    end
    close(state)
    GC.gc(true)
end

function main()
    logfile=strip(get(ENV, "GESSO_L1_JSONL", ""))
    isempty(logfile) && (println(stderr, "set GESSO_L1_JSONL to a new file"); return 2)
    ispath(logfile) && (println(stderr, "JSONL already exists: $logfile"); return 2)
    mkpath(dirname(abspath(logfile)))
    io=try
        open(logfile, "w")
    catch e
        println(stderr, sprint(showerror, e))
        return 2
    end
    failures=Ref(0)
    count=Ref(0)
    try
        role=lowercase(strip(get(ENV, "GESSO_L1_ROLE", "validate")))
        role=="selfcheck" && return selfcheck(io)
        role in ("oracle", "validate") ||
            error("role must be oracle, validate, or selfcheck")
        backend_name=lowercase(
            strip(get(ENV, "GESSO_BACKEND", role=="oracle" ? "cpu" : "lava")),
        )
        backend_name in ("cpu", "lava") || error("backend must be cpu or lava")
        role=="oracle" && backend_name!="cpu" && error("oracle requires CPU backend")
        gguf_raw=strip(get(ENV, "GESSO_ORNITH35_GGUF", ""))
        isempty(gguf_raw) && error("set GESSO_ORNITH35_GGUF")
        repo_raw=strip(get(ENV, "GESSO_REPO", ""))
        isempty(repo_raw) && error("set GESSO_REPO")
        gguf=realpath(gguf_raw)
        repo=realpath(repo_raw)
        ggml=get(ENV, "GESSO_GGML_DIR", "")
        isfile(gguf) || error("model file missing")
        isdir(ggml) || error("invalid GESSO_GGML_DIR")
        isfile(joinpath(repo, "docs", "Gesso_Stack.md")) || error("corpus missing")
        source=realpath(pkgdir(Gesso))
        source==repo || error("active Gesso source differs from GESSO_REPO")
        backend_name=="lava" &&
            get(ENV, "GESSO_MOE_HOST_EXPERTS", "1")!="1" &&
            error("Lava requires GESSO_MOE_HOST_EXPERTS=1")
        oracle_path=strip(get(ENV, "GESSO_L1_ORACLE", ""))
        oracle_out=strip(get(ENV, "GESSO_L1_ORACLE_OUT", ""))
        if role=="oracle"
            isempty(oracle_out) && error("set GESSO_L1_ORACLE_OUT")
        else
            isempty(oracle_path) && error("set GESSO_L1_ORACLE")
        end
        role=="oracle" && ispath(oracle_out) && error("oracle output already exists")
        role=="validate" && !isfile(oracle_path) && error("oracle file missing")
        abspath(logfile)==abspath(role=="oracle" ? oracle_out : oracle_path) &&
            error("JSONL and oracle must differ")
        maxnew=envint("GESSO_L1_MAX_NEW_TOKENS", 1, 1)
        repeats=envint("GESSO_L1_REPEATS", 2, 1)
        every=envint("GESSO_L1_GC_EVERY", 4)
        modes_raw=lowercase(strip(get(ENV, "GESSO_L1_STATE_MODES", "both")))
        modes=modes_raw=="both" ? ["implicit", "state"] : [modes_raw]
        all(m->m in ("implicit", "state"), modes) || error("bad GESSO_L1_STATE_MODES")
        cancel_raw=get(ENV, "GESSO_L1_CANCEL", "1")
        cancel_raw in ("0", "1") || error("GESSO_L1_CANCEL must be 0 or 1")
        cancel=cancel_raw=="1"
        cancel_case=strip(get(ENV, "GESSO_L1_CANCEL_CASE", "p64"))
        ca=envint("GESSO_L1_CANCEL_AFTER", 1, 1)
        rt=envint("GESSO_L1_RECOVERY_TOKENS", 1, 1)
        backend = if backend_name == "cpu"
            Gesso.CPUBackend()
        else
            Gesso.LavaBackend()
        end
        model, host, spec=Gesso.load_gguf(gguf)
        tensors=backend_name=="cpu" ? host : Gesso.to_device(backend, host)
        tk=Gesso.Inference.load_gguf_tokenizer(gguf)
        corpus=Int.(
            Gesso.Inference.encode(
                tk,
                read(joinpath(repo, "docs", "Gesso_Stack.md"), String);
                parse_special=false,
            ),
        )
        isempty(corpus) && error("empty encoded corpus")
        cases, seed, vlen=make_cases(corpus)
        cancel_case_idx=findfirst(c->c.name==cancel_case, cases)
        cancel && cancel_case_idx===nothing && error("cancel case must be selected")
        lengths=unique(vcat([maxnew], cancel ? [ca, ca+rt] : Int[]))
        ctx=maximum(length(c.prompt_ids) for c in cases)+maximum(lengths)
        bytes=filesize(gguf)
        head=gitread(repo, "rev-parse", "HEAD")
        dirty=gitread(repo, "status", "--porcelain")
        emit(
            io,
            (;
                kind="start",
                role,
                backend=backend_name,
                gguf,
                model_bytes=bytes,
                repo,
                active_source=source,
                git_head=head,
                source_dirty=(!isempty(dirty)),
                source_status=dirty,
                julia_threads=Threads.nthreads(),
                cases=[c.name for c in cases],
                state_modes=modes,
                max_new_tokens=maxnew,
                repeats,
                seed,
                variant_len=vlen,
                cancellation=(;
                    enabled=cancel,
                    case=cancel_case,
                    cancel_after=ca,
                    recovery_tokens=rt,
                ),
                moe_overlap=get(ENV, "GESSO_MOE_OVERLAP", "default"),
                prefill_multi=get(ENV, "GESSO_PREFILL_MULTI", "default"),
                rms_norm_eps=Float64(spec.rms_norm_eps),
            ),
        )
        expected=role=="oracle" ? Dict{Tuple{String, Int}, Vector{Int}}() :
                 read_oracle(realpath(oracle_path), gguf, bytes, cases, lengths)
        stability=Dict{Any, Vector{Int}}()
        cfg=(; role, backend=backend_name, maxnew)
        if role=="oracle"
            for n in lengths, rep in 1:repeats, c in cases
                r=request!(
                    io,
                    failures,
                    count,
                    every,
                    stability,
                    cfg;
                    model,
                    tensors=host,
                    backend,
                    eps=Float64(spec.rms_norm_eps),
                    case=c,
                    n,
                    state=nothing,
                    mode="implicit",
                    phase="oracle",
                    repeat=rep,
                    expected=get(expected, (c.name, n), nothing),
                )
                r.full===nothing || get!(expected, (c.name, n), r.full)
            end
        else
            for mode in modes
                state=mode=="state" ? Gesso.GenerateState(ctx) : nothing
                try
                    for rep in 1:repeats, c in cases
                        request!(
                            io,
                            failures,
                            count,
                            every,
                            stability,
                            cfg;
                            model,
                            tensors,
                            backend,
                            eps=Float64(spec.rms_norm_eps),
                            case=c,
                            n=maxnew,
                            state,
                            mode,
                            phase="normal",
                            repeat=rep,
                            expected=expected[(c.name, maxnew)],
                        )
                    end
                finally
                    state===nothing || close(state)
                    GC.gc(true)
                end
            end
        end
        probe=first(cases)
        if haskey(expected, (probe.name, maxnew))
            callback_recovery!(
                io,
                failures,
                count,
                every,
                stability,
                cfg;
                model,
                tensors=role=="oracle" ? host : tensors,
                backend=role=="oracle" ? Gesso.CPUBackend() : backend,
                eps=Float64(spec.rms_norm_eps),
                case=probe,
                expected=expected[(probe.name, maxnew)],
                ctx,
            )
        else
            emit(
                io,
                (;
                    kind="scenario_skipped",
                    scenario="callback_exception_recovery",
                    reason="oracle request unavailable",
                ),
            )
        end
        if cancel && all(haskey(expected, (cancel_case, n)) for n in (ca, ca+rt))
            c=cases[cancel_case_idx]
            cancel_recover!(
                io,
                failures,
                count,
                every,
                stability,
                cfg;
                model,
                tensors=role=="oracle" ? host : tensors,
                backend=role=="oracle" ? Gesso.CPUBackend() : backend,
                eps=Float64(spec.rms_norm_eps),
                case=c,
                expected,
                ctx,
                cancel_after=ca,
                recovery_tokens=rt,
            )
        elseif cancel
            emit(
                io,
                (;
                    kind="scenario_skipped",
                    scenario="cancel_recovery",
                    reason="oracle requests unavailable",
                ),
            )
        end
        if role=="oracle" && failures[]==0
            write_oracle(
                oracle_out,
                gguf,
                cases,
                lengths,
                expected,
                seed,
                vlen,
                head,
                !isempty(dirty),
            )
            emit(
                io,
                (;
                    kind="oracle_written",
                    path=abspath(oracle_out),
                    records=length(expected),
                    independent_hf=false,
                ),
            )
        end
        emit(
            io,
            (;
                kind="summary",
                status=failures[]==0 ? "PASS" : "FAIL",
                failures=failures[],
                role,
                backend=backend_name,
                exact_generated_ids=true,
                same_backend_stability=repeats>1,
                cpu_oracle_independent_hf=false,
            ),
        )
        return failures[]==0 ? 0 : 1
    catch e
        emit(
            io,
            (;
                kind="fatal",
                error_type=string(typeof(e)),
                error=sprint(showerror, e),
                passed=false,
            ),
        )
        showerror(stderr, e)
        println(stderr)
        return 2
    finally
        close(io)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    exit(main())
end
