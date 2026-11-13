# scripts/env-vars.jl: regenerate docs/env-vars.md, the list of Gesso's
# environment settings. Entry: `make env-vars`.
#
#   julia scripts/env-vars.jl           rewrite docs/env-vars.md
#   julia scripts/env-vars.jl --check   exit 1 if docs/env-vars.md is stale
#
# The user settings and the meaning of each GESSO_DISABLE / GESSO_PROFILE /
# GESSO_TUNE name are written here. The names, defaults and backends of the
# list items are scanned from the env_disabled / env_profile / env_tune call
# sites in src/ and ext/, and the script fails when the scan and the text
# below disagree, so an undocumented or removed name cannot slip through.

const ROOT = dirname(@__DIR__)
const DIRS = ["src", "ext", "scripts", "test", "benchmark"]
const CODE_DIRS = ["src", "ext"]
const OUT = joinpath(ROOT, "docs", "env-vars.md")
const PATTERN = r"\bGESSO_[A-Z0-9_]*[A-Z0-9]\b"

# name, default, meaning
const USER = [
    ("GESSO_BACKEND", "`lava`", "Backend for the scripts and the chat server: `lava`, `cpu` (the correctness oracle), `cuda` (comparator), `amdgpu` (chat server, GPU-untested)."),
    ("GESSO_KV_FORMAT", "per backend", "KV cache format: `f32`, `f16`, `q8` or `q4h`. Unset is `q8` on CUDA, `q8` on Lava below 512 tokens of context and `f32` from 512."),
    ("GESSO_GPU_BUDGET_MIB", "free memory less the reserve", "Device memory for weights, in MiB. Unset probes free memory; 4800 when the probe is unavailable."),
    ("GESSO_GPU_BUDGET_CTX", "65536", "Context length the weight budget keeps KV room for. A server's `--n-ctx` wins."),
    ("GESSO_GPU_RESERVE_MIB", "1024", "Least device memory kept free for KV, prefill passes and the desktop, in MiB."),
    ("GESSO_PREFILL_CHUNK", "1024", "Largest prefill pass, in rows."),
    ("GESSO_CTX_CHECKPOINTS", "8", "Prompt checkpoints kept per session for prefix reuse; 0 keeps only the end-of-prompt snapshot."),
    ("GESSO_CTX_CHECKPOINT_EVERY", "512", "Tokens between context checkpoints."),
    ("GESSO_CPU_POOL_THREADS", "Julia threads, at most 18", "Worker threads in the CPU kernel pool, capped at the Julia thread count."),
    ("GESSO_PIN_THREADS", "on for multi-node", "`1` pins pool workers to cores, `0` never does. Unset pins only on a multi-node box."),
    ("GESSO_NUMA", "on for multi-node", "`1` makes node-local weight copies, `0` never does. Unset copies only on a real multi-node topology."),
    ("GESSO_NUMA_NODES", "sysfs", "Fake NUMA nodes as `;`-separated cpulists (`0-8;9-17`), for testing the multi-node paths on one socket."),
    ("GESSO_LAVA_CPU_ORDER_QUANT", "`1`", "`1` makes Lava's quant kernels sum in CPU order, so Lava matches the CPU path bit for bit. `0` takes the faster subgroup kernels."),
    ("GESSO_DISABLE", "empty", "Comma-separated list of default-on paths to turn off. See below."),
    ("GESSO_PROFILE", "empty", "Comma-separated list of diagnostics to turn on. See below."),
    ("GESSO_TUNE", "empty", "Comma-separated `key=value` tuning overrides. See below."),
]

# Variables in src/ and ext/ that are not run-time settings.
const NOT_SETTINGS = Set(["GESSO_SCHEMA_VERSION", "GESSO_GGML_DIR"])

const DISABLE_DOC = Dict(
    "gdn_prefill_gnorm" => "Fused gated-norm in GatedDeltaNet prefill.",
    "gdn_recur_reg" => "Register-resident GatedDeltaNet recurrence kernel; falls back to the shared-memory kernel.",
    "gpu_dense_quant" => "Quantized dense GEMV/GEMM on the device; falls back to dequant-to-F16.",
    "gguf_read" => "Reading the GGUF into memory at load (slow model disks only, unless forced); falls back to the mmap.",
    "hot_experts" => "Device pool of frequently used host experts (on by default on CUDA, opt-in on Lava).",
    "hot_pin" => "Pinned host staging for the hot-expert pool.",
    "hot_prefill" => "Hot-expert planning during prefill.",
    "lava_quant_head" => "Quantized tied output head on the device; falls back to a device F32 copy of the embedding.",
    "moe_device_prefill" => "Batched device MoE prefill; falls back to the per-token loop.",
    "moe_device_route" => "Router top-k on the device; falls back to the host route.",
    "moe_host_experts" => "Routed experts kept on the host. Lava and AMDGPU require it.",
    "moe_overlap" => "Shared expert overlapped with the routed-expert copies.",
    "moe_prefill_route" => "Device routing during MoE prefill.",
    "precompile_workload" => "The precompile workload run while Gesso precompiles.",
    "prefault" => "Prefaulting GGUF weight pages at load.",
    "prefill_multi" => "Multi-row quant prefill kernels on the CPU.",
    "prefill_window" => "Windowed prefill; falls back to the per-row prefill.",
    "quant_simd" => "SIMD quant kernels on the CPU; falls back to the scalar kernels.",
    "rope_table_cache" => "Cached RoPE tables on Lava.",
    "window_fused" => "Fused sliding-window attention on Lava; falls back to the per-head path.",
)

const PROFILE_DOC = Dict(
    "decode" => "Per-op decode timing. `GESSO_PROFILE=1` means `decode`.",
    "prefill" => "Per-op prefill timing.",
    "timing" => "Per-token generate timing.",
    "hot_stats" => "Hot-expert pool hit statistics when the pool is released.",
    "expert" => "`expert=PATH` writes expert usage counts to PATH.",
    "routing" => "`routing=PATH` writes the per-token routing trace to PATH.",
)

# key => (default override or "", meaning). An empty default shows the code's.
const TUNE_DOC = Dict(
    "cpu_pool_spin_us" => ("200", "Microseconds a CPU pool worker spins before it sleeps."),
    "gemv_chunks_per_thread" => ("16", "CPU GEMV work chunks per thread."),
    "gemv_pool" => ("", "CPU GEMV scheduler: `auto`, `pool` or `threads`."),
    "dq_gemm_min" => ("", "Fewest tokens for which a quant matmul takes the dequant GEMM instead of the GEMV."),
    "dq_gemm_scratch_mib" => ("", "Dequant GEMM scratch buffer, in MiB."),
    "cpu_order_wg_max" => ("per kernel", "Largest workgroup for Lava's CPU-order kernels; `0` keeps the thread-per-row kernels."),
    "moe_device_prefill_chunk" => ("", "Tokens per launch group in device MoE prefill."),
    "expert_cache" => ("", "Host expert slices kept dequantized; `0` is off."),
    "gguf_read" => ("", "`1` reads the GGUF into memory on any disk; by default only on a rotational, FUSE or network disk."),
    "gguf_unreorder_v" => ("", "`1` gathers tiled V heads into grouped order at load."),
    "cuda_weight_dtype" => ("", "Device dtype of large CUDA weights: `f16` or `f32`."),
    "cuda_expert_cache" => ("", "`1` turns on the CUDA device expert cache."),
    "cuda_expert_cache_mib" => ("auto", "CUDA expert cache size, in MiB."),
    "lava_hot_experts" => ("", "`1` turns on the hot-expert pool on Lava."),
    "lava_g4_expert_prefill" => ("off", "`1` runs Gemma 4 routed experts on the Lava device for prefill with at least `g4_expert_prefill_min` tokens; decode stays on the host."),
    "g4_expert_prefill_min" => ("32", "Fewest prompt tokens before Gemma 4 routed experts run on the Lava device."),
    "hot_mib" => ("auto", "Hot-expert pool size, in MiB."),
    "hot_decay" => ("", "Decay of the hot-expert usage score per token."),
    "hot_fill" => ("", "Experts the hot pool loads per token."),
    "hot_margin" => ("", "Score margin an expert needs to replace a resident one."),
    "hot_prefill_keep" => ("no limit", "Most hot experts kept through a prefill."),
    "hot_prefill_max_seq" => ("", "Longest prefill the hot-expert planner handles (max 65535)."),
    "hot_prefill_round" => ("", "Token rounding of hot-expert prefill segments."),
    "hot_prefill_seg" => ("", "Tokens per hot-expert prefill segment."),
    "hot_prefill_gbps" => ("", "Host-to-device bandwidth the prefill planner assumes, in GB/s."),
    "hot_prefill_expert_us" => ("", "Per-expert launch cost the prefill planner assumes, in µs."),
    "hot_prefill_pair_us" => ("", "Per token-expert pair cost the prefill planner assumes, in µs."),
    "hot_prefill_headroom_kib" => ("", "Device headroom the prefill planner keeps, in KiB."),
    "hot_prefill_stream" => ("", "Experts the prefill planner streams per segment."),
)

rel(path) = replace(relpath(path, ROOT), '\\' => '/')

function jl_files(dirs)
    out = String[]
    for d in dirs, (dir, _, files) in walkdir(joinpath(ROOT, d)), f in files
        endswith(f, ".jl") && push!(out, joinpath(dir, f))
    end
    return sort!(out)
end

# Text of the argument list starting at `i` (just after the opening paren).
function call_args(s::String, i::Int)
    depth, j = 1, i
    while j <= ncodeunits(s)
        c = s[j]
        c == '(' && (depth += 1)
        c == ')' && (depth -= 1; depth == 0 && return s[i:prevind(s, j)])
        j = nextind(s, j)
    end
    return s[i:end]
end

backend(f) =
    startswith(f, "ext/cuda_") ? "CUDA" :
    startswith(f, "ext/lava_") ? "Lava" :
    startswith(f, "ext/amdgpu_") ? "AMDGPU" : "all"

function scan_lists()
    lists = Dict(k => Dict{String, Tuple{Set{String}, Set{String}}}() for k in ("disable", "profile", "tune"))
    for path in jl_files(CODE_DIRS)
        endswith(path, "src/env.jl") && continue
        s = read(path, String)
        for m in eachmatch(r"\benv_(disabled|profile_on|profile|tune)\(:([a-z0-9_]+)", s)
            kind = m.captures[1] == "disabled" ? "disable" : m.captures[1] == "tune" ? "tune" : "profile"
            name = String(m.captures[2])
            files, defaults = get!(() -> (Set{String}(), Set{String}()), lists[kind], name)
            push!(files, rel(path))
            if kind == "tune"
                args = call_args(s, m.offset + ncodeunits(m.match))
                push!(defaults, strip(split(args, ','; limit=2)[end]))
            end
        end
    end
    return lists
end

function scan_mentions()
    seen = Dict{String, Set{String}}()
    for path in jl_files(DIRS), m in eachmatch(PATTERN, read(path, String))
        push!(get!(Set{String}, seen, m.match), rel(path))
    end
    return seen
end

function replaced_pairs()
    s = read(joinpath(ROOT, "src", "env.jl"), String)
    block = match(r"const _ENV_REPLACED = \((.*?)\n\)"s, s).captures[1]
    return [(String(m[1]), String(m[2])) for m in eachmatch(r"\"(GESSO_[A-Z0-9_]+)\" => \"([^\"]*)\"", block)]
end

function check_names(errs, kind, doc, scanned)
    for n in setdiff(keys(scanned), keys(doc))
        push!(errs, "GESSO_$(uppercase(kind)) item `$n` is read in code but not documented here")
    end
    for n in setdiff(keys(doc), keys(scanned))
        push!(errs, "GESSO_$(uppercase(kind)) item `$n` is documented here but not read in code")
    end
end

function check_reads(errs, user_names, replaced)
    old = Set(first.(replaced))
    for path in jl_files(CODE_DIRS)
        endswith(path, "src/env.jl") && continue
        for m in eachmatch(r"\"(GESSO_[A-Z0-9_]+)\"", read(path, String))
            n = m.captures[1]
            n in user_names || n in NOT_SETTINGS || push!(errs,
                n in old ? "$(rel(path)) still reads replaced variable $n" :
                "$(rel(path)) reads $n, which is not a documented user setting")
        end
    end
end

cell_files(files) = join(("`$f`" for f in sort!(collect(files))), ", ")
cell_backends(files) = join(sort!(unique(backend.(collect(files)))), ", ")

function render(lists, mentions, replaced)
    io = IOBuffer()
    p(xs...) = println(io, xs...)
    p("# Environment variables")
    p()
    p("Generated by `julia scripts/env-vars.jl` and checked in CI; do not edit by hand.")
    p()
    p("## Settings")
    p()
    p("Everything Gesso reads at run time. Most advanced switches sit under the three")
    p("lists at the bottom, whose items are comma separated; names are case-insensitive")
    p("and spaces around items are ignored. A `GESSO_TUNE` value that does not parse is")
    p("an error.")
    p()
    p("    GESSO_DISABLE=prefill_window,quant_simd")
    p("    GESSO_PROFILE=decode,expert=/tmp/experts.tsv")
    p("    GESSO_TUNE=dq_gemm_min=64,hot_mib=480")
    p()
    p("| Variable | Default | Meaning |")
    p("|---|---|---|")
    for (n, d, m) in USER
        p("| `", n, "` | ", d, " | ", m, " |")
    end
    p()
    p("Backend `all` means the CPU path and every GPU backend.")
    p()
    p("## GESSO_DISABLE")
    p()
    p("| Name | Backend | Turns off |")
    p("|---|---|---|")
    for n in sort!(collect(keys(DISABLE_DOC)))
        p("| `", n, "` | ", cell_backends(lists["disable"][n][1]), " | ", DISABLE_DOC[n], " |")
    end
    p()
    p("## GESSO_PROFILE")
    p()
    p("| Name | Backend | Turns on |")
    p("|---|---|---|")
    for n in sort!(collect(keys(PROFILE_DOC)))
        p("| `", n, "` | ", cell_backends(lists["profile"][n][1]), " | ", PROFILE_DOC[n], " |")
    end
    p()
    p("## GESSO_TUNE")
    p()
    p("| Key | Default | Backend | Meaning |")
    p("|---|---|---|---|")
    for n in sort!(collect(keys(TUNE_DOC)))
        files, defaults = lists["tune"][n]
        d, m = TUNE_DOC[n]
        dflt = isempty(d) ? join(("`$(get(Dict("false" => "0", "true" => "1"), x, strip(x, '"')))`" for x in sort!(collect(defaults))), " / ") : d
        p("| `", n, "` | ", dflt, " | ", cell_backends(files), " | ", m, " |")
    end
    p()
    p("## Replaced variables")
    p()
    p("Gesso no longer reads these. It warns once at load when one is set.")
    p()
    p("| Old variable | Set instead |")
    p("|---|---|")
    for (old, new) in replaced
        p("| `", old, "` | `", replace(new, "|" => "\\|"), "` |")
    end
    p()
    p("## Script, test and oracle variables")
    p()
    p("Read only by scripts, tests and benchmarks (and `GESSO_GGML_DIR`, the llama.cpp")
    p("build the oracle tests compare against). Each is documented where it is read.")
    p()
    p("| Variable | Mentioned in |")
    p("|---|---|")
    skip = union(Set(first.(USER)), Set(first.(replaced)), Set(["GESSO_SCHEMA_VERSION"]))
    for n in sort!(collect(keys(mentions)))
        n in skip && continue
        p("| `", n, "` | ", cell_files(mentions[n]), " |")
    end
    return String(take!(io))
end

lists = scan_lists()
replaced = replaced_pairs()
errs = String[]
check_names(errs, "disable", DISABLE_DOC, lists["disable"])
check_names(errs, "profile", PROFILE_DOC, lists["profile"])
check_names(errs, "tune", TUNE_DOC, lists["tune"])
check_reads(errs, Set(first.(USER)), replaced)
if !isempty(errs)
    foreach(e -> println(stderr, "env-vars: ", e), errs)
    exit(1)
end

text = render(lists, scan_mentions(), replaced)
if "--check" in ARGS
    current = isfile(OUT) ? read(OUT, String) : ""
    if current != text
        println(stderr, "docs/env-vars.md is stale; run `julia scripts/env-vars.jl`")
        exit(1)
    end
    println("docs/env-vars.md is current")
else
    write(OUT, text)
    println("wrote docs/env-vars.md")
end
