#!/usr/bin/env bash
# Drafter acceptance on Saluki through llama.cpp's server, CPU only. It checks
# published drafters (MTP head, DFlash2, a small Qwen as draft-simple) against
# Saluki before any of them is ported into Gesso. Same chat prompts as the Gesso
# benches (saluki_prompts.jl), greedy, 96 new tokens.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   scripts/saluki_llamacpp_drafter.sh <label> [spec-type draft.gguf n_max]
#
# With only <label> it runs plain decode. Prints one line per prompt:
#   LCPP <label> <tag> new= ms_per_token= draft_n= draft_accepted= accept_rate=
# Optional env: GESSO_BENCH_NEW (default 96), GESSO_LCPP_THREADS (default 16),
# GESSO_LCPP_PORT (default 8931), GESSO_BENCH_ONLY (space-separated tags), GESSO_LCPP_SAVE
# (a directory: keeps each response as <label>-<tag>.json with token ids and top-2 probs).
# llama-server reads LLAMA_ARG_* settings from the environment, e.g.
# LLAMA_ARG_SPEC_DRAFT_P_MIN for a draft confidence gate.
set -euo pipefail

GGUF=${GESSO_QWEN38_27B_GGUF:?set GESSO_QWEN38_27B_GGUF}
BIN=${GESSO_GGML_DIR:?set GESSO_GGML_DIR}
LABEL=${1:?usage: saluki_llamacpp_drafter.sh <label> [spec-type draft.gguf n_max]}
SPEC=${2:-}
DRAFT=${3:-}
NMAX=${4:-4}
NEW=${GESSO_BENCH_NEW:-96}
THREADS=${GESSO_LCPP_THREADS:-16}
PORT=${GESSO_LCPP_PORT:-8931}
TAGS=${GESSO_BENCH_ONLY:-json code shell math prose}
SAVE=${GESSO_LCPP_SAVE:-}
keep=false; [[ -n $SAVE ]] && keep=true
HERE=$(cd "$(dirname "$0")" && pwd)

work=$(mktemp -d)
trap 'kill "${srv:-0}" 2>/dev/null || true; rm -rf "$work"' EXIT

# One prompt file per tag, built from the shared Julia prompt list.
julia --startup-file=no -e '
    include(ARGS[1])
    for (t, m) in PROMPTS
        write(joinpath(ARGS[2], t * ".txt"), chat(m))
    end' "$HERE/saluki_prompts.jl" "$work"

args=(-m "$GGUF" -t "$THREADS" -ngl 0 -c 4096 --port "$PORT" -np 1)
if [[ -n $SPEC ]]; then
    args+=(--spec-type "$SPEC" --spec-draft-n-max "$NMAX" -td "$THREADS" -ngld 0)
    [[ -n $DRAFT ]] && args+=(-md "$DRAFT")
fi
# CPU only: hide the GPU so this never touches other jobs on it.
CUDA_VISIBLE_DEVICES="" "$BIN/llama-server" "${args[@]}" >"$work/server.log" 2>&1 &
srv=$!
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 "$srv" 2>/dev/null || { cat "$work/server.log"; exit 1; }
    sleep 2
done

# JIT-free, but the first request still pages weights in: warm once.
curl -sf "http://127.0.0.1:$PORT/completion" -d '{"prompt":"Hi","n_predict":4}' >/dev/null

for tag in $TAGS; do
    body=$(jq -n --rawfile p "$work/$tag.txt" --argjson n "$NEW" --argjson keep $keep \
        '{prompt:$p, n_predict:$n, temperature:0, top_k:1, cache_prompt:false}
        + if $keep then {n_probs:2, return_tokens:true} else {} end')
    curl -sf "http://127.0.0.1:$PORT/completion" -d "$body" >"$work/out.json"
    $keep && cp "$work/out.json" "$SAVE/$LABEL-$tag.json"
    # text hash: greedy speculation must reproduce the plain run's text
    sha=$(jq -j .content "$work/out.json" | sha1sum | cut -c1-12)
    jq -r --arg l "$LABEL" --arg t "$tag" --arg s "$sha" '
        .timings as $x
        | "LCPP \($l) \($t) new=\($x.predicted_n) ms_per_token=\($x.predicted_per_token_ms * 10 | round / 10) draft_n=\($x.draft_n // 0) draft_accepted=\($x.draft_n_accepted // 0) accept_rate=\(if ($x.draft_n // 0) > 0 then ($x.draft_n_accepted / $x.draft_n * 100 | round / 100) else 0 end) text=\($s)"' "$work/out.json"
done
