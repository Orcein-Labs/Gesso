#!/usr/bin/env bash
# Clean-install check for Gesso.
#
# Usage: tools/clean-install-check.sh <model.gguf> <registry-url>
#
# Builds a fresh Julia depot, adds the General registry and the registry at
# <registry-url>, installs Gesso from it, starts the CPU chat server on
# <model.gguf>, sends one /v1/chat/completions request and checks the reply.
# Exits nonzero on any failure. Nothing is hardcoded to a machine: the depot,
# environment and log live in a temp directory that is removed on exit.
#
# Environment: JULIA (default: julia), PORT (default: a random high port).

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <model.gguf> <registry-url>" >&2
    exit 2
fi

MODEL=$1
REGISTRY_URL=$2
JULIA=${JULIA:-julia}

if [[ ! -f "$MODEL" ]]; then
    echo "clean-install-check: model not found: $MODEL" >&2
    exit 2
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/gesso-clean-install.XXXXXX")
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

export JULIA_DEPOT_PATH="$WORK/depot"
ENV_DIR="$WORK/env"
mkdir -p "$JULIA_DEPOT_PATH" "$ENV_DIR"
unset GESSO_GGML_DIR

echo "== install: registries and Gesso"
"$JULIA" --project="$ENV_DIR" -e '
    using Pkg
    Pkg.Registry.add("General")
    Pkg.Registry.add(Pkg.RegistrySpec(url=ARGS[1]))
    Pkg.add("Gesso")
    Pkg.status()
' "$REGISTRY_URL"

# The server script is taken from the installed package, not from this checkout,
# so the check covers what a user actually gets.
SERVE=$("$JULIA" --project="$ENV_DIR" -e 'using Gesso; print(joinpath(pkgdir(Gesso), "scripts", "gesso-serve.jl"))')
[[ -f "$SERVE" ]] || { echo "clean-install-check: installed package has no $SERVE" >&2; exit 1; }

PORT=${PORT:-$(( 20000 + RANDOM % 20000 ))}
LOG="$WORK/serve.log"

echo "== serve: CPU backend on port $PORT"
"$JULIA" --project="$ENV_DIR" "$SERVE" \
    --gguf "$MODEL" --cpu --port "$PORT" --model-id clean-install >"$LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done
if [[ $ready -ne 1 ]]; then
    echo "clean-install-check: server did not answer /v1/models" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
fi

echo "== request: /v1/chat/completions"
REPLY=$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"clean-install","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":64}')
echo "$REPLY"

object=$(jq -r '.object' <<<"$REPLY")
tokens=$(jq -r '.usage.completion_tokens' <<<"$REPLY")
text=$(jq -r '.choices[0].message | ((.content // "") + (.reasoning_content // ""))' <<<"$REPLY")

if [[ "$object" != "chat.completion" ]]; then
    echo "clean-install-check: unexpected object: $object" >&2
    exit 1
fi
if [[ "$tokens" -lt 1 ]]; then
    echo "clean-install-check: no completion tokens generated" >&2
    exit 1
fi
if [[ -z "${text// /}" ]]; then
    echo "clean-install-check: empty completion text" >&2
    exit 1
fi

echo "clean-install-check: ok ($tokens completion tokens)"
