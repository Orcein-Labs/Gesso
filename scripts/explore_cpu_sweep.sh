#!/usr/bin/env bash
# Thread-count and pinning sweep for scripts/explore_cpu_threading.jl.
# Each timed run takes the CPU bench lock and records uptime first.
#
#   scripts/explore_cpu_sweep.sh <out-dir> [thread-counts...]
#
# Runs EXP_MODE=forkjoin and EXP_MODE=pool for each thread count, unpinned;
# then pinned runs at 16 and 18 threads. Output: <out-dir>/<label>.txt and
# <out-dir>/uptime.log.
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?usage: explore_cpu_sweep.sh <out-dir> [thread-counts...]}"
shift
counts=("$@")
if [ ${#counts[@]} -eq 0 ]; then counts=(8 12 16 18 24 32 36); fi
mkdir -p "$out"
cd "$here" || exit 1

run_one() {
    local label="$1" mode="$2" nt="$3" pin="$4"
    {
        echo "== $label  $(date -Iseconds)"
        uptime
    } >> "$out/uptime.log"
    EXP_MODE="$mode" EXP_PIN="$pin" flock /tmp/gesso-cpu-bench.lock \
        nice -n 5 julia --project=. -t "$nt" scripts/explore_cpu_threading.jl \
        > "$out/$label.txt" 2>&1
    echo "exit=$?" >> "$out/$label.txt"
    echo "$label done $(date -Iseconds)" >> "$out/uptime.log"
}

for nt in "${counts[@]}"; do
    run_one "fj_t${nt}" forkjoin "$nt" 0
    run_one "pool_t${nt}" pool "$nt" 0
done
for nt in 16 18; do
    run_one "fj_pinned_t${nt}" forkjoin "$nt" 1
    run_one "pool_pinned_t${nt}" pool "$nt" 1
done
echo "sweep complete" >> "$out/uptime.log"
