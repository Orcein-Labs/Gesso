# Campaign 1: maintained checked-softmax cost comparison

## Objective
Measure first-call compilation, warmed public-operator latency/allocation and managed GPU memory observations for the latest fixed-row-tree repair, comparing frozen parent6d with candidatead47 on identical CPU-checked inputs.

## Permitted files
New external versioned helper/auditor/captures and planning records only. Preserve both source trees, qualified environments, helpers, CPU oracle and existing artifacts. No source, vendor, compiler, dispatch or global precision changes.

## Interfaces
Ordinary public Gesso.softmax! with checked default behavior, preallocated GPU input/output and explicit completion synchronization. Existing CPU softmax supplies numerical comparison only. Public Lava memory-usage and compiled-program inspection APIs supply observations. No process-local replacement or custom denominator kernel in this helper.

## Invariants
Same eight seeded Float32 geometries, score bytes and CPU reference bytes on both roots. Poison every output with a finite negative sentinel outside timing; verify every result, input preservation and repeated output bits. Data upload, poison setup, readback and raw capture remain outside operator timers. Preserve actual default GC; collect between bounded warmup batches/cases and report its timing separately. No model or throughput measurement.

## Tests
Execution follows original boundary/fresh matrix qualification. Each case: first checked call, at least25checked warmup calls over at least1second, then50synchronized measured calls with archived full outputs. Existing CPU absolute tolerance1e-6 remains unchanged. Independently recompute all archived numerical/measurement summaries. Candidate actual fixed64 FloatControls32/NoContraction shader is archived/validated; parent contains no new row-sum kernel. Reject a copied raw output mutation with repaired digest and green flags retained. Retain numerical negatives and investigate material regressions.

## Performance target
No speedup target. Record complete distributions, compile/GC/host allocation data, OS peak RSS and labeled managed-GPU live-memory snapshots; do not present snapshots as continuously sampled physical GPU peak. Logical denominator bytes=4*rows. First-call timing excludes package/device setup and preallocation. This comparison isolates the latest softmax repair, not the total cost of earlier RMS/GDN/RoPE/quant repairs. Full engineering/resource receipt remains separately mandatory.

## Expected artifact
Qualified observational parent/candidate operator-cost receipt with explicit limits, actual source/program identities and effective raw mutation. Raw capture <=64MiB/process, <=1200seconds,16Julia threads,4800MiBGPU, bounded native threads and shared locks. No execution while the current16-process matrix owns the stress stage.

## Receipt
OPEN. Preparation only; no GPU cost process launched.

### Preparation
softmax_operator_cost_v1.jl is written and syntax checked, with public APIs checked against the vendored source. Context-bound post-GC synchronization uses Lava.LavaBackend(ctx). No GPU cost process has run. It saves eight fixtures, eight first calls and400warm outputs, bounded to64MiB. Independent full cost auditor/mutation remain unwritten. This is only the latest softmax delta, not the complete final engineering/resource receipt.
