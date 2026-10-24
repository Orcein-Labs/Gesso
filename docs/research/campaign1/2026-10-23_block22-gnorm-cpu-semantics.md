# Campaign 1: CPU-fast GN arithmetic control on matched block22 operands

## Objective
Build a bounded external GPU GN operator that matches the actual CPU fast entry on the qualified archived operands, then use it to justify one later model discriminator. No production source change or model run is included in preparation.

## Evidence and target
Factor capture v4 with independent auditor v7 passes: all three unchanged controls and three observations reproduce the archived GPU endpoint exactly; factors repeat, inputs are preserved, actual shaders validate, and a repaired-descriptor factor mutation is rejected. Per repetition, normalized values differ from CPU in 1702/4096, gate values in 1832/4096, and final values in 2441/4096. Captured normalized times gate reproduces final output exactly on these normal-valued operands. CPU/GPU squared sums differ in 26/32 heads. These are bounded diagnostics, not proof of the inverse dataflow or of which operation changes a model token.

The authority is the actual maintained `_gdn_gnorm_merge!` fast entry in `src/Inference/cpu_decode.jl`: serial Float32 square/add per head; Float32 mean/epsilon and square root denominator; division by that denominator, then Float32 scale multiplication; pinned Base Float32 SiLU gate; final Float32 multiplication. CPU general/public RMS arithmetic is a different contract.

## Ownership and scope
Luna owns new `block22_gnorm_cpu_semantics_*` S/R helpers, runner and preparation receipts. Root owns this work item, independent review, launch and admission. Maintained2108748, archivedad47, allocation774aa80, all consumed helpers/captures/environments, model and frozen oracle remain immutable. Do not copy CPU-computed intermediate/output tensors into the GPU candidate. Only original core/z/scale and native epsilon are inputs; CPU factors are references.

## Predicates
1. Revalidate source/environment, replay2 and accepted CPU reference/v4/factor receipts. Preserve head-major layout and prove the CPU interleaved conversion inverse.
2. Reuse qualified Base-exp helper and physical wide-division techniques where needed. Carry wide stages through real Float64 storage so the compiler cannot silently narrow the intended arithmetic. Preserve CPU Float32 rounding boundaries; do not replace the serial CPU sum with wide or tree summation. New control kernels may use explicit arithmetic policy; do not change the stock observer/control policy or tolerance.
3. Prefer a small readable kernel sequence. Archive actual SPIR-V/LLVM and verify intended operations/widths, geometry and policy. Capture squared sum, denominator, normalized term, gate and merged outputs so any failed target identifies a stage.
4. Use at least the qualified Lava-core case and archived CPU-core case with their correct paired inputs, three repetitions each. Reference actual CPU fast calls and accepted raw targets, not a broad synthetic approximation. Require bit-exact merged results before any model intervention. Mark fixture-only limits explicitly; no universal nonfinite/subnormal or family claim.
5. Independent raw/source/program audit and meaningful repaired-descriptor mutation are required. Strict parse/lowering, actual runtime builder/constructor checks and independent static review precede one root-admitted GPU attempt. No self-retry, source integration or model launch.

## Resources and cost
Existing pause marker and timed/GPU locks; explicit6GiB RAM/1GiB swap,600sec timeout, live256MiB minimum-free guard, raw cap4MiB and Diskguard. The initial1MiB estimate was expanded during CPU preparation, before any GPU attempt, to retain all wide/F32 stage traces and actual program archives for both roles and three repetitions. Require a prelaunch byte estimate; numerical criteria and references are unchanged. No model payload loading, timing comparison or throughput claim. Measure final maintained cost only after numerical/model admission.

## Receipt
CLOSED for the bounded operator fixture: clean V3 capture exited0, with six of six actual CPU-fast outputs and every captured Float32 stage bit-exact, stable repetitions, 15 compiled programs and 846,630 raw/program bytes across 62 unique files. Independent final audit `block22_gnorm_cpu_semantics_validation_v1.json` is `PASS_CPU_SEMANTICS_CAPTURE`, SHA256 `ff07889dd629bd238268f42ac1118bc71b53b2d252e4a311da82132226ea7958`. Root rehashed all 62 artifacts, compared all six final outputs and 48 common CPU/GPU stage byte sequences, and independently disassembled the 15 programs.

Root operator admission `block22_gnorm_cpu_semantics_operator_admission_v1.json`, SHA256 `7c2f150baacbf7a71820eefad123cc373eb6a056f17e5f79b280609f137f2219`, binds the frozen helper, runner, raw capture, independent auditor/review and postcapture admission. It permits preparation of the separate model discriminator; it does not authorize model execution, maintained-source integration or full stress acceptance. The original Campaign1 model gate remains9/13; source-model stress, documented make/test/format, capability and cost gates remain open.

Preserved failures: V1 exited1 before device execution because Julia rejected `open(...,"x")`; V2 completed six exact cases and all programs but exited1 while building its final summary because `repeat_stable` was undefined. V3 repairs the report path with CPU summary smoke checks. Failed captures remain separate; only clean V3 has final fixture admission.

The scope is the two archived paired inputs, three repetitions each, with nv32/dv128 and normal values. There is no universal nonfinite, subnormal or hardware claim. Float64 host-operation comparisons are diagnostic; actual Float32 boundaries and outputs are exact on these fixtures. Runtime ndrange is unavailable in LaunchPlan and remains bound to pinned calls; actual local workgroups match SPIR-V. The qualified Base-exp gate contains nine intended `OpExtInst Fma` operations, with zero in the other 14 programs. Configured RAM/swap caps do not establish measured peak memory. A copied single-scalar stage mutation with a repaired descriptor hash was rejected semantically.

Independent CPU contract review `block22_gnorm_cpu_semantics_contract_review_v1.json` confirms the actual Julia1.12.6 compiled CPU Float32 order, direct quotient and pinned Base-exp gate. Independent input pairing `gnorm-semantics-independent-input-bindings-v1.json` binds CPU/CPU and Lava/Lava replay inputs to their distinct CPUv4 fast-entry targets. Candidate scratch uses the input backend queue. Continue in `2026-10-23_gnorm-cpu-semantics-model-control.md`.
