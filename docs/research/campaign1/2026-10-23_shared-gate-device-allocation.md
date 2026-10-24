# Campaign 1: bind shared-gate scratch to the input device

## Objective
Remove the known global-default-device allocation in shared-gate scratch while preserving qualified arithmetic and launch policy.

## Ownership and scope
Luna implements in isolated `gesso-shared-gate-device-alloc-20261023`, branch `codex/c1-shared-gate-device-alloc-20261023`, from frozen `2108748f29b44bbfcba7c992e0f326f0abe2b4ae`. Only shared-gate allocation in `ext/lava_hybrid.jl` and narrowly relevant tests may change. Root owns canonical promotion. The consumed candidate, archived source, supplier, oracle, factor probes and Claude worktrees remain immutable.

## Predicates
Use the actual input backend's dispatch queue through the supported Lava allocation API for both Float64 scratch arrays. Preserve dimensions, element types, mathematics, numerical policy and public behavior. Review the real constructor/API contract and exact diff. CPU checks may establish syntax and API wiring; they cannot establish device behavior.

## Admission
No GPU job is authorized by this work item. After review and GPU availability, require actual public-entry numerical regression and proof that scratch uses the input device/queue. Distinguish one-device proof from any unavailable multidevice test. Integrate only into a newly bound source candidate; prior model/stress results do not transfer automatically.

## Receipt
BOUNDED GPU PROOF PASS, FINAL INTEGRATION OPEN. Local commit `774aa8054cccceb8395b72c8dab7551ec35ebbae` changes only `ext/lava_hybrid.jl`: derive the backend before scratch allocation and pass its dispatch queue to both Float64 constructors. Source/API reflection, strict parsing, lowering and diff checks pass without initializing Vulkan. Independent review confirms the queue-ownership API chain and unchanged math/launch geometry. No new tests were added to mirror the allocation calls.

Receipts: `block22_shared_gate_device_alloc_20261023_receipt.json`, `block22_shared_gate_device_alloc_20261023_review.json`, and `block22_shared_gate_device_alloc_20261023_commit.json`. The staged worktree is tracked-clean. Frozen `2108748` remains the active factor-probe source, unchanged. Final-candidate integration and model/stress requalification remain open; no prior model/stress result transfers to this commit.

The new isolated environment `env-campaign1-shared-gate-device-alloc-20261023` is prepared. Independent root QA verifies exact source-root substitution in Project/Manifest, unchanged old environment hashes, all 916 tracked source files with only the intended allocation delta, and actual imports from the new Gesso/vendor-Lava roots without initializing Vulkan. Full source binding: `block22_shared_gate_device_alloc_20261023_bound_source.json`; QA: `block22_shared_gate_device_alloc_20261023_env_manager_qa.json`. The environment snapshot records the existing exact regression command with its required Test/Random and hybrid globals.

Runtime session67719 ended rc0 under both locks, live512MiB guard,6GiB RAM/1GiB swap and600sec cap. Same-physical-adapter two-context fixture passes17/17: process-default context is unchanged, input/output arrays belong to a second caller-owned context, and three production public-entry calls match CPU bits. The existing exact shared-gate regression passes1658/1658. Root revalidates all916 source bytes afterward; tracked state is clean. Evidence: `shared-gate-device-alloc-gpu-1.log`, launch/lock receipts, and `shared-gate-device-alloc-gpu-1-manager-qa.json`. This establishes tested same-adapter logical-context behavior with source/API queue-ownership binding; multiple physical devices and universal failure cleanup are not proven. No timing, model, full-stress or release claim follows.
