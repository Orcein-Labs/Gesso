# Campaign 1: final documented workspace test environment

## Objective
Prepare the actual make test workspace with the qualified dependency versions and prove its local source bindings before final execution.

## Permitted files
Only ignored root Manifest.toml in isolated candidate ad47cb3, new external preparation/provenance receipts/runners and planning records. Preserve all912tracked source files, qualified external environments, consumed helpers and existing artifacts. No Project.toml, vendor, dependency declaration, user environment or production changes.

## Interfaces
The documented make test command, scripts/test.jl, root Pkg workspace and declared test/benchmark projects. Existing cached packages only; offline validation, no install or package resolution in qualified environments.

## Invariants
Preserve exact complete package records from the qualified external manifest. Normalize only the Gesso/Lava paths to this workspace's relative paths; add the already-declared Test stdlib record and the root workspace project hash from the prior byte-identical projects. Verify all common package records, source paths and workspace currency through local Pkg APIs. Root manifest is ignored metadata; actual runtime/test source and qualified external environments stay byte-identical. Stop and retain a negative if Pkg requires incompatible resolution.

## Tests
Verify root/test/benchmark/vendor Project bytes against the prior workspace; compare complete dependency records and ensure no registry-version drift. Verify Pkg identifies this Gesso root and its vendored Lava and sees a current workspace manifest. Recheck912source hashes and qualified external manifest bytes. Actual make test remains gated behind the earlier stress phases; staging metadata alone is not suite acceptance.

## Performance target
Lightweight CPU-only metadata preparation; no GPU initialization, package download, model load, throughput or test-pass claim.

## Expected artifact
Reversible ignored workspace manifest and exact preparation receipt, followed later by the unchanged documented command and its own full result ledger.

## Receipt
OPEN. Prior root manifest cannot be copied unchanged: CUDA_Compiler_jll and OrderedCollections differ from the qualified environment. The only additional package is the already-declared Test stdlib. All four Project files are byte-identical.

### Qualified offline workspace metadata
Ignored root Manifest.toml is staged from the qualified179package records, changing only Gesso/Lava paths to relative workspace paths, root project hash and the already-declared Test stdlib record. The two prior-workspace registry version differences were not adopted. All912tracked source files and both qualified external environment files remain unchanged. softmax-tree64-final-workspace-stage-1.json records exact hashes. No package resolution or installation occurred.

Initial check_final_workspace_v1.jl failed because it assumed Pkg.dependencies() includes the active root package. This is a check-harness defect; the workspace currency check had passed. Script and softmax-tree64-final-workspace-pkg-check-1.failure.json are preserved. New v2 checks root self-discovery and activates the declared test member for dependency bindings. softmax-tree64-final-workspace-pkg-check-2.json passes: workspace current, same root manifest bytes, Gesso/Lava bind to this candidate/vendor, compilerJLL0.6.3+1 and OrderedCollections2.0.2 match qualified records, Test stdlib available, neither Gesso nor Lava imported. This is metadata qualification, not full suite acceptance.

run_softmax_tree64_final_make_test_v2.sh is prepared and shell syntax checked, not executed. It pins source912, external environment, qualified workspace metadata, both53boundaries,16fresh matrix and both raw numeric-state/mutation gates before the unchanged make test invocation. It records initial/final workspace identities and uses offline Pkg with bounded threads and shared locks. V1 remains preserved/unrun.
