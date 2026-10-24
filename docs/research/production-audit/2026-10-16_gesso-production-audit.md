> Status 2026-10-24: the remediation branch is merged into master (the KernelAbstractions cap was dropped in favour of the vendored, patched Lava; the CI checks it names are fixed there too).

# Gesso production engineering audit

- Date: 2026-10-16
- Mode: 1 (full audit), report-only. No changes were made to the repo.
- Remediation pass: 2026-10-17, on branch `claude/pea-fix-20261017` in a separate worktree off `37d87da`. Not pushed. See section 3.
- Target: local `master` at `37d87da`. It is 11 commits ahead of `origin/master` and 1 behind (`38ca199`).
- Scope: Gesso's own code, tests, CI, docs and repo hygiene, plus how Gesso uses Lava. Lava internals were not audited.

## 1. Executive result

Gesso's core design is sound and well-tested on the CPU. The CPU F64 oracle, the logged-fallback law (`@gfallback`), typed errors and per-call receipts are real and enforced. The test suite is large: 124 files and about 8,700 assertions.

The release gate is the problem. Every recorded CI run on `master` has failed since 2026-09-29. The test step itself passes, but `make test` and the bench smoke both fail because the Lava extension cannot precompile on a headless runner, and the format check fails because the tree has never been formatted (118 files differ from the CI formatter's output). The 11 newest local commits (+1476/-208 lines, mostly Lava and MoE kernels) are unpushed and have never been through CI.

Main code findings:
- A profane message is hidden in a hex comment that pretends to be a SPIR-V remnant. (Withdrawn 2026-10-17: the owner says it is an intentional easter egg.)
- The chat server re-decodes the whole output on every token, which is quadratic in output length.
- The CUDA hot-expert cache silently drops to the CPU path under memory pressure.
- The ggml ABI read has no sanity check.

Counts: 1 Blocker, 6 Important (1 later withdrawn), 13 Advisory.
Release decision at audit time: **Blocked** until the gate is green on the commit being released.

After the 2026-10-17 fix pass (section 3):
- B1, I2, I3, I5 and I6 are fixed on an unpushed branch, and I1 is withdrawn.
- Every gate check passes locally on the branch tip. CI has not run it yet.
- The decision is now **Blocked, pending push and one green CI run** (section 6).

## 2. Ranked findings

Ranking is impact × exposure × confidence. Each high-severity claim was challenged; the challenge is recorded under Evidence.

### B1. The release gate has never been green on master

- **Campaign:** E (Hygiene) / D (Reliability)
- **Severity:** Blocker
- **Location:** `.github/workflows/ci.yml`, `Makefile`, `scripts/test.jl`, `test/Project.toml`; the format job runs JuliaFormatter 2.14.0 over the whole tree
- **Observed problem:** All 13+ recorded CI runs on `master`, back to 2026-09-29, ended red. The latest (run 38329137328, on `38ca199`) shows:
  - The direct `test/runtests.jl` step passes: 8723 pass, 60 broken, 8783 total, 8m06s.
  - `make test` fails: `GessoLavaExt` will not precompile. GLFW init throws `GLFWError(65550, "X11: The DISPLAY environment variable is missing")` and `vkCreateInstance` finds no drivers.
  - `bench-smoke` fails the same way.
  - The format job fails. **Correction (2026-10-17):** the first draft said three files. That was wrong: those three names were only the last `Formatting ...` lines of the log. Running the same formatter locally reformats 118 files, so the format check has never passed on any commit.
- **Evidence:** `gh run list` / `gh run view` on the repo. AGENTS.md names `make test` and `make format` as the required checks.
- **Consequence:**
  - There is no working signal for regressions. A new red run looks the same as the usual red.
  - The documented developer loop (`make test`) fails on any machine without a display.
  - Agents told to "make test pass" cannot meet the bar honestly.
- **Smallest credible root-cause remedy:**
  1. In CI, give the Lava extension what it needs to load: install a software Vulkan driver (Mesa lavapipe) and run under `xvfb-run`. Or, if Lava allows it, load Lava without GLFW, which is a Lava-side ask.
  2. Pin the formatter version, then run it once over the whole tree and commit the result.
  3. Make the `test` and `format` jobs required on `master`.
- **Verification:** One green CI run on the released commit with all three jobs passing.
- **Change risk:** Low. CI and formatting only; no runtime code changes.
- **Second cause:** This audit's local baseline hit another precompile failure. Lava's `KA.adapt_structure` method overwrites the KernelAbstractions one (`Method overwriting is not permitted during Module precompilation`). That is the piracy DEBT.md already records, so even with a display and a driver the extension cannot precompile. The extension failure is not fatal to the test step, which is why `runtests.jl` passes, but `make test` treats it as an error.
- **Status:** Fixed on the remediation branch (section 3), pending a first CI run. The Lava piracy is capped, not paid off: the payoff (drop the method in Lava, then lift the cap) still holds. The GLFW/headless cause is now in DEBT.md.

### I1. Profanity hidden in a fake "SPIR-V remnant" comment (withdrawn)

- **Campaign:** E (Hygiene)
- **Severity:** Important
- **Location:** `ext/GessoLavaExt.jl:209-215`, added in commit `d77bce4` (2026-10-08, "Land Regime II Lava decode parity...")
- **Observed problem:** The comment calls the hex block a "Dead SPIR-V walk remnant (51 B)... pipeline fingerprint dump". Decoded, the bytes are an obscene message aimed at NVIDIA.
- **Evidence:** Decoding the hex in the comment gives readable ASCII; it is not SPIR-V. A SPIR-V module would start with the magic word `0x07230203`, and this one does not.
- **Consequence:**
  - The repo is public, so this is a reputational risk for the project and its author.
  - The comment misleads readers into treating the bytes as engineering data.
  - It is in the file most likely to be read by Lava and Vulkan collaborators.
- **Smallest credible root-cause remedy:** Delete the comment block. Check the other files `d77bce4` touched for similar hidden content. Optionally rewrite history, if the author wants the string gone from the log as well.
- **Verification:** Decode every hex-looking literal in `ext/` and `src/` and confirm none decodes to text.
- **Change risk:** None. It is a comment.
- **Status:** Withdrawn: user decision, intentional easter egg. The comment stays.

### I2. The chat server re-detokenizes the whole output on every token

- **Campaign:** C (Performance)
- **Severity:** Important (medium confidence; cost not measured)
- **Location:** `src/serve/handlers.jl:170-191` (`_on_token`, `_sync_text!`); the `max_tokens` default in `handle_chat`
- **Observed problem:**
  - `_on_token` calls `_sync_text!` on every token.
  - `_sync_text!` calls `turn.model.detokenize(turn.ids)` over the full generated sequence, then throws away all but the new bytes. That is O(n) work per token and O(n²) per response.
  - When the client sends no `max_tokens`, `max_new` is set to the full remaining context, so n can reach the context length.
- **Evidence:** Code reading. Rough estimate: at 4k output tokens, each detokenize is a few hundred µs, about 1% of a 20 ms token. At 32k tokens it is several ms per token, around 10% or more, and the waste grows with every token. Not profiled, because profiling needs a loaded model.
- **Consequence:** Long streamed answers slow down as they grow, and the server thread spends that time holding the single generation lock.
- **Smallest credible root-cause remedy:** Detokenize incrementally. Keep a short tail window (the last few ids, enough to complete a UTF-8 sequence) and decode only the window. Or have the tokenizer return bytes per id and append them. The existing trailing U+FFFD hold logic stays the same.
- **Verification:**
  - A unit test streaming 10k ids through a fake detokenizer, asserting time per token is flat.
  - A byte-identical output check against the current method on the SmolLM2 gate.
- **Change risk:** Low to medium. UTF-8 boundaries and byte-fallback tokens must still be handled, and the existing test for the U+FFFD hold covers that.
- **Status:** Fixed in `04832ca`. Measured with a counting detokenizer on a 4,216-id reply: 20,146 ids decoded, against 8,893,652 before. The new test fails on the old code.

### I3. The hot-expert cache silently falls back to the CPU under memory pressure

- **Campaign:** D (Reliability)
- **Severity:** Important
- **Location:** `ext/cuda_hot_experts.jl:282-320` (`_hot_alloc!`), `:272` (`_hot_pin_slabs!`), `:1138` (`_hot_stats`)
- **Observed problem:**
  - When the hot-expert pool cannot be allocated, a layer runs the CPU-only expert path for `_HOT_RETRY_TOKENS` (64) tokens and then retries.
  - A failed page-lock is logged only at `@debug`.
  - Neither event goes through `@gfallback`. Hit, miss and swap counts exist in `_hot_stats` but are not written to receipts.
- **Evidence:**
  - Code reading.
  - `@gfallback` is used in exactly one place in the repo (`src/Inference/gguf_quant_matrix.jl:104`).
  - The README promises "Gesso never falls back to the CPU without telling you". The code comment itself records the cost: decode dropped from 62 to 34 tok/s when the layer was retired.
- **Challenge:** Hot experts are opt-in, and the CPU path is the oracle, so the output is still correct within tolerance. But GPU hits and CPU misses round differently, so greedy output can depend on the state of device memory from earlier requests. And the speed drop is invisible to the user.
- **Consequence:** Speed and greedy token choices can change between identical requests with no record of why, which goes against the project's own fallback law.
- **Smallest credible root-cause remedy:**
  - Route the pool-OOM and pin-failure paths through `@gfallback`, once per layer per retry window.
  - Add `_hot_stats` totals (hits, misses, layers on the CPU path) to the session receipt.
- **Verification:** A forced-OOM test (shrink the reserve) asserts that a fallback event and a receipt field appear.
- **Change risk:** Low. Logging and receipts only.
- **Status:** Fixed in `4943f9c` (logging). Plan-only: putting `_hot_stats` (pool hits, misses, fallback episodes) into receipts. The CPU/GPU drift claim still needs a forced-OOM GPU run (section 7).

### I4. The 11 local master commits have no CI and no push

- **Campaign:** D (Reliability) / E (Hygiene)
- **Severity:** Important
- **Location:** `844d9e8..37d87da` on local `master`. Touches 18 files, +1476/-208.
- **Observed problem:**
  - Local `master` holds the Lava decode arena, repacked Q8_0 GEMV, fused GDN decode, MoE router and shared-expert changes. None of it is on `origin`.
  - `origin/master` has one commit that local lacks (`38ca199`, which removes the README "Training" section), so the two have diverged.
  - The last commit also relaxes the 35B profile gate from exact greedy ids to a 0.05-logit tolerance. The commit message documents why and notes that a uniform 3% scale error still passes that gate.
- **Evidence:** `git log origin/master..master`, `git diff --stat`. The CPU-only baseline for this audit is in section 4. GPU paths were not run, as the brief required.
- **Consequence:**
  - The newest kernels have never been tested anywhere except the author's machine.
  - The local copy is a single point of loss.
  - Merging will need a README conflict resolved.
- **Smallest credible root-cause remedy:** Rebase onto `origin/master` (taking `38ca199`'s README change), push to a branch, and merge only after B1's gate is green.
- **Verification:** Green CI on the pushed tip, and the `master` branches no longer diverge.
- **Change risk:** Low.
- **Status:** Open, by instruction: no push, rebase or merge was done. Reconciling plan in section 3.

### I5. The ggml type-traits struct is read at fixed offsets with no sanity check

- **Campaign:** D (Reliability)
- **Severity:** Important
- **Location:** `src/Inference/gguf_import.jl:118-127` (`_ggml_traits`), used by `_dequant_f32` at `:448-469`
- **Observed problem:**
  - `blck_size`, `type_size` and the `to_float` function pointer are read at byte offsets 8, 24 and 40 of the ggml type-traits struct in whatever llama.cpp build `GESSO_GGML_DIR` points to.
  - Nothing checks them, even though `src/Inference/gguf_quants.jl` already has the correct block sizes and bytes per block for every type.
- **Evidence:** Code reading. DEBT.md entry 2 records the ABI fragility.
- **Consequence:** If a llama.cpp update changes the struct layout, Gesso calls a wrong function pointer or uses wrong sizes. The result is a segfault or silently wrong weights, not a clear error.
- **Smallest credible root-cause remedy:** After reading the traits, compare `blck_size` and `type_size` against Gesso's own tables, and throw a typed error naming `GESSO_GGML_DIR` if they disagree. This is a few lines, and it is worth doing well before DEBT.md's full payoff (native dequant for every format).
- **Verification:** A unit test that feeds a fake traits pointer with shifted fields and expects the typed error.
- **Change risk:** Very low.
- **Status:** Fixed in `7d27442`. The check passes against a local llama.cpp master build for all 17 table entries. The DEBT.md payoff (native dequantization on the load path) still holds.

### I6. The chat server has no timeouts and no connection cap

- **Campaign:** D (Reliability)
- **Severity:** Important if bound to a non-loopback address; Advisory as shipped (it defaults to `127.0.0.1`)
- **Location:** `src/serve/server.jl` (one `@async` task per accepted connection), `src/serve/http.jl` (`MAX_HEADER_BYTES` 64 KiB, `MAX_BODY_BYTES` 32 MiB, no read or idle timeout; `_read_chunked` gives each chunk-size line a fresh budget)
- **Observed problem:** A client that opens connections and sends bytes slowly (slowloris) holds tasks and sockets forever. Nothing limits the number of open connections.
- **Evidence:** Code reading.
- **Challenge:** The default bind is localhost, and generation is single-flight behind `srv.gate`, so the exposure is low until someone passes `host="0.0.0.0"`. Counted as Important because an OpenAI-compatible server is exactly what people expose on a LAN.
- **Consequence:** Easy denial of service once the server is exposed.
- **Smallest credible root-cause remedy:**
  - Add a header/body read deadline (for example 30 s) and an idle keep-alive timeout.
  - Cap concurrent connections with a semaphore.
  - Document that the server is not hardened for public exposure.
- **Verification:** A test that opens a socket, sends one header byte, and asserts the server closes it after the deadline.
- **Change risk:** Low.
- **Status:** Fixed in `ca427db`. Plan-only: a write-side stall timeout (see section 3).

### Advisory findings

| ID | Campaign | Location | Problem, and the smallest remedy | Status |
|---|---|---|---|---|
| A1 | E | 120 tracked files, all under `docs/` (mostly `docs/archive/2026-10_regime-i/receipts/`), plus `DEBT.md` and `docs/research/CYAN_ELASTIC_ORCHESTRATION.md` | Absolute machine paths (home directory, drive mounts) in a public repo. None appear in code, tests, scripts or tomls. Remedy: one sed pass that rewrites them to repo-relative or `<models>` placeholders, plus a CI grep that fails on new ones. | Partly fixed in `72f0969`: CI now fails on machine paths in code, tests, scripts and config. The docs sweep is plan-only (section 3). |
| A2 | D | `benchmark/Project.toml` | Lava is pulled by URL with no `rev`, so bench results can't be reproduced. Remedy: pin the same `rev` as `test/Project.toml` (`11c7e31`). | Fixed in `3f81fb7`. |
| A3 | A | Device scratch in `ext/cuda_ops.jl`, `ext/lava_ops.jl` | Scratch buffers are shared per device, so concurrent decodes would collide. Not exposed today, because the server runs one generation at a time. Remedy as in DEBT.md: per-session scratch, or an explicit single-flight assertion in the backend. | Skipped: a design change, and not exposed while the server is single-flight. Stays in DEBT.md. |
| A4 | B | `DEBT.md` entry 3 | Stale. It says matmul for IQ2_S / IQ3_S / IQ1_M still goes through `to_float`, but native Q8_K kernels now serve all three (`src/Inference/gguf_quants.jl:290`, `src/Inference/quant_cpu_simd.jl:524`). Entry 2 is still right: `_dequant_f32` uses `to_float` for every non-F32 type. Remedy: retire entry 3, and narrow entry 2 to dequant (embeddings, full arrays). | Fixed in `49d9dd7`. |
| A5 | E | Root: `GESSO_STACK_EVIDENCE_MAP.md` (58 KB), `RPD_SOP.md` | Docs at the repo root, with upper-case names. Remedy: move them to `docs/` with `git mv`. `RPD_SOP.md` is a mirrored copy shared with three other repos, so move it in all four or leave it with a note. | Plan-only (it is a move; section 3). |
| A6 | E | `docs/` | There is no `docs/README.md` index. Names break the kebab-case rule: `Gesso_Stack.md`, `Harpe_Stack_old.md` (also uses the banned word "old"), `Gesso_musings.md`, `Native_Julia_Kernel_Autotuning_North_Star_README.md`, `PACKAGING_REGIME_I.md`, `DECISION_PACKETS.md`, most of `docs/research/*` and `docs/goals/*`, and archive receipts such as `make-format-final-arm1.log` ("final"). Remedy: add the index, then rename in one `git mv` commit, with the plan shown first. | Plan-only (renames and moves; section 3). |
| A7 | E | On-disk root (untracked) | `outputs(from chatgpt)`, `work(from chatgpt)`, `snapshots`, `research`, `libs` and `.local` are correctly ignored. `Performant Julia toolchain/` is ignored only by an uncommitted `.gitignore` change. `Julia Compiler Optimizations for Gesso/` is untracked and not ignored. There are empty dirs (`.agents`, `.aws`, `.codex`, `.freebuff`, `kvfold`), and names with spaces and brackets. Nothing junk is tracked. Remedy: the owner commits the `.gitignore` change and moves the scratch dirs under `research/`. | Skipped: the owner's uncommitted work in the main checkout, which this pass did not touch. |
| A8 | E | `ext/GessoLavaExt.jl` aside, docs | Two more profanities in docs: `docs/Gesso_Stack.md:3625` and `docs/research/EXOTIC_CAPABILITY_MASTER_LEDGER.md:110`. Remedy: reword them. | Skipped: needs an owner decision, given the I1 decision. |
| A9 | E | `AGENTS.md` (first lines) | The public charter opens with a Codex-only banner ("SPECIAL NOTE FOR CODEX!! ... CHEAP LUNA 6 MAX"). Remedy: move agent-specific notes to local config, or reword them as a neutral note. | Skipped: the owner's agent configuration. |
| A10 | E | `.github/ISSUE_TEMPLATE/work-item.md` | The file uses issue-form YAML (`name:`, `body:`, `type: textarea`) but has a `.md` extension. GitHub only reads forms from `.yml`, so it shows up as raw text. Remedy: rename it to `work-item.yml`. | Plan-only (it is a rename; section 3). |
| A11 | B | 74 distinct `GESSO_*` env vars in `src/`, `ext/`, `scripts/`; 66 of them are not mentioned in `README.md` or `docs/ARCHITECTURE.md` | Configuration is spread across undocumented env switches, which makes runs hard to reproduce. Remedy: one generated table (name, default, owner file), plus a receipt field that lists the non-default `GESSO_*` values. | Partly fixed in `78f666f`: generated `docs/env-vars.md` (87 names with the files that read them), checked in CI. The defaults column and the receipt field are plan-only. |
| A12 | C | `benchmark/results/` | The newest record is `2026-10-03.tsv`. The README's 35B CUDA numbers and the 11 local perf commits have no committed bench record; evidence lives in research notes and commit messages. Remedy: write a bench TSV row with each perf commit, as already done through 2026-10-03. | Skipped: needs GPU bench runs. |
| A13 | A/E | `ext/cuda_ops.jl` (3424 lines), `src/Inference/Inference.jl` (1883), `docs/ARCHITECTURE.md`; 50 local branches (about 17 stale `worktree-agent-*`, 2 `wip/opus/*`); about 17 worktrees | Very large files, and an architecture doc written as a phase changelog. `Planning`, `Agents` and `CAPI` are empty modules. Branch sprawl. `scripts/` mixes tools with one-off experiments (`explore_*`, `p1_storage_any_cost.jl`). Remedy: split `cuda_ops.jl` by op family when next touched; rewrite ARCHITECTURE.md as current state only; prune merged branches; move experiments to `research/`. | Plan-only (section 3). |

Clean areas (one line each):
- Typed errors and `LoweringNotImplemented`: consistent. Lowerings decline through the one sanctioned path.
- Receipts: every engine call emits one. `_audited_owned` writes a failure receipt and then rethrows. Nothing is swallowed except sink errors, which is by design (`_safe_emit!`).
- `to_device` across the two backend extensions: the earlier load-order wart is fixed. Both extensions attach methods to one function object (`ext/GessoCUDAExt.jl:197-208`, `ext/GessoLavaExt.jl:190`).
- How Gesso uses Lava: the constructor probes `vk_context()` and throws a typed `ERR_RESOURCE_LIMIT`. GPU tests become named skips when there is no device.
- Dependencies: the root `Project.toml` is minimal (stdlib plus JSON), and CUDA and Lava are weakdeps.
- Code paths: no machine paths in `src/`, `ext/`, `scripts/`, `test/` or `benchmark/`.
- Test hygiene: real-model gates skip unless `GESSO_SMOLLM2_DIR` is set, and the CPU oracle path is covered.

## 3. Remediation record

The audit itself was report-only. The fix pass ran on 2026-10-17, authorized by the owner:
- Branch `claude/pea-fix-20261017`, in its own worktree, branched from local `master` at `37d87da`.
- The main checkout and the other agents' worktrees were not touched.
- Nothing was pushed, rebased onto origin, merged, deleted or moved.
- GPUs were hidden for every run, so no GPU job was started.

### Commits (oldest first)

| Commit | Finding | Change |
|---|---|---|
| `3f81fb7` | B1, A2 | Cap KernelAbstractions at 0.9.42 in the test and bench envs, which avoids the method overwrite with Lava. Pin the bench env's Lava to the test env's revision. |
| `3443655` | B1 | CI test and bench jobs run under `xvfb-run`. `scripts/test.jl` (`make test`) re-runs itself under `xvfb-run` on a headless Linux box, or warns when it can't. |
| `c50f076` | B1 | `scripts/format.jl` pins JuliaFormatter 2.14.0 in a shared env and also formats `ext/`. The CI format job now runs that script plus `git diff --exit-code`, so CI and `make format-check` are one check. |
| `04832ca` | I2 | Streamed tokens decode a short window of ids (prefix offset / read offset), not the whole reply. Incomplete UTF-8 is held back. Adds a test that counts decoded ids. |
| `4943f9c` | I3 | Hot-expert pool failures (too little device memory, OOM) and the pin failure log through `@gfallback`, once per layer per episode. Adds a test that needs no device. |
| `7d27442` | I5 | The ggml traits read is checked against Gesso's own block table (17 types), with range checks for unknown types. A mismatch throws a `GessoError` that names `GESSO_GGML_DIR`. The test is ungated and runs without the library. |
| `ca427db` | I6 | The server gains `read_timeout` (default 30 s per request, covering idle keep-alive and trickled heads) and `max_connections` (default 64; past that, 503 and close). Adds tests. |
| `49d9dd7` | A4, DEBT | Retire the stale DEBT.md `to_float` matmul entry and narrow the llama.cpp entry. Record the KernelAbstractions cap and the GLFW/headless cause. |
| `72f0969` | A1 (part) | CI step that fails on home-directory or drive-mount paths in `src`, `ext`, `scripts`, `test`, `benchmark`, `.github`, `Makefile` and tomls. |
| `78f666f` | A11 (part) | `scripts/env-vars.jl` generates `docs/env-vars.md` (87 `GESSO_*` names, each with the files that read it). Adds `make env-vars`, a CI `--check` and a README link. |
| `27c7dbe` | B1 | Whole-tree reformat with the pinned formatter: 119 files, +16,113/-4,558, no behaviour change. **Kept last and separate so it can be dropped** (see "Owner decisions"). |

Every commit message ends with the Co-Authored-By line. New files are mode 100644. No machine paths are added anywhere in the series (checked with `git log -p` and a grep).

### Finding by finding

| Finding | Outcome |
|---|---|
| B1 | Fixed on the branch. A first CI run is still needed to prove it. |
| I1 | Withdrawn: user decision, intentional easter egg. |
| I2 | Fixed (`04832ca`). |
| I3 | Fixed (`4943f9c`). Receipts for `_hot_stats` are plan-only. |
| I4 | Open by instruction. Plan below. |
| I5 | Fixed (`7d27442`). |
| I6 | Fixed (`ca427db`). The write-side stall is plan-only. |
| A1 | Code guard fixed (`72f0969`). The docs sweep is plan-only. |
| A2 | Fixed (`3f81fb7`). |
| A3 | Skipped: a design change, not exposed today. |
| A4 | Fixed (`49d9dd7`). |
| A5, A6, A10, A13 | Plan-only: moves and renames are not allowed in this pass. |
| A7 | Skipped: the owner's uncommitted work. |
| A8 | Skipped: an owner decision, in line with I1. |
| A9 | Skipped: the owner's agent configuration. |
| A11 | Name index fixed (`78f666f`). Defaults and the receipt field are plan-only. |
| A12 | Skipped: needs GPU bench runs. |

### Owner decisions still needed

- **The reformat commit (`27c7dbe`).** Without it, the CI format job stays red. With it, every open branch (local `master`'s descendants and the Codex worktrees) will conflict on most files when rebased. Options:
  - (a) Land it now, and have each branch run `julia scripts/format.jl` before rebasing.
  - (b) Drop it (`git rebase --onto 78f666f 27c7dbe`), land the rest, and re-run the formatter once the open branches are in. Then add its hash to a `.git-blame-ignore-revs` file.
- **The KernelAbstractions cap** is a stopgap. Lift it once Lava stops defining `adapt_structure(ConstAdaptor, ReshapedArray)`. That is a Lava-side change, not made here.

### I4: what reconciling needs (not done)

1. Local `master` (`37d87da`) and `origin/master` (`38ca199`) have diverged by 11 and 1 commits. A test merge (`git merge-tree`) conflicts only in `README.md`:
   - Local `844d9e8` rewrote the Training paragraph as a `## Training` section.
   - `38ca199` deletes that paragraph ("Lux.jl handles training").
   - Resolution: take origin's deletion and remove the `## Training` section. That is the owner's newer stated intent.
2. Rebase local `master` onto `origin/master` and resolve the conflict once.
3. Then rebase this branch onto the new `master`. Its README edit is far from the Training section, but the same conflict comes through if `master` was not rebased first.
4. Push to a branch, not `master`. Let CI run all jobs, and merge only on green.
5. In GitHub settings, make the `test` and `format` jobs required on `master`.

### Plan-only items (nothing moved, renamed or deleted)

- **A1 docs sweep.**
  - Scope: 2,451 lines in 119 docs. The prefixes are mostly the home `Documents` directory (2,484 hits), then `.julia` (77), `.local` (34) and a few drive-mount paths.
  - Plan: one `sed` pass that maps each prefix to a placeholder (`<home>/...`, `<julia-depot>/...`, `<repo>/...`), reviewed per directory. Then widen the CI guard from `72f0969` to cover `docs/`.
  - Most hits are in archived receipts, so the owner should decide whether archived evidence is rewritten or left as is with a note.
- **A5.** `git mv GESSO_STACK_EVIDENCE_MAP.md docs/gesso-stack-evidence-map.md`. `RPD_SOP.md` is a mirrored copy shared by four repos, so move it in all of them or leave it with a note.
- **A6.**
  - Add `docs/README.md` indexing every current doc. `docs/env-vars.md` is the first entry.
  - Then do one `git mv` commit renaming to kebab-case:
    - `Gesso_Stack.md` → `gesso-stack.md`
    - `Gesso_musings.md` → `gesso-musings.md`
    - `DECISION_PACKETS.md` → `decision-packets.md`
    - `PACKAGING_REGIME_I.md` → `packaging-regime-i.md`
    - `Native_Julia_Kernel_Autotuning_North_Star_README.md` → `native-julia-kernel-autotuning-north-star.md`
  - Move `Harpe_Stack_old.md` to `docs/archive/<date>_harpe-stack/` with a `SUMMARY.md`.
  - Rename receipts with "final" in the name the same way.
  - Update links in the same commit.
- **A10.** `git mv .github/ISSUE_TEMPLATE/work-item.md .github/ISSUE_TEMPLATE/work-item.yml`.
- **A13.**
  - Split `ext/cuda_ops.jl` by op family when it is next edited.
  - Rewrite `docs/ARCHITECTURE.md` as current state only.
  - List merged and stale branches and worktrees (about 17 `worktree-agent-*`) for the owner to approve before pruning.
  - Move one-off `scripts/explore_*` experiments to `research/`.
- **I3 follow-up.** Add `_hot_stats` (hits, misses, fallback episodes) to the per-call receipt, so a run that fell back is visible in its receipt as well as in the log.
- **I6 follow-up.** A client that stops reading a streamed reply blocks the write, and the single-flight generation lock is held while it does. Add a per-write deadline (a timer that closes the socket if a chunk write stalls), with a test that uses a reader that never reads.
- **A11 follow-up.** Add a defaults column, filled by hand or by a lint that requires each read to go through one helper. Add a receipt field that lists the non-default `GESSO_*` values.

### Other notes

- `hygiene gc register` refused to register the remediation worktree because it contains `.git`. The worktree is a normal git worktree, owned by this pass. Remove it with `git worktree remove` after the branch is merged or dropped.
- `README.md` still says hot experts are opt-in, but `ext/cuda_hot_experts.jl` says they are on by default. This was not changed here; the owner should confirm which is right.

## 4. Evidence

### Remediation pass (2026-10-17), branch tip `27c7dbe`

All runs had GPUs hidden (`CUDA_VISIBLE_DEVICES=""`, Vulkan ICD pointed at a missing file) and ran under `timeout`. A display was present, so `scripts/test.jl` did not need to re-run under `xvfb-run`. That re-run path and the CI `xvfb-run` steps are untested until CI runs.

| Check | Before (`37d87da`) | After (`27c7dbe`) |
|---|---|---|
| `julia --project=test --check-bounds=yes test/runtests.jl` | 8735 pass / 57 broken / 8792 total, 13m11s | **8767 pass / 57 broken / 8824 total**, 12m00s, exit 0. No failures, no errors. The +32 are the new tests. |
| `make test` (`Pkg.test`, `-t 2`) | Fails: GessoLavaExt precompile (method overwrite) | **8805 pass / 55 broken / 8860 total**, 8m58s, "Gesso tests passed", exit 0. GessoLavaExt precompiles. The only Vulkan message is the typed `vk_context()` probe warning for a missing driver. |
| Bench smoke (`julia --project=benchmark benchmark/runbenchmarks.jl`) | Fails the same way in CI | "11 benchmarks recorded", results TSV written, exit 0 |
| `make format-check` + `git diff --exit-code` | 118 files differ | "formatting OK", no diff |
| `julia scripts/env-vars.jl --check` | (new) | current |
| Machine-path guard (CI step from `72f0969`) | (new) | clean |

- The two `make test` counts differ from the direct run because `Pkg.test` runs with two threads, which enables the threaded testsets.
- Targeted runs on each fix commit before the reformat:
  - `test/test_serve.jl`: all testsets pass, including the new decode-cost, read-deadline and connection-cap testsets.
  - `test/test_cuda_moe.jl`: the hot-fallback testset passes, 7 checks.
  - `test/test_gguf_import.jl`: the traits testset passes, 24 checks. That run set `GESSO_GGML_DIR` to a local llama.cpp master build, so all 17 table entries were also checked against the real library.
- The bench run left an untracked `benchmark/results/2026-10-17.tsv` in the worktree. It was not committed, because it is a CPU-only smoke record rather than a real bench row. Remove it, or ignore it with the worktree.
- Before the format and `make test` checks, the series was scanned for machine paths with `git log -p 37d87da..HEAD`. No hits.

### Audit (2026-10-16)

- **CI:** GitHub Actions history for `master`. 13+ consecutive failed runs since 2026-09-29. Latest is run 38329137328 on `38ca199`: test step 8723 pass / 60 broken / 8783 total in 8m06s. `make test` and `bench-smoke` fail on the GessoLavaExt precompile (GLFW error 65550, no `DISPLAY`; `vkCreateInstance` found no drivers). Format fails on `src/errors.jl`, `test/test_errors.jl` and `test/test_numeric_failures.jl`.
- **Local CPU baseline:** run on a `git archive` export of `37d87da`, with GPUs hidden (no CUDA devices, no Vulkan ICD). Command: `julia --project=test --check-bounds=yes test/runtests.jl`, under `timeout`. Result: exit 0, 8735 pass / 57 broken / 8792 total in 13m11s, with no failures or errors. The GessoLavaExt precompile failed (see B1), but that does not stop the test step. The local commits pass the CPU suite; GPU paths were skipped.
- **Divergence:** `git rev-list --left-right --count origin/master...master` gives 1 / 11. The local-only commits are `844d9e8..37d87da` (2026-10-12/13).
- **Code reading:** the files and line numbers cited in each finding. The fake SPIR-V comment was decoded from its hex bytes. `@gfallback` has one use site in the repo, found by grep.
- **Performance:** no new GPU runs, as the brief required. The existing records are `benchmark/results/2026-09-29.tsv` through `2026-10-03.tsv` (G2 factor 1.45×) and the README tables (35B CUDA decode 20 ms/token with hot experts; SmolLM2 Lava 37 tok/s vs CUDA 31 tok/s, with receipts in `docs/archive/2026-10_regime-ii/`). The I2 estimate is analytic, not measured.
- **Hygiene:** `git ls-files` for what is tracked versus only on disk. Grep for machine paths over tracked files (120 hits, all in docs). File-name scan for case, spaces, brackets and banned words.

## 5. Inheritance assessment

A new maintainer can take Gesso on. The engine's own rules (the CPU oracle, logged fallbacks, typed errors, receipts) are written down and mostly followed, and DEBT.md is honest and recent (2026-10-08), though entry 3 is stale.

What makes inheriting it hard:
- There is no green gate to trust (B1).
- The newest work exists only in one local checkout (I4).
- About 74 env switches, most undocumented (A11).
- An architecture doc that reads as history rather than current state (A13).
- About 17 worktrees and 50 branches with no ownership index.
- A hidden string a newcomer would find early (I1; withdrawn 2026-10-17, intentional per the owner).

Fixing B1 and I4 turns this into a repo that can be handed over. B1 is fixed on the remediation branch.

## 6. Release decision

**At audit time (2026-10-16): Blocked.** No commit of `master`, local or `origin`, had passed the project's own gate (`make test` plus format), and the tip (`37d87da`) had never run in CI.

**After the fix pass (2026-10-17): still Blocked, but only on steps outside this pass.**
- On branch `claude/pea-fix-20261017` (`27c7dbe`), every gate check passes locally (section 4).
- I1 is withdrawn, and I2, I3, I5 and I6 are fixed.

To unblock:
1. The owner decides whether to keep the reformat commit (section 3).
2. Reconcile I4 (rebase onto `origin/master`, resolve the README Training conflict), then push to a branch.
3. Get one green CI run with all four jobs (test, `make test`, format, bench-smoke). The CI-only parts (the `xvfb-run` steps, xvfb install, and the format job on a fresh runner) have not run anywhere yet.
4. Merge, and mark the jobs as required.

With that green run, the decision becomes **Pass with accepted risks**. The accepted risks are:
- The KernelAbstractions cap, until Lava drops the duplicate method.
- The I3 numeric-drift claim, until a GPU run confirms or clears it.
- The I6 write-side stall.
- The 11 local GPU commits, which have only CPU coverage.

## 7. Deferred work

- GPU verification of the 11 local commits (CUDA and Lava parity, the 35B gate). Out of scope here, since GPU runs were not allowed.
- Measuring I2's real cost with a loaded model (detokenize time at 1k, 4k and 32k tokens).
- A forced-OOM run of the hot-expert path to confirm the CPU/GPU numerical drift claimed in I3.
- Lava internals: the GLFW-at-precompile behaviour is Lava's and should go to the Lava repo as an issue. The `KA.adapt_structure` piracy is already in DEBT.md.
- An allocation and type-stability profile of the CPU decode hot loop (`@code_warntype`, `@allocated`). It needs a model fixture, so it is deferred.
- The branch and worktree cleanup plan (A13). It must be shown to the owner before anything is deleted.
