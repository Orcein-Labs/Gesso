# Gesso research parking lot

Current release campaigns:
[`Qwen3.5 and Gemma4 release plan`](../goals/RELEASE_PARITY_2026-10.md)
(2026-10-14). Follow its evidence gates before opening exotic research.

    Document class:     index (this folder)
    Status:             programs parked until a phase owns them
    Canon:              docs/Gesso_Stack.md
    Dangerous notebook: docs/Gesso_musings.md  (do not promote wholesale)

This folder is the place we look when we remember we had an exotic idea
and want to know **when** it is allowed to become code.

Gesso is not trying to be exotic yet.

    FIRST:  a boringly reliable functional ML stack
            (Phases 0–6: meaning, oracle, import, backends, engine, measure)
    THEN:   one semantic win (Phase 7)
    THEN:   representation planner (Phase 10), Magenta topology, the rest
    NEVER:  contaminate Phase 5 with a quantizer, a cage kernel, a
            Huffman codec, or a "lossless 2-bit" story

The engine is `Session` / `generate` over paged KV.
`reference_*` remain the oracle. Phases 0–9 are complete:
5a engine, 6 observability, 7 the declared-share win
(CoW + identity prefix share, `fork`, `unique_kv_bytes`),
8 Lava/Vulkan portable seam (`PHASE8_LAVA.md`),
9 autotune — the §XXVI loop for CUDA `matmul!`, winner cached, engine
consults it (`PHASE9_AUTOTUNE.md`; selection, not speed).
Living Phase 10 rungs 1–2 landed (`PHASE10_SPEED_FLOOR.md`, closed
by `PHASE10B_SPEED_FLOOR_CLOSE.md`). G1 is demo-box green on a local
snapshot; G2 factor is 0.692× (`2026-10-02.tsv`). 10C landed
(`PHASE10C_NAMED_MODEL_HYGIENE.md`): HF 0-based import, golden frozen
(oracle `gesso-cpu`, 49152 values), SmolLM2 fork byte share measured
(CPU N = 1_474_560; fork == one session, isolated == 2N). 10D landed
2026-10-02 (`PHASE10D_FOUNDATION_HARDENING.md`): export inventory +
parked-empty fence, fail-closed hunt, `@inferred` byte counters.
10E landed 2026-10-02 (`PHASE10E_FUSED_DECODE.md`): Session-owned
decode workspace, in-place `gather_kv!`, in-place GQA repeat; warmed
`decode!` host allocation is O(1) in seqlen (SmolLM2 CPU 15,864,048 →
100,480 B); G2 factor republished at 0.737×.
10F landed 2026-10-03 (`PHASE10F_CUDA_DECODE_ALLOC.md`): the CUDA
decode path's per-head broadcast wrappers are gone — device kernels
behind `CuArray` specializations of the SAME core copy helpers, so
`Inference.jl` now charges **0 B** to a warmed SmolLM2 CUDA `decode!`.
Warmed CUDA `decode!` 5,094,736 → 1,269,616 B (4.0×); toy2/micro CUDA
are back under item D's original 256 KiB gate (97,312 / 100,736 B);
G2 factor republished at 1.452× (Gesso warmed 487 → 240 ms). The
SmolLM2 CUDA 1 MiB ceiling was still missed by 221 KB — escalated as a
packet inside the goal, gate left at the DECLARED 1 MiB as a tracked
`@test_broken` rather than raised again.
10G landed 2026-10-03 (`PHASE10G_AUTOTUNE_RECEIPT.md`): the packet is
closed and the gate is GREEN. A cache hit is not a decision (§LXXII
records a change; the miss receipt already named the winner), so
`Autotune.select` returns the cached `TuneResult` and emits no receipt.
Warmed SmolLM2 CUDA `decode!` 1,269,616 → 942,128 B (1.35×, 106,448 B
under 1 MiB); `Autotune.jl` per-token cost 320,509 → 37,136 B. Miss
receipts are untouched — the receipt schema did not bump. Canon
§LXXXIII parked.
10H landed 2026-10-03 (`PHASE10H_TYPE_STABILITY.md`): 10D packet P-1
closed. §CIX amended by exactly one clause — the ARRAY TYPE that
physically holds a tensor is a trait-equivalent and may be a single
type parameter `S` on the §XI family structs; shape, seqlen, batch
and allocation identity stay fields, and no second axis was added.
`@inferred reference_prefill` and `@inferred decode!` are GREEN on
toy2 and llama_micro, so the `@test_broken` trio is gone and the
snapshot-set Broken count is 0. Type-stable dispatch also cut warmed
SmolLM2 CUDA `decode!` 942,128 B → 814,544 B.
Tackle each problem in this folder individually when its phase owns it.

---

## Programs (open these, do not rebuild them in `src/` first)

| File | Object | Sentence | Owned by |
|---|---|---|---|
| [KV_MEMORY_PROGRAM.md](KV_MEMORY_PROGRAM.md) | working state (KV / attention memory) | working-state representation is a lowering | Phase 5 paged KV → 9 / 10 / 11 |
| [KV_MEMORY_PROGRAM_part2.md](KV_MEMORY_PROGRAM_part2.md) | Magenta execution plane (topology, residency, schedule) | same object, access/residency half | after a real KV manager exists |
| [REPRESENTATION_PROGRAM.md](REPRESENTATION_PROGRAM.md) | static learned operators (weights) | do not quantize the accidental representation | Phase 7 candidate / Phase 10 host |
| [CYAN_TRIAL_GESSO_FALLOUT.md](CYAN_TRIAL_GESSO_FALLOUT.md) | promotion filter from the first Cyan/Palette engineer trial | generality × composability × ecosystem leverage, or it stays above Gesso | Phase 6+ spikes; not Phase 5 |
| [CYAN_ELASTIC_ORCHESTRATION.md](CYAN_ELASTIC_ORCHESTRATION.md) | Gesso seam for Cyan MAO (WorkTree / leases / branch ids) | Cyan owns orchestration; Gesso consumes ids later | pointer only; freeze lives in Cyan Harness docs; not a /goal |
| [ROADMAP_NOW.md](ROADMAP_NOW.md) | living reading of Phases 5–22 after 0–4 + the Cyan trial | 5a engine now; speed floor next; 5b and Phase 10 after G1∧G2∧G3 | map only — not a /goal |
| [BREADTH-0_FINDINGS.md](BREADTH-0_FINDINGS.md) | BREADTH-0 landed: model breadth is an adapter exercise, not an engine rewrite | ArchitectureSpec + family adapters + canonical parameter identity + tokenizer protocol + capability lattice; a second family (Phi-3, fused QKV) runs on the CPU oracle | findings and receipt for the BREADTH-0 batch; open limitations listed there |
| [CAPABILITY-0a_FINDINGS.md](CAPABILITY-0a_FINDINGS.md) | CAPABILITY-0a landed: the capability declaration is tied to execution | five-arm biconditional over `supports`; both directions mutation-verified; a from-scratch `:attention` fixture | findings and receipt; `supports` is still a hand-written Set — deriving it from dispatch is CAPABILITY-0b |
| [SPEED_FLOOR.md](SPEED_FLOOR.md) | board constraint: somewhat-close decode before exotic compilers | named model + eager-PyTorch factor + fork-preserving fused step; compiler attribution before a Julia fork | gate on §LXXXIII; 10C recipe landed (`PHASE10C_NAMED_MODEL_HYGIENE.md`): golden frozen, named-model fork bytes pinned |
| [EXOTIC_CAPABILITY_MASTER_LEDGER.md](EXOTIC_CAPABILITY_MASTER_LEDGER.md) | capability mine: meta-primitives, Gesso/Palette/Cyan primitives, six meta-tools | strings to undiscovered capabilities; evidence labels; not a roadmap override | parked; deep-dives after speed floor |
| [spark2-5-port/2026-10-22_spark2-5-port.md](spark2-5-port/2026-10-22_spark2-5-port.md) | Spark 2.5 (Spark-X2.6-4B) on the CPU oracle | sliding window, per-layer RoPE, per-head gate and exact-erf GEGLU; greedy 32/32 vs llama.cpp on three prompts incl. one past the window | receipt; paged Session support and an SWA ring-buffer KV are open |
| [spark2-5-port/2026-10-22_spark-quant-parity.md](spark2-5-port/2026-10-22_spark-quant-parity.md) | Spark quant parity vs BF16 | KLD/top-1 on wikitext and GPQA-science text; GSQ-RCO schedule 0.97 KLD, new A8F6 (attn Q8_0, FFN Q6_K, imatrix) 0.011 KLD / 95.5% top-1 at 3.69 GB | receipt; long-budget GPQA and Gesso load of A8F6 are open |
| [ornith-port/2026-10-23_ornith-quant-parity.md](ornith-port/2026-10-23_ornith-quant-parity.md) | Ornith 1.5 35B-A3B quant parity vs BF16 | vendor GSQ-RCO-3 fails at 91.8% of BF16 MMLU-Pro; harness reproduced and backend noise traced to 0.08% answer-letter mass; our backend reads −1.1 ± 0.7 pp vs vendor; d4 (gate/up Q2_K, down Q4_K, 3.60 BPW) scores 45.75 and clears the gate; no candidate reaches 95% top-1 | receipt; parked for Saluki 27B; i3xxs/i3s/e3 unscored |
| [saluki-cpu/2026-10-23_window-oracle.md](saluki-cpu/2026-10-23_window-oracle.md) | Saluki 27B verify-window oracle on CPU | ticketed multi-row chunks cut W(4) from 2.4–2.7 to 2.0–2.3; Q8_0 bean (80 ms/step) and 8-column tile give 1.25–1.58 at T=4 on json/code/math (normalized), prose and shell lose; generate prefill 4.27 tok/s | receipt; real bean drafter and adaptive T next |

Same philosophy, two expensive objects:

    WEIGHTS     static learned operators     REPRESENTATION_PROGRAM.md
    KV / attn   dynamic working state        Magenta (KV_MEMORY_PROGRAM*)

Engineer KV first (Phase 5 paged manager, then CoW / identity prefix
share — landed, Phase 7, `fork` + `Profiling.unique_kv_bytes`). Probe weight gauges on the CPU oracle in parallel **only**
when a SmolLM2 snapshot exists. Weaponize GPU kernels after
Measurement 1 is interesting. See representation program §7.

---

## Later, still parked in canon (no file here yet)

These live in `docs/Gesso_Stack.md`. They do not get a research
program until someone writes one **and** a phase owns it.

| Topic | Canon | When |
|---|---|---|
| Native engine | §LXXVIII | **5a landed 2026-09-30**; 5b serving later |
| Observability | §LXXIX | **6 landed 2026-09-30** (`PHASE6_OBSERVABILITY.md`) |
| One semantic win (pick one) | §LXXX | **7 landed 2026-09-30** (`PHASE7_PREFIX_SHARE.md`) |
| Lava / Vulkan | §LXXXI | **8 landed 2026-09-30** (`PHASE8_LAVA.md`) |
| Autotune | §LXXXII | **9 landed 2026-10-01** (`PHASE9_AUTOTUNE.md`) |
| Speed floor rungs 1–2 (device greedy + GEMM, G2 harness) | §LXXXII/§LXXXIII | **10/10B/10C/10D/10E/10F/10G/10H landed**; G2 factor 1.452× (was 0.737× at 10E, 0.692× at 10B). §LXXXIII representation planner stays PARKED |
| Representation planner | §LXXXIII | after G1∧G2∧G3 — seed is REPRESENTATION_PROGRAM.md |
| Memory planner | §LXXXIV | Phase 11 |
| Agent runtime / Palette / Cyan / C ABI | §LXXXV–LXXXIX | Phases 12–16 |
| Generalized speculation | §XCIII | **Phase 20** |
| Training stack | §XCIV | never Gesso; open call |
| Plastic / lifecycle compute | §XCV | Phase 22 |
| Distributed / sharding as a lowering | musings | V1.5+; no program yet |

Campaign findings: `grok-wip/2026-10-09_35b-cuda-decode-parity.md`. The
35B CUDA decode reached llama.cpp parity (36.64 tok/s), and with opt-in
VRAM hot experts the gate reads 39.9 tok/s. Single-channel DRAM is the
binding cost.

Phase 7 pick (locked): identity prefix share / CoW KV
(`docs/goals/PHASE7_PREFIX_SHARE.md`). Other candidates stay
parked (fusion needs more numbers; cages need Measurement 1).

Speculative decoding is Phase 20. It is not a Phase 5 client.

---

## Laws this folder does not get to waive

* Training is out of scope (§LVIII).
* Citation = claim of having read it (Magenta §4.3).
* No public / performance claim without a receipt from this repo (§XLII).
* Approximate lowerings are declared contracts, never silent flags
  (Magenta §6, `APPROXIMATION_BUDGET_EXCEEDED`).
* Agent confidence is not evidence. The harness decides (§LXXII).
* Magenta / cages / codecs stay out of `src/` until a work item names
  the files and the phase.

---

## How to add a direction

1. Write it here as a program (Magenta header style: status, canon
   sections, phase gates, verify-before-citing, what the file is not).
2. Link it from this README and from `docs/ARCHITECTURE.md`.
3. Add it to `scripts/freeze.jl` (curated list + freeze briefing).
4. Do not add types, kernels, or planners to `src/` in the same breath.

If it is still a riff, it belongs in `docs/Gesso_musings.md`.
