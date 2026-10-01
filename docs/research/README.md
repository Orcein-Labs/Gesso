# Gesso research parking lot

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
`reference_*` remain the oracle. Phases 0–7 are complete:
5a engine, 6 observability, 7 the declared-share win
(CoW + identity prefix share, `fork`, `unique_kv_bytes`).
Next gates: 5b serving fabric (a later goal under Phase 5),
then Phases 8–9.
Tackle each problem in this folder individually when its phase owns it.

---

## Programs (open these, do not rebuild them in `src/` first)

| File | Object | Sentence | Owned by |
|---|---|---|---|
| [KV_MEMORY_PROGRAM.md](KV_MEMORY_PROGRAM.md) | working state (KV / attention memory) | working-state representation is a lowering | Phase 5 paged KV → 9 / 10 / 11 |
| [KV_MEMORY_PROGRAM_part2.md](KV_MEMORY_PROGRAM_part2.md) | Magenta execution plane (topology, residency, schedule) | same object, access/residency half | after a real KV manager exists |
| [REPRESENTATION_PROGRAM.md](REPRESENTATION_PROGRAM.md) | static learned operators (weights) | do not quantize the accidental representation | Phase 7 candidate / Phase 10 host |
| [CYAN_TRIAL_GESSO_FALLOUT.md](CYAN_TRIAL_GESSO_FALLOUT.md) | promotion filter from the first Cyan/Palette engineer trial | generality × composability × ecosystem leverage, or it stays above Gesso | Phase 6+ spikes; not Phase 5 |
| [ROADMAP_NOW.md](ROADMAP_NOW.md) | living reading of Phases 5–22 after 0–4 + the Cyan trial | 5a engine now; 5b serving later; Cyan consumes Gesso | map only — not a /goal |

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
| Autotune | §LXXXII | Phase 9 |
| Representation planner | §LXXXIII | Phase 10 — seed is REPRESENTATION_PROGRAM.md |
| Memory planner | §LXXXIV | Phase 11 |
| Agent runtime / Palette / Cyan / C ABI | §LXXXV–LXXXIX | Phases 12–16 |
| Generalized speculation | §XCIII | **Phase 20** |
| Training stack | §XCIV | never Gesso; open call |
| Plastic / lifecycle compute | §XCV | Phase 22 |
| Distributed / sharding as a lowering | musings | V1.5+; no program yet |

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
