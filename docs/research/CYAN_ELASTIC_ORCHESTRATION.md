# CYAN ELASTIC ORCHESTRATION — GESSO SEAM

    Document class:     seam / pointer
    Status:             accepted 2026-10-02
    Index:              docs/research/README.md
    Canon freeze:       /mnt/d/Code Projects/Project Cyan/Cyan Harness/docs/CYAN_ELASTIC_ORCHESTRATION_v0.md
    This file is NOT:   a Gesso /goal, permission to fill src/Agents.jl,
                        a Phase 12/13 recipe, or a license to put
                        orchestration inside Inference

---

Cyan owns elastic multi-agent orchestration: unfinished **work** is
the durable object; workers are disposable leases over a
single-writer `WorkTree`. Palette is the operator view of that
organization. Gesso is the inference body.

Gesso 10D is COMPLETE (landing). Cyan Phase A closed recipe:
`Project Cyan/Cyan Harness/docs/goals/PHASE_A_WORKTREE_REDUCER.md`.
Do not implement this freeze in Gesso.

## What Gesso must preserve (inert until a Phase 12 recipe)

When Cyan later talks to Gesso, events/leases should already carry:

```
mission_id  node_id  parent_node_id  lease_id  branch_id
ancestor_chain  context_version  shared_prefix_identity
model_session_family
```

Gesso may then share prefix execution, `fork` Session trees, KV
pages, residency. Cyan exposes structure. Gesso decides. Cyan
correctness does not depend on that optimization.

```
Correctness:  WorkTree → generic provider → orchestration
Optimized:    WorkTree → Gesso Session/fork → same semantics
```

## What Gesso must not do

- Fill `src/Agents.jl` from this freeze (still Phase 12, empty).
- Put a conductor, WorkTree, or worker lease in `src/Inference/`.
- Block 10D or fusion on Cyan MAO.
- Couple fail-closed Session law to Cyan scheduling policy.

Phase 12–13 (shared-model runtime) is the Gesso recipe that consumes
the seam. It does not exist until the encoding owner writes it,
after the foundation is boring and a Cyan v0 reducer exists.
