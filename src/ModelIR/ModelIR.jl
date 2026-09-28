# ModelIR — semantic model intermediate representation (§VII; Phase 1).
#
# Owns: the structural form of a model as Harpe sees it — semantic graph of
# composed architecture primitives, NOT a tensor graph. Sufficient for later
# planners to reason about legal realizations.
#
# Does NOT own: operator implementations, execution schedules, weights.
#
# Canon: "Supporting a new architecture should mean describing how it composes
# existing semantics" (§VIII) — the IR is what makes composition possible.
module ModelIR

# Phase 1 fills this module. Contract only — no speculative implementation.

end
