# Autotune — bounded physical-realization search (§XXVI; Phase 9;
# KV memory program §7).
#
# Owns: the search/verify/measure/cache/receipt machinery — bounded budgets,
# deterministic candidate ordering, correctness gates (oracle ladder), cost
# measurement (M, L, C), conservative cache invalidation, full receipts.
#
# Framing (KV program §7): this is a PHYSICAL-REALIZATION SEARCH ENGINE whose
# first client is kernel schedules (North Star project) and whose later
# clients include quantization programs, KV realization pipelines, and
# placement. The build stays kernel-first; the generalization is earned at
# Phase 10. Speculative plans are explicitly NOT a client (KV program §7.1).
#
# Integration stance: survey/integrate existing Julia autotuning work before
# building redundant infrastructure (§XXVI) — the North Star companion spec
# is that survey's starting point.
module Autotune

# Phase 9 fills this module. Contract only — no speculative implementation.

end
