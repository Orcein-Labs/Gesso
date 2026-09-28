# Planning — execution-plan synthesis and memory planning (§XVII, §XIX;
# Phase 7).
#
# Owns: choosing among legal execution plans (fusion, staging, overlap,
# precision, backend choice) and whole-model memory planning (residency,
# tiering, budgets). Policies as first-class inputs (§LXI: MinLatency,
# MaxThroughput, MinMemory, Deterministic, AccuracyBound).
#
# Does NOT own: the realizations themselves (Representation), execution
# (Runtime), the search machinery (Autotune).
module Planning

# Phase 7 fills this module. Contract only — no speculative implementation.

end
