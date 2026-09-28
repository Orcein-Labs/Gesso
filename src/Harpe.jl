"""
    Harpe

Julia-native semantic ML and agent execution runtime.

Harpe loads existing open-weight models, preserves what they mean, and uses that
information to determine how they should physically exist and execute on the
hardware and workload actually present (docs/Harpe_Stack.md §CV).

Phase 0 — repository foundation. The package has zero hard dependencies by law
(§VII: "Harpe earns every hard dependency"). Backend lowering (CUDA.jl, Lava)
arrives as package extensions in later phases, never as core dependencies.
Training is out of scope permanently (§LVIII: TRAINING BOUNDARY).
"""
module Harpe

include("logging.jl")
include("backends.jl")

end
