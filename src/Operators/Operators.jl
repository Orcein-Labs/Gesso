# Operators — semantic operator dispatch surface (§XII, §XIX; Phase 1/2).
#
# Owns: the operator vocabulary and its dispatch structure — what operations
# exist (rmsnorm, rope, attention families, FFN families, ...) and how
# semantic context selects implementations. Multiple dispatch IS the
# execution mechanism (§XII).
#
# Does NOT own: kernel implementations (Lowering/backends), autotuning.
#
# The dispatch question (§XII): "What implementation is appropriate for this
# interaction among these objects?"
module Operators

# Phase 1/2 fill this module. Contract only — no speculative implementation.

end
