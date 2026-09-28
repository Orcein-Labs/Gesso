# Lowering — backend dispatch for semantic operations (§XXII, §XXIII; Phase 4).
#
# Owns: the machinery that routes a semantic operation to the backend that
# will execute it, honoring capability queries (§XX) and execution tiers
# (§XXI). Mixed-backend execution is ordinary here, never exceptional.
#
# Does NOT own: backend implementations themselves (package extensions),
# plan selection (Planning).
module Lowering

# Phase 4 fills this module. Contract only — no speculative implementation.

end
