# Parameters — semantic parameter and tensor objects (§XI; Phase 1/3).
#
# Owns: the semantic-class vocabulary for parameters/tensors —
# ProjectionWeight, KVCache, EmbeddingTable, ExpertWeight, FrozenParameter,
# QuantizedParameter, Activation, TemporaryWorkspace, RoutingState,
# DecodeState, AdapterDelta — distinguishing STORAGE from MEANING.
#
# Discipline (§XI, §XIII): semantic richness at parameter/tensor level; do NOT
# make every scalar symbolic; do NOT encode volatile runtime facts as types.
# Gradient/OptimizerState are training-side and forbidden (§LVIII).
module Parameters

# Phase 1/3 fill this module. Contract only — no speculative implementation.

end
