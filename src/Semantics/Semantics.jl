# Semantics — model/parameter/operator semantic vocabulary (§I, §XI; Phase 1).
#
# Owns: the meaning-preservation vocabulary — what a model IS beyond its
# tensors: architecture semantics, semantic roles, traits vs types vs runtime
# metadata (§XIII discipline).
#
# Does NOT own: physical representation (Representation), execution planning
# (Planning), concrete operators (Operators).
#
# Encoding is law under §CIX: this module owns the vocabulary, not a third
# object model. Phase 1 implements §CIX; it does not choose another encoding.

module Semantics

# --- Workload dispatch types (§CIX Packet 1; §XII, §XXX) --------------------
#
# Prefill and decode are different workloads (§XXX: "Do not pretend
# otherwise") and participate in operator dispatch. They are TYPES, not a
# mega-enum (§CIX): one singleton per dispatch cut. The finer §XVI categories
# (prompt_prefill, batch1_decode, ...) are documented TAGS for receipts and
# plans until a lowering actually dispatches on them; promoting a tag to a
# type is a work item, not a drive-by.

"""
    PrefillWorkload

The prompt-processing workload (§XXX). Singleton type; dispatch value in
the operator surface. Prompt prefill, batched prefill, and long-context
prefill are §XVI TAGS within this cut, not separate types.
"""
struct PrefillWorkload end

"""
    DecodeWorkload

The token-generation workload (§XXX). Singleton type; dispatch value in
the operator surface. Batch-1 decode and batched decode are §XVI TAGS
within this cut, not separate types.
"""
struct DecodeWorkload end

export PrefillWorkload, DecodeWorkload

end # module Semantics
