# CAPI — libharpe stable C interface (§XLVI; Phase 16).
#
# Owns: the boring, stable C ABI — opaque handles (harpe_runtime_t,
# harpe_model_t, harpe_context_t, harpe_request_t) and lifecycle operations.
# An ADOPTION SURFACE, not the internal architecture; all semantic/compiler
# complexity stays behind the handles.
#
# Compatibility direction (§XLVII): adapters adapt TO Harpe. Harpe does not
# adapt its ontology to compatibility.
module CAPI

# Phase 16 fills this module. Contract only — no speculative implementation.

end
