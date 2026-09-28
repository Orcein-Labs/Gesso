# Inference — the native inference engine (§XXIX, §XXX; Phase 5).
#
# Owns: model loader integration, prefill engine, decode engine, KV manager,
# sampling, streaming, sessions. Prefill and decode are EXPLICIT separate
# execution modes from the beginning (§XXX) — never one generic generate().
#
# KV note: the KV manager lands with span-provenance hooks per the KV memory
# program §12 — the CoW-fork and identity-sharing wins (§9.5 there) are built
# on it.
#
# Does NOT own: scheduling across requests/agents (Runtime), kernels
# (Lowering), representation policy (Representation).
module Inference

# Phase 5 fills this module. Contract only — no speculative implementation.

end
