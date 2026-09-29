# Runtime — serving runtime and agent-runtime mechanisms (§XXXII, §XXXIII;
# Phase 5+/12).
#
# Owns: the request/inference scheduler, continuous batching, memory-pressure
# handling, sessions/streaming integration; later the agent-runtime
# MECHANISM primitives (tasks, dependencies, channels, budgets, priorities,
# cancellation — §XXXIII) and cross-agent inference scheduling (Phase 13).
#
# Boundary (§XLIII): Gesso owns mechanism. Expression is Palette's, policy is
# NIRA's. No LangChain.jl.
module Runtime

# Phase 5+ fills this module. Contract only — no speculative implementation.

end
