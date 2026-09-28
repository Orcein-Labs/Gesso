# Profiling — performance observability (§XLIX, §L; Phase 6).
#
# Owns: telemetry collection for the §XLIX metric list (kernel latency,
# launch overhead, bandwidth, allocation, synchronization, compilation time,
# VRAM, TTFT, decode tok/s, batching efficiency, KV footprint, scheduler
# queue time, ...) and the §L failure taxonomy mapping — every performance
# gap classifiable, "no idea why it is slow" unacceptable.
#
# Event vocabulary already exists (logging.jl); this module owns collection.
module Profiling

# Phase 6 fills this module. Contract only — no speculative implementation.

end
