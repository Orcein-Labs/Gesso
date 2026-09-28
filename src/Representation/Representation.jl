# Representation — representation planning / materialization (§XIV, §XV;
# KV memory program §5; Phase 10).
#
# Owns: the logical→physical vocabulary. The realization lattice (share/
# select/subspace/transform/VQ/residual/bit-allocation/resolution/tier/
# reconstruct) lives here as named lowering choices. Quantization-as-lowering
# (§XV). Approximation contracts: ExactLowering / BoundedApproximation
# (KV program §6).
#
# The heavy idea (§XIV): logical model ≠ physical model. The same logical
# object may materialize differently per device/workload/policy — and that is
# expected behavior, not portability failure.
#
# Does NOT own: the search over realizations (Autotune), plan selection
# (Planning).
module Representation

# Phase 10 fills this module. Contract only — no speculative implementation.

end
