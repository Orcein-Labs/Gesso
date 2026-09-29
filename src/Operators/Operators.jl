# Operators — dispatch as execution (§XII, §CIX; Phase 1).
#
# §CIX encoding: an Operator IS a function. The vocabulary is owned by
# `src/backends.jl` (rmsnorm! … dequantize!) — this module adds METHODS to
# those same functions; it does not create a second vocabulary.
#
# What Phase 1 adds: the dispatch surface. Methods are keyed on
#
#     op!(backend, dst::SemanticTensor, src::SemanticTensor, workload)
#
# with `workload` a `PrefillWorkload` / `DecodeWorkload` singleton (§XXX,
# §CIX Packet 1: the workload cut participates in dispatch).
#
# What Phase 1 does NOT add: the math. Every Phase-1 method declines
# explicitly with `LoweringNotImplemented` (§LXX) — the ONLY legal decline —
# identifying op and backend exactly as the stubs do. Phase 2 fills CPU math
# by adding more specific methods (backend ::CPUBackend); backend extensions
# (CUDA Phase 4, Lava Phase 8) add their own. Nothing here falls back,
# substitutes, or returns a sibling's result.
#
# Vocabulary is exactly the stub list — adding an op is a work item, and the
# inventory test (test/test_backends.jl) enforces it.

module Operators

using ..Semantics: PrefillWorkload, DecodeWorkload
using ..Parameters: SemanticTensor
# The op names MUST be EXPLICITLY IMPORTED before `function op!` can extend
# them (Julia 1.12 enforces this; a plain `using` would define shadow
# functions inside Operators — a second vocabulary, which §CIX forbids and
# the test suite pins).
import ..Harpe:
    rmsnorm!, rope!, softmax!, swiglu!, matmul!, embedding_lookup!, quantize!, dequantize!
using ..Harpe: AbstractHarpeBackend, lowering_not_implemented

# dst/src are SemanticTensor values (§XI families); the workload singleton
# is the last positional dispatch argument (§CIX). Bodies decline explicitly.
for op in (
        :rmsnorm!,
        :rope!,
        :softmax!,
        :swiglu!,
        :matmul!,
        :embedding_lookup!,
        :quantize!,
        :dequantize!,
    ),
    wl in (:PrefillWorkload, :DecodeWorkload)

    @eval function $op(
        backend::AbstractHarpeBackend,
        dst::SemanticTensor,
        src::SemanticTensor,
        ::$wl,
    )
        lowering_not_implemented($(QuoteNode(op)), backend)
    end
end

end # module Operators
