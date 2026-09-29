"""
    Harpe

Julia-native semantic ML and agent execution runtime.

Harpe loads existing open-weight models, preserves what they mean, and uses that
information to determine how they should physically exist and execute on the
hardware and workload actually present (docs/Harpe_Stack.md §CV).

The package has zero third-party dependencies by law (§VII: "Harpe earns every
hard dependency"). Backend lowering (CUDA.jl, Lava) arrives as package
extensions in later phases, never as core dependencies. Training is out of
scope permanently (§LVIII: TRAINING BOUNDARY).

Module map: docs/ARCHITECTURE.md. Agent rules: AGENTS.md.
"""
module Harpe

# --- foundation (always loaded) ---------------------------------------------
include("logging.jl")      # §XLII/§LXX  structured events, fallback recording
include("versions.jl")     # §LXIX       schema versions for persisted artifacts
include("backends.jl")     # §XX/XXI     backend contract + lowering stubs
include("errors.jl")       # §LXX        typed failure vocabulary + taxonomy
include("receipts.jl")     # §XLII       receipt records + sink interface

# --- domain modules (contract skeletons; bodies land with their phases) -----
# See docs/ARCHITECTURE.md for the module → section → phase map.
include("Semantics/Semantics.jl")           # Phase 1
include("ModelIR/ModelIR.jl")               # Phase 1
include("Parameters/Parameters.jl")         # Phase 1/3
include("Operators/Operators.jl")           # Phase 1/2
include("Lowering/Lowering.jl")             # Phase 4
include("Inference/Inference.jl")           # Phase 5
include("Runtime/Runtime.jl")               # Phase 5+
include("Profiling/Profiling.jl")           # Phase 6
include("Representation/Representation.jl") # Phase 10
include("Autotune/Autotune.jl")             # Phase 9
include("Planning/Planning.jl")             # Phase 7
include("Agents/Agents.jl")                 # Phase 12
include("CAPI/CAPI.jl")                     # Phase 16

# --- public semantic vocabulary (§CIX; Phase 1 item A) ----------------------
# Re-exported so `using Harpe` sees the core vocabulary. Each module owns its
# names; the root only forwards them.
using .Semantics: PrefillWorkload, DecodeWorkload
export PrefillWorkload, DecodeWorkload

using .Parameters:
    SemanticTensor,
    ProjectionWeight,
    KVCache,
    EmbeddingTable,
    ExpertWeight,
    FrozenParameter,
    QuantizedParameter,
    Activation,
    TemporaryWorkspace,
    RoutingState,
    DecodeState,
    AdapterDelta,
    frozen
export SemanticTensor,
    ProjectionWeight,
    KVCache,
    EmbeddingTable,
    ExpertWeight,
    FrozenParameter,
    QuantizedParameter,
    Activation,
    TemporaryWorkspace,
    RoutingState,
    DecodeState,
    AdapterDelta,
    frozen

end
