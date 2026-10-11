"""
    Gesso

Julia-native semantic ML and agent execution runtime.

Gesso loads existing open-weight models, preserves what they mean, and uses that
information to determine how they should physically exist and execute on the
hardware and workload actually present (docs/Gesso_Stack.md §CV).

The package has zero third-party dependencies by law (§VII: "Gesso earns every
hard dependency"). Backend lowering (CUDA.jl, Lava) arrives as package
extensions in later phases, never as core dependencies. Training is out of
scope permanently (§LVIII: TRAINING BOUNDARY).

Module map: docs/ARCHITECTURE.md. Agent rules: AGENTS.md.
"""
module Gesso

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
include("Operators/cpu.jl")                 # Phase 2: CPU reference methods (§LXXV)
include("Operators/cpu_linear.jl")          # Qwen3.5 Gated DeltaNet + sigmoid gate
include("Inference/Inference.jl")           # Phase 5
include("Runtime/Runtime.jl")               # Phase 5+
include("Profiling/Profiling.jl")           # Phase 6
include("Representation/Representation.jl") # Phase 10
include("Autotune/Autotune.jl")             # Phase 9
include("Speculation/Speculation.jl")       # Phase 20: host-side draft seam (SPECULATION_SEAM.md)
include("Planning/Planning.jl")             # Phase 7
include("Agents/Agents.jl")                 # Phase 12
include("CAPI/CAPI.jl")                     # Phase 16
include("serve/Serve.jl")                   # OpenAI-compatible chat server (after Inference: the engine binding and ordered JSON)

# --- public semantic vocabulary (§CIX; Phase 1 item A) ----------------------
# Re-exported so `using Gesso` sees the core vocabulary. Each module owns its
# names; the root only forwards them.
using .Semantics: PrefillWorkload, DecodeWorkload
export PrefillWorkload, DecodeWorkload

# --- chat server (src/serve) ------------------------------------------------
using .Serve: ChatModel, serve, gguf_chat_model
export ChatModel, serve, gguf_chat_model

using .ModelIR: Embedding, RMSNorm, RoPE, Attention, SwiGLU, Block, Model
export Embedding, RMSNorm, RoPE, Attention, SwiGLU, Block, Model

# Phase 2 (§LXXV): the reference interpreter is the first public engine slice
# Phase 3 (§LXXVI item B): the Llama import path rides the same surface
# Phase 5 (§LXXVIII): the Session engine — prefill!/decode!/generate over the
# paged KV manager; reference_* remain the oracle
using .Inference:
    reference_prefill,
    reference_generate,
    GenerateState,
    save_state,
    load_state,
    kv_bytes_per_token,
    memory_ceiling_tokens,
    CostModel,
    prefill_seconds,
    decode_seconds,
    ContextBudget,
    latency_ceiling_tokens,
    CompactionPolicy,
    CompactionTracker,
    reset!,
    decide_compaction,
    load_llama_config,
    config_to_model,
    load_safetensors,
    materialize_llama,
    load_llama,
    load_gguf,
    gguf_architecture_spec,
    gguf_header,
    GgufQuantSlice,
    GpuQuantSlice,
    HostOffload,
    gguf_dequant_expert,
    GPT2BPE,
    load_gpt2_tokenizer,
    encode,
    Session,
    prefill!,
    decode!,
    fork,
    generate,
    default_receipt_sink,
    RoPEPolicy,
    ArchitectureSpec,
    ArchitectureCapabilities,
    capabilities,
    required_semantics,
    SemanticParamId,
    FamilyParamMap,
    ParamRef,
    ArchitectureAdapter,
    LlamaAdapter,
    Qwen2Adapter,
    Qwen35Adapter,
    Qwen35MoeAdapter,
    GemmaAdapter,
    MistralAdapter,
    Phi3Adapter,
    PhiAdapter,
    FusedQKVLayout,
    adapter_for,
    known_families,
    architecture_spec,
    LinearMixerSpec,
    MoeSpec,
    layer_mixer,
    materialize_architecture,
    load_architecture,
    spec_to_model,
    parse_config,
    param_map,
    gguf_param_map,
    fused_qkv_rows,
    fused_qg_rows,
    family_symbol,
    is_scaled,
    canonical_name,
    slot_for,
    role_slots,
    rope_inv_freq,
    tensors_rope_inv_freq,
    MetaspaceBPE,
    load_metaspace_tokenizer,
    decode,
    import_report,
    compatibility_matrix,
    compatibility_table,
    tokenizer_metadata,
    vocab_size
export reference_prefill,
    reference_generate,
    GenerateState,
    save_state,
    load_state,
    kv_bytes_per_token,
    memory_ceiling_tokens,
    CostModel,
    prefill_seconds,
    decode_seconds,
    ContextBudget,
    latency_ceiling_tokens,
    CompactionPolicy,
    CompactionTracker,
    reset!,
    decide_compaction,
    load_llama_config,
    config_to_model,
    load_safetensors,
    materialize_llama,
    load_llama,
    load_gguf,
    gguf_architecture_spec,
    gguf_header,
    GgufQuantSlice,
    GpuQuantSlice,
    HostOffload,
    gguf_dequant_expert,
    GPT2BPE,
    load_gpt2_tokenizer,
    encode,
    Session,
    prefill!,
    decode!,
    fork,
    generate,
    default_receipt_sink,
    RoPEPolicy,
    ArchitectureSpec,
    ArchitectureCapabilities,
    capabilities,
    required_semantics,
    SemanticParamId,
    FamilyParamMap,
    ParamRef,
    ArchitectureAdapter,
    LlamaAdapter,
    Qwen2Adapter,
    Qwen35Adapter,
    Qwen35MoeAdapter,
    GemmaAdapter,
    MistralAdapter,
    Phi3Adapter,
    PhiAdapter,
    FusedQKVLayout,
    adapter_for,
    known_families,
    architecture_spec,
    LinearMixerSpec,
    MoeSpec,
    layer_mixer,
    materialize_architecture,
    load_architecture,
    spec_to_model

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
    frozen,
    RoPEPolicy,
    ArchitectureSpec,
    ArchitectureCapabilities,
    capabilities,
    required_semantics,
    SemanticParamId,
    FamilyParamMap,
    ParamRef,
    ArchitectureAdapter,
    LlamaAdapter,
    Qwen2Adapter,
    Qwen35Adapter,
    Qwen35MoeAdapter,
    GemmaAdapter,
    MistralAdapter,
    Phi3Adapter,
    PhiAdapter,
    FusedQKVLayout,
    fused_qkv_rows,
    fused_qg_rows,
    adapter_for,
    known_families,
    architecture_spec,
    LinearMixerSpec,
    MoeSpec,
    layer_mixer,
    materialize_architecture,
    load_architecture,
    spec_to_model,
    parse_config,
    param_map,
    gguf_param_map,
    family_symbol,
    is_scaled,
    canonical_name,
    slot_for,
    role_slots,
    rope_inv_freq,
    tensors_rope_inv_freq,
    MetaspaceBPE,
    load_metaspace_tokenizer,
    decode,
    import_report,
    compatibility_matrix,
    compatibility_table,
    tokenizer_metadata,
    vocab_size

# Phase 20 (§XCIII; docs/design/SPECULATION_SEAM.md): the draft-source seam and
# its host-only sources. No verifier is wired here; decoding is unchanged.
using .Speculation:
    DraftSource,
    DraftContext,
    Draft,
    propose,
    observe!,
    SuffixSource,
    TemplateRule,
    TemplateSource,
    SymbolicSource,
    IntegerArithmetic,
    IntegerSequence,
    LastLineProcedure,
    complete_text,
    feedback_counts
export DraftSource,
    DraftContext,
    Draft,
    propose,
    observe!,
    SuffixSource,
    TemplateRule,
    TemplateSource,
    SymbolicSource,
    IntegerArithmetic,
    IntegerSequence,
    LastLineProcedure,
    complete_text,
    feedback_counts

end
