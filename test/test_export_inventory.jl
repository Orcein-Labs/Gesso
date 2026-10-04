# test_export_inventory.jl — Phase 10D item A: every public name has a home.
#
# Two fences, one file:
#
# 1. EXPORT INVENTORY. Every name in `setdiff(names(Gesso), [:Gesso])` must
#    have a row here: either the test file(s) that exercise it, or an
#    explicit bucket. A drive-by export fails CI (missing row), and a
#    renamed/removed export fails CI (stale row) — the inventory cannot rot
#    in either direction.
#
# 2. PARKED-EMPTY FENCE. The six contract-only modules
#    (Lowering/Representation/Planning/Runtime/Agents/CAPI) export exactly
#    their own name and nothing else (§LXXXIII parked; growing one is a
#    work item, not a drive-by).
#
# Bucket vocabulary (10D item A):
#   :live       — the listed test files exercise the name directly
#   :here       — exercised directly in THIS file (no other home today)
#   :empty      — parked §CIX vocabulary: contract/structure only, no body
#   :ext_only   — exercised only through a backend extension suite
#   :skip_named — the named skip on a device-less/box-less run is its home

using Test

const PARKED_MODULES = (:Lowering, :Representation, :Planning, :Runtime, :Agents, :CAPI)

const INVENTORY = Dict{Symbol, Tuple{Symbol, Vector{String}}}(
    # --- logging (§XLII) ----------------------------------------------------
    Symbol("@gfallback") => (:live, ["runtests.jl"]),
    :glog => (:live, ["runtests.jl", "test_receipts.jl"]),
    :min_level! => (:live, ["runtests.jl"]),
    :current_config => (:live, ["runtests.jl", "test_receipts.jl"]),
    :GessoLogConfig => (:here, String[]),

    # --- versions (§LXIX) ---------------------------------------------------
    :GESSO_SCHEMA_VERSION => (:live, ["test_foundation.jl"]),
    :RECEIPT_SCHEMA_VERSION => (:live, ["test_foundation.jl", "test_receipts.jl"]),
    :BENCH_RESULT_SCHEMA_VERSION => (:live, ["test_foundation.jl"]),
    :AUTOTUNE_CACHE_VERSION => (:live, ["test_autotune.jl"]),

    # --- backends (§XX/§XXI) ------------------------------------------------
    :AbstractGessoBackend => (
        :live,
        ["runtests.jl", "test_backends.jl", "test_cuda_seam.jl", "test_lava_seam.jl"],
    ),
    :CPUBackend => (:live, ["test_backends.jl"]),
    :backend_name => (
        :live,
        ["runtests.jl", "test_backends.jl", "test_cuda_seam.jl", "test_lava_seam.jl"],
    ),
    :execution_tier => (
        :live,
        ["runtests.jl", "test_backends.jl", "test_cuda_seam.jl", "test_lava_seam.jl"],
    ),
    :supports => (
        :live,
        [
            "runtests.jl",
            "test_backends.jl",
            "test_cuda_seam.jl",
            "test_lava_seam.jl",
            "test_session_cuda.jl",
        ],
    ),
    :LoweringNotImplemented =>
        (:live, ["runtests.jl", "test_backends.jl", "test_cpu_ops.jl"]),
    :lowering_not_implemented => (:here, String[]),

    # --- errors (§LXX) ------------------------------------------------------
    :ErrorCode => (:live, ["test_errors.jl"]),
    :GessoError => (:live, ["test_errors.jl"]),
    :GessoException => (:live, ["test_errors.jl"]),
    :gesso_error => (:live, ["test_errors.jl"]),
    :ERR_ALLOCATION => (:live, ["test_errors.jl"]),
    :ERR_APPROXIMATION_BUDGET_EXCEEDED => (:live, ["test_errors.jl"]),
    :ERR_BENCHMARK => (:live, ["test_errors.jl"]),
    :ERR_CACHE => (:live, ["test_errors.jl"]),
    :ERR_COMPILE => (:live, ["test_errors.jl"]),
    :ERR_CONSTRAINT_REJECTED => (:live, ["test_errors.jl"]),
    :ERR_INTERNAL => (:live, ["test_errors.jl"]),
    :ERR_INVALID_PLAN => (:live, ["test_errors.jl"]),
    :ERR_LAUNCH => (:live, ["test_errors.jl"]),
    :ERR_NUMERICAL_INSTABILITY => (:live, ["test_errors.jl"]),
    :ERR_RESOURCE_LIMIT => (:live, ["test_errors.jl"]),
    :ERR_RUNTIME => (:live, ["test_errors.jl"]),
    :ERR_TIMEOUT => (:live, ["test_errors.jl"]),
    :ERR_VERIFY_MISMATCH => (:live, ["test_errors.jl"]),

    # --- receipts (§XLII) ---------------------------------------------------
    :Receipt => (:live, ["test_receipts.jl", "test_foundation.jl"]),
    :ReceiptSink => (:here, String[]),
    :InMemorySink => (:live, ["test_receipts.jl"]),
    :emit! => (:live, ["test_receipts.jl"]),
    :new_receipt => (:live, ["test_receipts.jl"]),
    :next_receipt_id => (:live, ["test_receipts.jl", "test_empty_core.jl"]),
    :default_receipt_sink => (:live, ["test_session_receipts.jl"]),

    # --- §CIX semantics vocabulary (workload dispatch types) ----------------
    :PrefillWorkload => (:live, ["test_semantics.jl", "test_operators.jl"]),
    :DecodeWorkload => (:live, ["test_semantics.jl", "test_operators.jl"]),

    # --- §CIX ModelIR vocabulary --------------------------------------------
    :Embedding => (:live, ["test_modelir.jl"]),
    :RMSNorm => (:live, ["test_modelir.jl"]),
    :RoPE => (:live, ["test_modelir.jl"]),
    :Attention => (:live, ["test_modelir.jl"]),
    :SwiGLU => (:live, ["test_modelir.jl"]),
    :Block => (:live, ["test_modelir.jl"]),
    :Model => (:live, ["test_modelir.jl"]),

    # --- §XI parameter vocabulary (some are structure-only by law) ----------
    :SemanticTensor => (:live, ["test_parameters.jl"]),
    :ProjectionWeight => (:live, ["test_parameters.jl", "test_operators.jl"]),
    :KVCache => (:live, ["test_parameters.jl", "test_kv_manager.jl"]),
    :EmbeddingTable => (:live, ["test_parameters.jl", "test_cpu_ops.jl"]),
    :ExpertWeight => (:empty, String[]),          # §LVIII-adjacent vocabulary: contract only
    :FrozenParameter => (:live, ["test_parameters.jl", "test_cpu_ops.jl"]),
    :QuantizedParameter => (:empty, String[]),    # §XV vocabulary: contract only (Phase 10)
    :Activation => (:live, ["test_parameters.jl", "test_operators.jl"]),
    :TemporaryWorkspace => (:live, ["test_parameters.jl", "test_cpu_ops.jl"]),
    :RoutingState => (:empty, String[]),          # MoE vocabulary: contract only
    :DecodeState => (:empty, String[]),           # contract only
    :AdapterDelta => (:empty, String[]),          # contract only (serving is later)
    :frozen => (:live, ["test_parameters.jl"]),

    # --- Phase 2/3 oracle + import (§LXXV/§LXXVI) ---------------------------
    :reference_prefill => (:live, ["test_reference_prefill.jl"]),
    :reference_generate => (:live, ["test_reference_generate.jl"]),
    :load_llama_config => (:live, ["test_import_llama.jl"]),
    :config_to_model => (:live, ["test_import_llama.jl"]),
    :load_safetensors => (:live, ["test_import_llama.jl"]),
    :materialize_llama => (:live, ["test_import_llama.jl"]),
    :load_llama => (:live, ["test_import_llama.jl"]),
    :GPT2BPE => (:here, String[]),                # type exercised via its loader everywhere
    :load_gpt2_tokenizer => (:live, ["test_tokenizer_gpt2.jl"]),
    :encode => (:live, ["test_tokenizer_gpt2.jl"]),

    # --- BREADTH-0: the universal model doorway (§VIII; Passes A/B/D/E/F) ---
    Symbol("RoPEPolicy") => (:live, ["test_breadth0.jl"]),
    :ArchitectureSpec => (:live, ["test_breadth0.jl"]),
    :ArchitectureCapabilities => (:live, ["test_breadth0.jl"]),
    :capabilities => (:live, ["test_breadth0.jl"]),
    :required_semantics => (:live, ["test_breadth0.jl"]),
    :SemanticParamId => (:live, ["test_breadth0.jl"]),
    :FamilyParamMap => (:live, ["test_breadth0.jl"]),
    :ParamRef => (:live, ["test_breadth0.jl"]),
    :ArchitectureAdapter => (:live, ["test_breadth0.jl"]),
    :LlamaAdapter => (:live, ["test_breadth0.jl"]),
    :Qwen2Adapter => (:live, ["test_breadth0.jl"]),
    :GemmaAdapter => (:live, ["test_breadth0.jl"]),
    :MistralAdapter => (:live, ["test_breadth0.jl"]),
    :Phi3Adapter => (:live, ["test_breadth0.jl"]),
    :PhiAdapter => (:here, String[]),
    :FusedQKVLayout => (:live, ["test_breadth0.jl"]),
    :fused_qkv_rows => (:live, ["test_breadth0.jl"]),
    :adapter_for => (:live, ["test_breadth0.jl"]),
    :known_families => (:live, ["test_breadth0.jl"]),
    :architecture_spec => (:live, ["test_breadth0.jl"]),
    :materialize_architecture => (:live, ["test_breadth0.jl"]),
    :parse_config => (:live, ["test_breadth0.jl"]),
    :param_map => (:live, ["test_breadth0.jl"]),
    :family_symbol => (:live, ["test_breadth0.jl"]),
    :is_scaled => (:live, ["test_breadth0.jl"]),
    :canonical_name => (:live, ["test_breadth0.jl"]),
    :slot_for => (:live, ["test_breadth0.jl"]),
    :role_slots => (:here, String[]),
    :rope_inv_freq => (:live, ["test_breadth0.jl"]),
    :tensors_rope_inv_freq => (:live, ["test_breadth0.jl"]),
    :MetaspaceBPE => (:live, ["test_tokenizer_protocol.jl"]),
    :load_metaspace_tokenizer => (:live, ["test_tokenizer_protocol.jl"]),
    :tokenizer_metadata => (:live, ["test_tokenizer_protocol.jl"]),
    :decode => (:live, ["test_tokenizer_protocol.jl"]),
    :import_report => (:live, ["test_breadth0.jl"]),
    :compatibility_matrix => (:live, ["test_breadth0.jl"]),
    :compatibility_table => (:live, ["test_breadth0.jl"]),
    :vocab_size => (:live, ["test_tokenizer_protocol.jl"]),

    # --- Phase 5 engine (§LXXVIII) ------------------------------------------
    :Session => (:live, ["test_session.jl"]),
    :prefill! => (:live, ["test_session.jl"]),
    :decode! => (:live, ["test_session.jl", "test_session_receipts.jl"]),
    :generate => (:live, ["test_session.jl"]),
    :fork => (:live, ["test_session_fork.jl"]),
)

@testset "10D item A: export inventory — every public name has a home" begin
    actual = sort(setdiff(names(Gesso), [:Gesso]))

    missing_rows = [n for n in actual if !haskey(INVENTORY, n)]
    @test isempty(missing_rows) ||
          "public name(s) without an inventory row: $missing_rows — " *
          "a drive-by export cannot land; give it a test home or a parked " *
          "bucket in test/test_export_inventory.jl" == ""

    stale_rows = sort([n for n in keys(INVENTORY) if n ∉ actual])
    @test isempty(stale_rows) ||
          "inventory row(s) for names that are no longer exported: $stale_rows — " *
          "delete the row (10D item A)" == ""

    # :live rows must point at test files that exist — the inventory cannot
    # silently point at a renamed/deleted file
    bad_files = Tuple{Symbol, String}[]
    for (name, (bucket, files)) in INVENTORY
        bucket === :live || continue
        for f in files
            isfile(joinpath(@__DIR__, f)) || push!(bad_files, (name, f))
        end
    end
    @test isempty(bad_files) ||
          "inventory rows point at missing test files: $bad_files" == ""
end

@testset "10D item A: direct exercise of the :here names (no other home)" begin
    # GessoLogConfig: the logging surface's config type
    @test Gesso.current_config() isa Gesso.GessoLogConfig
    @test Gesso.GessoLogConfig() isa Gesso.GessoLogConfig

    # ReceiptSink: InMemorySink is the concrete sink of the suite
    @test Gesso.InMemorySink() isa Gesso.ReceiptSink

    # GPT2BPE: the tokenizer type, loaded from the tiny fixture
    tk = Gesso.load_gpt2_tokenizer(joinpath(@__DIR__, "fixtures", "gpt2_tiny"))
    @test tk isa Gesso.GPT2BPE

    # lowering_not_implemented: the §LXX stub body — a pure explicit throw
    # naming op and backend (errors.jl:135; no logging, no substitution).
    err = try
        Gesso.lowering_not_implemented(:inventory_probe, Gesso.CPUBackend())
        nothing
    catch e
        e
    end
    @test err isa Gesso.LoweringNotImplemented
    @test occursin("inventory_probe", sprint(showerror, err))
    @test occursin(":cpu", sprint(showerror, err))
end

@testset "10D item A: parked-empty fence — six contract-only modules" begin
    # §LXXXIII stays parked: these modules export exactly their own name.
    # Growing one is a work item (permitted-files law), never a drive-by.
    for m in PARKED_MODULES
        mod = getfield(Gesso, m)
        @test mod isa Module
        exported = setdiff(names(mod), [m])
        @test isempty(exported) ||
              "parked module $m exports $exported — it stays contract-only " *
              "(§LXXXIII); extending it is a work item, not a drive-by" == ""
    end
end
