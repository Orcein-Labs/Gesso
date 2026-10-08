# qwen35moe GGUF doorway: header + ArchitectureSpec against the local
# Ornith-1.5-35B-A3B file. Full `load_gguf` (~9 GiB F32 non-experts) is
# gated on GESSO_LOAD_35B=1.

using Test
using Gesso

const _ORNITH35_GGUF = get(
    ENV,
    "GESSO_ORNITH35_GGUF",
    joinpath(@__DIR__, "..", "..", "Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf"),
)
const _GGML_BIN = get(
    ENV,
    "GESSO_GGML_DIR",
    joinpath(@__DIR__, "..", "..", "llama-cpp-upstream", "build", "bin"),
)

@testset "gguf_architecture_spec: Ornith-1.5-35B-A3B" begin
    if !isfile(_ORNITH35_GGUF)
        @test_skip "Ornith 35B GGUF not at $_ORNITH35_GGUF"
        return
    end
    spec = Gesso.gguf_architecture_spec(_ORNITH35_GGUF)
    @test spec.family === :qwen3_5_moe
    @test spec.num_layers == 40
    @test spec.hidden_size == 2048
    @test spec.n_heads == 16
    @test spec.n_kv_heads == 2
    @test spec.head_dim == 256
    @test spec.vocab_size == 248320
    @test spec.moe !== nothing
    @test spec.moe.n_experts == 256
    @test spec.moe.n_expert_used == 8
    @test spec.moe.expert_ff == 512
    @test spec.moe.shared_ff == 512
    @test spec.rope.rotary_dim == 64
    @test spec.rope.mrope_section == [11, 11, 10]
    @test :rms_unit_offset ∉ spec.features
    @test Gesso.layer_mixer(spec, 0) === :linear_attention
    @test Gesso.layer_mixer(spec, 3) === :full_attention
    @test spec.mixer.num_key_heads == 16
    @test spec.mixer.key_head_dim == 128
    @test spec.mixer.num_value_heads == 32
    @test spec.mixer.value_head_dim == 128
    @test spec.mixer.conv_kernel_dim == 4
    @test spec.mixer.v_head_tiled === true

    hdr = Gesso.gguf_header(_ORNITH35_GGUF)
    @test length(hdr.tensors) == 733
    @test hdr.kv["general.architecture"] == "qwen35moe"
    names = Set(t.name for t in hdr.tensors)
    @test "token_embd.weight" in names
    @test "blk.0.ffn_gate_exps.weight" in names
    @test "blk.39.ssm_a" ∉ names          # layer 39 is full-attn (40 % 4 == 0)
    @test "blk.0.ssm_a" in names
    @test "blk.3.attn_q.weight" in names
end

@testset "V-head unreorder is a bijection" begin
    n = 16 * 2 * 128
    p = Gesso.Inference._v_head_perm(16, 2, 128)
    invp = Gesso.Inference._v_head_perm(2, 16, 128)
    @test length(p) == n
    @test sort(p) == collect(1:n)
    @test p[invp] == collect(1:n)
    @test invp[p] == collect(1:n)
    v = collect(Float32, 1:32)
    p1 = Gesso.Inference._v_head_perm(16, 2, 1)
    inv1 = Gesso.Inference._v_head_perm(2, 16, 1)
    @test Gesso.Inference._gather_axis(Gesso.Inference._gather_axis(v, 1, p1), 1, inv1) == v
end

@testset "gguf ggml to_float + lazy expert" begin
    if !isfile(_ORNITH35_GGUF)
        @test_skip "Ornith 35B GGUF not at $_ORNITH35_GGUF"
        return
    end
    if !isfile(joinpath(_GGML_BIN, "libggml-base.so"))
        @test_skip "libggml-base.so not at $_GGML_BIN"
        return
    end
    hdr = Gesso.gguf_header(_ORNITH35_GGUF)
    # F32 output_norm: baked RMS +1, so values sit around 1.
    info = only(t for t in hdr.tensors if t.name == "output_norm.weight")
    @test info.ggml_type == Int32(0)
    arr = Gesso.Inference._gguf_array(hdr, info)
    @test arr isa Vector{Float32}
    @test length(arr) == 2048
    @test all(isfinite, arr)
    @test minimum(arr) > 0.5
    @test maximum(arr) < 4.0

    # One IQ2_XS expert: (out, in) = (512, 2048) after reverse-perm.
    einfo = only(t for t in hdr.tensors if t.name == "blk.0.ffn_gate_exps.weight")
    slice = Gesso.Inference._expert_slice(hdr, einfo)
    @test slice isa Gesso.GgufQuantSlice
    @test size(slice) == (256, 512, 2048)
    W = Gesso.gguf_dequant_expert(slice, 1)
    @test size(W) == (512, 2048)
    @test eltype(W) === Float32
    @test all(isfinite, W)
    @test any(!iszero, W)

    traits32 = Gesso.Inference._ggml_traits(0)
    @test traits32.blck_size == 1
    @test traits32.type_size == 4

    # Fused qgemv matches ggml to_float + BLAS on a real IQ2_XS expert.
    rng = GessoTestHelpers.deterministic_rng(42)
    x = randn(rng, Float32, slice.ne0)
    y_ref = W * x
    y = zeros(Float32, slice.ne1)
    Gesso.Inference._qgemv_expert!(y, x, slice, 1)
    @test y ≈ y_ref atol = 2e-4 rtol = 2e-4
    packed = Gesso.Inference.gpu_quant_slice(
        slice,
        Vector{UInt8}(Gesso.Inference.packed_quant_bytes(slice)),
    )
    @test packed isa Gesso.GpuQuantSlice
    y2 = zeros(Float32, slice.ne1)
    Gesso.Inference._qgemv_expert!(y2, x, packed, 1)
    @test y2 ≈ y_ref atol = 2e-4 rtol = 2e-4
end

@testset "GGUF MoE BLAS matches dense scalar" begin
    seq, h, ff, n_exp, n_used = 2, 8, 4, 3, 2
    rng = GessoTestHelpers.deterministic_rng(7)
    x = randn(rng, Float32, seq, h)
    router = randn(rng, Float32, n_exp, h)
    gate3 = randn(rng, Float32, n_exp, ff, h)
    up3 = randn(rng, Float32, n_exp, ff, h)
    down3 = randn(rng, Float32, n_exp, h, ff)
    y_dense = zeros(Float32, seq, h)
    Gesso._cpu_moe_topk_swiglu!(y_dense, x, router, gate3, up3, down3, n_used)

    function pack_slice(A3, ne0, ne1)
        parts = UInt8[]
        for e in 1:size(A3, 1)
            W = A3[e, :, :]
            g = vec(permutedims(W, (2, 1)))
            append!(parts, reinterpret(UInt8, g))
        end
        Gesso.GgufQuantSlice(
            parts,
            0,
            Int32(0),
            size(A3, 1),
            ne0,
            ne1,
            ne0 * ne1 * 4,
            C_NULL,
            Int64(1),
        )
    end
    y_gguf = zeros(Float32, seq, h)
    Gesso.Inference._cpu_moe_topk_swiglu_gguf!(
        y_gguf,
        x,
        router,
        pack_slice(gate3, h, ff),
        pack_slice(up3, h, ff),
        pack_slice(down3, ff, h),
        n_used,
    )
    @test y_gguf ≈ y_dense rtol = 1e-5 atol = 1e-5
end

@testset "load_gguf 35B (opt-in)" begin
    if get(ENV, "GESSO_LOAD_35B", "") != "1"
        @test_skip "set GESSO_LOAD_35B=1 to materialize non-expert F32 (~9 GiB)"
        return
    end
    model, tensors, spec = Gesso.load_gguf(_ORNITH35_GGUF)
    @test spec.family === :qwen3_5_moe
    @test length(tensors.blocks) == 40
    @test tensors.blocks[1].wexp_gate.storage isa Gesso.GgufQuantSlice
    @test eltype(tensors.embedding.storage) === Float32
    ids = Gesso.reference_generate(model, tensors, [9419]; max_new_tokens=1, eps=spec.rms_norm_eps)
    @test length(ids) == 2
    @test ids[1] == 9419
    @test ids[2] == 11
end
