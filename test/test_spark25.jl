# Spark 2.5 (Spark-X2.6) pieces: the exact-erf GELU against MPFR, the sliding
# window mask, the per-layer window and RoPE lookups, and (gated on
# GESSO_SPARK25_GGUF) the ArchitectureSpec read from the real GGUF header.

using Test
using Gesso

# erf of a BigFloat from the MPFR library Julia already links.
function _mpfr_erf(x::BigFloat)
    z = BigFloat(; precision=precision(x))
    ccall(
        (:mpfr_erf, Base.MPFR.libmpfr),
        Int32,
        (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
        z,
        x,
        Base.MPFR.MPFRRoundNearest,
    )
    return z
end

_ref_erf(x::Float64) = setprecision(() -> Float64(_mpfr_erf(BigFloat(x))), BigFloat, 256)
_ref_gelu(x::Float64) = setprecision(BigFloat, 2048) do
    b = BigFloat(x)
    Float64(b / 2 * (1 + _mpfr_erf(b / sqrt(BigFloat(2)))))
end

@testset "spark2_5: erf and GELU match MPFR" begin
    xs = vcat(
        collect(-37.0:0.01:8.0),
        [-30.0, -12.5, 1e-8, -1e-8, 0.0, 1.2499, 1.25, 1.7678, 2.0, 5.999, 6.0],
    )
    erf_err = maximum(abs(Gesso._cpu_erf(x) - _ref_erf(x)) for x in xs)
    @test erf_err < 2e-15
    rel = 0.0
    for x in xs
        ref = _ref_gelu(x)
        ref == 0 && continue
        rel = max(rel, abs((Gesso._cpu_gelu_erf(x) - ref) / ref))
    end
    # Worst case is 1 + erf cancelling just right of the erfc switch (x ≈ -1.77).
    @test rel < 5e-15
    @test Gesso._cpu_gelu_erf(0.0) == 0.0
    @test Gesso._cpu_gelu_erf(-40.0) == 0.0
    @test Gesso._cpu_gelu_erf(40.0) == 40.0

    gate = Float32[-2 -0.5; 0.25 3]
    up = Float32[1 2; 3 4]
    dst = similar(gate)
    Gesso._cpu_geglu!(dst, gate, up)
    @test dst ≈ Float32.(Gesso._cpu_gelu_erf.(Float64.(gate)) .* up)
end

@testset "spark2_5: sliding window mask" begin
    # Prefill: query i (of L) sits at absolute position K - L + i and sees only
    # the last `window` keys up to itself.
    L, K, w = 4, 6, 3
    sc = Gesso.Inference._window_mask!(zeros(L, K), L, K, w)
    for i in 1:L, j in 1:K
        pos = K - L + i
        @test isinf(sc[i, j]) == (j <= pos - w)
    end
    # A window at least as long as the context masks nothing.
    @test all(iszero, Gesso.Inference._window_mask!(zeros(3, 5), 3, 5, 5))
    # Decode (one query over K keys) keeps exactly `window` keys.
    d = Gesso.Inference._window_mask!(zeros(1, 10), 1, 10, 4)
    @test count(isfinite, d) == 4
    @test all(isfinite, d[1, 7:10])
end

@testset "spark2_5: per-layer window and RoPE lookups" begin
    full = Gesso.RoPEPolicy(; theta=5e6, rotary_dim=64)
    swa = Gesso.RoPEPolicy(; theta=1e4, rotary_dim=256)
    spec = Gesso.ArchitectureSpec(;
        family=:spark2_5,
        num_layers=4,
        hidden_size=256,
        n_heads=1,
        n_kv_heads=1,
        head_dim=256,
        intermediate_size=512,
        vocab_size=16,
        activation_kind=:geglu,
        rope=full,
        rope_swa=swa,
        sliding_window=512,
        swa_layers=[true, true, true, false],
    )
    @test [Gesso.Inference.layer_window(spec, l) for l in 0:3] == [512, 512, 512, 0]
    @test Gesso.Inference.layer_rope(spec, 0) === swa
    @test Gesso.Inference.layer_rope(spec, 3) === full
    @test_throws ErrorException Gesso.ArchitectureSpec(;
        family=:spark2_5,
        num_layers=4,
        hidden_size=256,
        n_heads=1,
        n_kv_heads=1,
        head_dim=256,
        intermediate_size=512,
        vocab_size=16,
        rope=full,
        sliding_window=512,
        swa_layers=[true, false],
    )
end

const _SPARK25_GGUF = get(ENV, "GESSO_SPARK25_GGUF", "")

@testset "gguf_architecture_spec: Spark-X2.6-4B" begin
    if !isfile(_SPARK25_GGUF)
        @test_skip "set GESSO_SPARK25_GGUF to the Spark-X2.6-4B GGUF"
        return
    end
    spec = Gesso.gguf_architecture_spec(_SPARK25_GGUF)
    @test spec.family === :spark2_5
    @test (spec.num_layers, spec.hidden_size, spec.intermediate_size) == (36, 2560, 10240)
    @test (spec.n_heads, spec.n_kv_heads, spec.head_dim, spec.vocab_size) ==
          (16, 4, 256, 131072)
    @test spec.activation_kind === :geglu
    @test spec.sliding_window == 512
    @test spec.swa_layers == repeat([true, true, true, false], 9)
    @test spec.rope.theta == 5e6 && spec.rope.rotary_dim == 64
    # rotary_dim 0 means the whole head (256 dims).
    @test spec.rope_swa.theta == 1e4 && spec.rope_swa.rotary_dim == 0
    @test :attn_head_gate in spec.features
end
