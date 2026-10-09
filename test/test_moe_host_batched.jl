# Batched Q8_K host MoE (src/Inference/moe_host.jl) against the per-expert
# host MoE (_moe_host_experts! in gguf_quants.jl). Reuses the slab helpers from
# test_moe_kquant_experts.jl, so it is included after that file.
#   * K-quant slabs: the two paths run the same per-row dot products on the same
#     Q8_K activations, so they agree to rounding
#   * IQ4_XS gate/up: the per-expert path reads float x while the batched path
#     quantizes x to Q8_K, so the tolerance is the Q8_K activation error
#   * Q8_0 slabs have no Q8_K kernel and must take the per-expert path exactly
#   * red control: routing weights permuted across experts must miss the tolerance

using Random: Xoshiro

_mhb_relrms(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

@testset "batched host MoE vs per-expert host MoE" begin
    rng = Xoshiro(43)
    h, ff, nexp, nused = 512, 256, 6, 3
    order = [4, 1, 6]
    weights = Float32[0.5, 0.3, 0.2]
    permuted = weights[[3, 1, 2]]
    cases = (
        ("Q4_K gate/up, Q6_K down", _MKE.GGML_TYPE_Q4_K, _MKE.GGML_TYPE_Q4_K, _MKE.GGML_TYPE_Q6_K, 1.0e-6),
        ("Q2_K gate/up, Q3_K down", _MKE.GGML_TYPE_Q2_K, _MKE.GGML_TYPE_Q2_K, _MKE.GGML_TYPE_Q3_K, 1.0e-6),
        ("IQ4_XS gate/up, Q5_K down", _MKE.GGML_TYPE_IQ4_XS, _MKE.GGML_TYPE_IQ4_XS, _MKE.GGML_TYPE_Q5_K, 5.0e-2),
    )
    for (label, tg, tu, td, tol) in cases
        wg = _mke_random_bytes(rng, tg, nexp, ff, h)
        wu = _mke_random_bytes(rng, tu, nexp, ff, h)
        wd = _mke_random_bytes(rng, td, nexp, h, ff)
        gate_s = _mke_slice(wg, tg, nexp, h, ff)
        up_s = _mke_slice(wu, tu, nexp, h, ff)
        down_s = _mke_slice(wd, td, nexp, ff, h)
        x = randn(rng, Float32, h)

        @test _MKE._moe_host_batchable(gate_s, up_s, down_s, h, ff)

        yref = zeros(Float32, h)
        _MKE._moe_host_experts!(
            yref, x, order, weights, nused, gate_s, up_s, down_s,
            zeros(Float32, ff), zeros(Float32, ff), zeros(Float32, ff), zeros(Float32, h),
        )
        ybat = zeros(Float32, h)
        _MKE._moe_host_experts_auto!(ybat, x, order, weights, nused, gate_s, up_s, down_s)
        rel = _mhb_relrms(ybat, yref)
        println("moe batched ", label, " rel_rms_vs_per_expert=", rel,
            " max_abs=", maximum(abs.(ybat .- yref)), " bitwise=", ybat == yref)
        @test rel <= tol

        ybad = zeros(Float32, h)
        _MKE._moe_host_experts_auto!(ybad, x, order, permuted, nused, gate_s, up_s, down_s)
        ctrl = _mhb_relrms(ybad, yref)
        println("moe batched ", label, " red_control_permuted_weights_rel_rms=", ctrl)
        @test ctrl > 10 * tol
    end

    @testset "Q8_0 gate falls back to the per-expert path" begin
        ty = _MKE.GGML_TYPE_Q8_0
        wg = _mke_random_bytes(rng, ty, nexp, ff, h)
        wu = _mke_random_bytes(rng, ty, nexp, ff, h)
        wd = _mke_random_bytes(rng, _MKE.GGML_TYPE_Q5_K, nexp, h, ff)
        gate_s = _mke_slice(wg, ty, nexp, h, ff)
        up_s = _mke_slice(wu, ty, nexp, h, ff)
        down_s = _mke_slice(wd, _MKE.GGML_TYPE_Q5_K, nexp, ff, h)
        x = randn(rng, Float32, h)
        @test !_MKE._moe_host_batchable(gate_s, up_s, down_s, h, ff)
        yref = zeros(Float32, h)
        _MKE._moe_host_experts!(
            yref, x, order, weights, nused, gate_s, up_s, down_s,
            zeros(Float32, ff), zeros(Float32, ff), zeros(Float32, ff), zeros(Float32, h),
        )
        yauto = zeros(Float32, h)
        _MKE._moe_host_experts_auto!(yauto, x, order, weights, nused, gate_s, up_s, down_s)
        @test yauto == yref
    end
end
