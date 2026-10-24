# Batched Q8_K host MoE (src/Inference/moe_host.jl) against the per-expert
# host MoE (_moe_host_experts! in gguf_quants.jl). Reuses the slab helpers from
# test_moe_kquant_experts.jl, so it is included after that file.
#   * K-quant slabs: the two paths run the same per-row dot products on the same
#     Q8_K activations, so they agree bit for bit
#   * IQ gate/up: the per-expert path reads float x while the batched path
#     quantizes x to Q8_K, so the two differ by the Q8_K activation error
#   * Q8_0 slabs have no Q8_K kernel and must take the per-expert path exactly
#   * references: an exact Float64 SwiGLU over the dequantized weights with
#     float x (float-x reference), and the same SwiGLU with the activation and
#     hidden vector rounded to Q8_K the way the batched path rounds them
#     (Q8_K reference). The batched path must match the Q8_K reference to f32
#     accumulation error.
#   * red control: routing weights permuted across experts must miss the tolerance

using Random: Xoshiro

_mhb_relrms(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

# Exact dequant of expert `e` as Float64 (rows ne1, cols ne0). Column i is the
# float kernel applied to the unit vector e_i, so each entry is the stored
# weight up to one f32 product.
function _mhb_dequant_f64(s::_MKE.GgufQuantSlice, e::Int)
    W = zeros(Float64, s.ne1, s.ne0)
    x = zeros(Float32, s.ne0)
    y = zeros(Float32, s.ne1)
    for i in 1:s.ne0
        x[i] = 1.0f0
        _MKE._qgemv_expert!(y, x, s, e)
        x[i] = 0.0f0
        W[:, i] .= Float64.(y)
    end
    return W
end

_mhb_silu64(g::Float64) = g / (1.0 + exp(-g))

# Vector rounded to Q8_K and back, as Float64.
function _mhb_q8k_round(v::AbstractVector{Float32})
    xq = _MKE.Q8KVec(div(length(v), _MKE.QK_K))
    _MKE.quantize_q8_K!(xq, v)
    return Float64.(_mke_dequant_act(xq))
end

# Exact SwiGLU over the selected experts in Float64. With `actq` the activation
# and the hidden vector are Q8_K-rounded as the batched path rounds them.
function _mhb_swiglu_f64(xf, order, weights, nused, gate_s, up_s, down_s, h, ff; actq::Bool)
    y = zeros(Float64, h)
    x64 = actq ? _mhb_q8k_round(Float32.(xf)) : Float64.(xf)
    for k in 1:nused
        e = order[k]
        Wg = _mhb_dequant_f64(gate_s, e)
        Wu = _mhb_dequant_f64(up_s, e)
        Wd = _mhb_dequant_f64(down_s, e)
        g = Wg * x64
        u = Wu * x64
        hv = _mhb_silu64.(g) .* u
        hin = actq ? _mhb_q8k_round(Float32.(hv)) : hv
        y .+= Float64(weights[k]) .* (Wd * hin)
    end
    return y
end

@testset "batched host MoE vs per-expert host MoE" begin
    rng = Xoshiro(43)
    h, ff, nexp, nused = 512, 256, 6, 3
    order = [4, 1, 6]
    weights = Float32[0.5, 0.3, 0.2]
    permuted = weights[[3, 1, 2]]
    cases = (
        (
            "Q4_K gate/up, Q6_K down",
            _MKE.GGML_TYPE_Q4_K,
            _MKE.GGML_TYPE_Q4_K,
            _MKE.GGML_TYPE_Q6_K,
            1.0e-6,
        ),
        (
            "Q2_K gate/up, Q3_K down",
            _MKE.GGML_TYPE_Q2_K,
            _MKE.GGML_TYPE_Q2_K,
            _MKE.GGML_TYPE_Q3_K,
            1.0e-6,
        ),
        (
            "IQ4_XS gate/up, Q5_K down",
            _MKE.GGML_TYPE_IQ4_XS,
            _MKE.GGML_TYPE_IQ4_XS,
            _MKE.GGML_TYPE_Q5_K,
            5.0e-2,
        ),
        (
            "IQ2_XS gate/up, IQ3_XXS down (35B)",
            _MKE.GGML_TYPE_IQ2_XS,
            _MKE.GGML_TYPE_IQ2_XS,
            _MKE.GGML_TYPE_IQ3_XXS,
            5.0e-2,
        ),
        (
            "IQ2_XS gate/up, IQ4_XS down (35B)",
            _MKE.GGML_TYPE_IQ2_XS,
            _MKE.GGML_TYPE_IQ2_XS,
            _MKE.GGML_TYPE_IQ4_XS,
            5.0e-2,
        ),
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
            yref,
            x,
            order,
            weights,
            nused,
            gate_s,
            up_s,
            down_s,
            zeros(Float32, ff),
            zeros(Float32, ff),
            zeros(Float32, ff),
            zeros(Float32, h),
        )
        ybat = zeros(Float32, h)
        _MKE._moe_host_experts_auto!(ybat, x, order, weights, nused, gate_s, up_s, down_s)
        rel = _mhb_relrms(ybat, yref)
        println(
            "moe batched ",
            label,
            " rel_rms_vs_per_expert=",
            rel,
            " max_abs=",
            maximum(abs.(ybat .- yref)),
            " bitwise=",
            ybat == yref,
        )
        # Float-x per-expert comparison is loose for IQ gate/up on purpose: the
        # batched path rounds x to Q8_K, as ggml's IQ vec_dot does.
        @test rel <= tol

        yf64 = _mhb_swiglu_f64(
            x,
            order,
            weights,
            nused,
            gate_s,
            up_s,
            down_s,
            h,
            ff;
            actq=false,
        )
        yq8 = _mhb_swiglu_f64(
            x,
            order,
            weights,
            nused,
            gate_s,
            up_s,
            down_s,
            h,
            ff;
            actq=true,
        )
        rel_pe_f64 = _mhb_relrms(yref, yf64)
        rel_bat_f64 = _mhb_relrms(ybat, yf64)
        rel_pe_q8 = _mhb_relrms(yref, yq8)
        rel_bat_q8 = _mhb_relrms(ybat, yq8)
        println(
            "moe batched ",
            label,
            " rel_rms_vs_f64_float_x: per_expert=",
            rel_pe_f64,
            " batched=",
            rel_bat_f64,
            " batched_over_per_expert=",
            rel_bat_f64 / max(rel_pe_f64, eps(Float64)),
            " | rel_rms_vs_f64_q8k: per_expert=",
            rel_pe_q8,
            " batched=",
            rel_bat_q8,
        )
        @test rel_bat_q8 <= 1.0e-6

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
            yref,
            x,
            order,
            weights,
            nused,
            gate_s,
            up_s,
            down_s,
            zeros(Float32, ff),
            zeros(Float32, ff),
            zeros(Float32, ff),
            zeros(Float32, h),
        )
        yauto = zeros(Float32, h)
        _MKE._moe_host_experts_auto!(yauto, x, order, weights, nused, gate_s, up_s, down_s)
        @test yauto == yref
    end
end
