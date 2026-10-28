# CPU MoE expert GEMV for K-quant and Q8_0 experts (src/Inference/gguf_quants.jl,
# _qgemv_bytes! routed through the native Q8_K / Q8_0 kernels):
#   * synthetic expert slabs of every K-quant and Q8_0 type: the expert GEMV
#     against the F32 dequant reference, at tolerance
#   * the top-k SwiGLU MoE (_moe_host_experts!) against the same reference, for
#     K-quant gate, up and down slabs
#   * real rows from the Kolibri-1 Q2_K GGUF (env GESSO_KOLIBRI_Q2K_GGUF; skipped
#     if unset). Header and the rows tested only.

using Random: Xoshiro
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _MKE = Gesso.Inference

_mke_qblock(ty::Int32) = ty == _MKE.GGML_TYPE_Q8_0 ? 32 : _MKE.QK_K

# Random valid expert bytes: arbitrary in every field, finite f16 super-scales
# at the format's d and dmin offsets.
function _mke_random_bytes(rng, ty::Int32, nexp::Int, nrows::Int, ne0::Int)
    bsz = _MKE._iq_block_bytes(ty)
    nb = div(ne0, _mke_qblock(ty))
    w = rand(rng, UInt8, nexp * nrows * nb * bsz)
    doff = _MKE._q8k_d_offset(ty)
    dmo = _MKE._q8k_dmin_offset(ty)
    for blk in 0:(nexp*nrows*nb-1)
        for off in (doff, dmo)
            off < 0 && continue
            bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
            w[blk*bsz+off+1] = UInt8(bits & 0xff)
            w[blk*bsz+off+2] = UInt8(bits >> 8)
        end
    end
    return w
end

# d * q per value for a quantized activation: Q8_K or Q8_0 blocks.
_mke_dequant_act(xq::_MKE.Q8KVec) = _MKE.dequantize_q8_K(xq)
function _mke_dequant_act(xq::_MKE.Q80Vec)
    out = Vector{Float32}(undef, length(xq))
    for i in eachindex(out)
        out[i] = xq.d[(i-1)÷32+1] * Float32(xq.qs[i])
    end
    return out
end

function _mke_slice(w::Vector{UInt8}, ty::Int32, nexp::Int, ne0::Int, ne1::Int)
    rb = _MKE._qrow_bytes(ty, ne0)
    return Gesso.GgufQuantSlice(
        w,
        0,
        ty,
        nexp,
        ne0,
        ne1,
        ne1 * rb,
        Int64(_mke_qblock(ty)),
    )
end

# F32 dequant of rows 1:nrows of a slab at byte offset off0 (0-based).
function _mke_dequant_rows(w, off0::Int, ty::Int32, ne0::Int, nrows::Int)
    rb = _MKE._qrow_bytes(ty, ne0)
    out = zeros(Float32, nrows, ne0)
    row = zeros(Float32, ne0)
    for j in 1:nrows
        _MKE._kq_dequant_row!(row, w, off0 + (j - 1) * rb, ty, ne0)
        out[j, :] .= row
    end
    return out
end

const _MKE_TYPES = (
    ("Q2_K", _MKE.GGML_TYPE_Q2_K),
    ("Q3_K", _MKE.GGML_TYPE_Q3_K),
    ("Q4_K", _MKE.GGML_TYPE_Q4_K),
    ("Q5_K", _MKE.GGML_TYPE_Q5_K),
    ("Q6_K", _MKE.GGML_TYPE_Q6_K),
    ("Q8_0", _MKE.GGML_TYPE_Q8_0),
)

@testset "K-quant expert GEMV vs dequant + F32" begin
    rng = Xoshiro(40)
    ne0, ne1, nexp = 512, 24, 3
    for (name, ty) in _MKE_TYPES
        w = _mke_random_bytes(rng, ty, nexp, ne1, ne0)
        s = _mke_slice(w, ty, nexp, ne0, ne1)
        x = randn(rng, Float32, ne0)
        xq = _MKE._q8k_quantize_act(ty, x)
        for e in (1, nexp)
            y = zeros(Float32, ne1)
            _MKE._qgemv_expert!(y, x, s, e)
            # reference: the same expert's rows, dequantized, against the
            # dequantized activation the kernel actually multiplies
            off0 = _MKE._quant_expert_off0(s, e)
            Wd = _mke_dequant_rows(w, off0, ty, ne0, ne1)
            xr = _mke_dequant_act(xq)
            yref = Float64.(Wd) * Float64.(xr)
            @test maximum(abs, Float64.(y) .- yref) <= 1.0e-4 * maximum(abs, yref)
            # against the unquantized activation: activation quantization error only
            yexact = Float64.(Wd) * Float64.(x)
            relrms = sqrt(sum(abs2, Float64.(y) .- yexact) / sum(abs2, yexact))
            @test relrms <= 2.0e-2
            # the AVX2 path of the same expert agrees bit for bit with the scalar path
            if _MKE._cpu_has_avx2()
                ys = zeros(Float32, ne1)
                ya = zeros(Float32, ne1)
                _MKE._qgemv_q8k!(
                    ys,
                    xq,
                    _MKE._quant_expert_bytes(s, e),
                    ty,
                    ne0,
                    ne1;
                    simd=false,
                )
                _MKE._qgemv_q8k!(
                    ya,
                    xq,
                    _MKE._quant_expert_bytes(s, e),
                    ty,
                    ne0,
                    ne1;
                    simd=true,
                )
                @test reinterpret.(UInt32, ys) == reinterpret.(UInt32, ya)
            end
        end
    end
end


@testset "K-quant top-k SwiGLU MoE vs dequant + F32" begin
    rng = Xoshiro(41)
    h, ff, nexp, nused = 512, 256, 4, 2
    cases = (
        (
            "Q4_K gate/up, Q6_K down",
            _MKE.GGML_TYPE_Q4_K,
            _MKE.GGML_TYPE_Q4_K,
            _MKE.GGML_TYPE_Q6_K,
        ),
        (
            "Q2_K gate/up, Q3_K down",
            _MKE.GGML_TYPE_Q2_K,
            _MKE.GGML_TYPE_Q2_K,
            _MKE.GGML_TYPE_Q3_K,
        ),
        (
            "Q8_0 gate/up, Q5_K down",
            _MKE.GGML_TYPE_Q8_0,
            _MKE.GGML_TYPE_Q8_0,
            _MKE.GGML_TYPE_Q5_K,
        ),
    )
    for (label, tg, tu, td) in cases
        wg = _mke_random_bytes(rng, tg, nexp, ff, h)
        wu = _mke_random_bytes(rng, tu, nexp, ff, h)
        wd = _mke_random_bytes(rng, td, nexp, h, ff)
        gate_s = _mke_slice(wg, tg, nexp, h, ff)
        up_s = _mke_slice(wu, tu, nexp, h, ff)
        down_s = _mke_slice(wd, td, nexp, ff, h)
        x = randn(rng, Float32, h)
        order = [3, 1]
        weights = Float32[0.7, 0.3]
        yrow = zeros(Float32, h)
        gacc = zeros(Float32, ff)
        uacc = zeros(Float32, ff)
        hact = zeros(Float32, ff)
        ybuf = zeros(Float32, h)
        _MKE._moe_host_experts!(
            yrow,
            x,
            order,
            weights,
            nused,
            gate_s,
            up_s,
            down_s,
            gacc,
            uacc,
            hact,
            ybuf,
        )

        # reference: each matmul sees the activation the kernel quantizes it to
        xq_in = _MKE._q8k_quantize_act(tg, x)
        xin = _mke_dequant_act(xq_in)
        yref = zeros(Float64, h)
        for k in 1:nused
            e = order[k]
            Wg = _mke_dequant_rows(wg, _MKE._quant_expert_off0(gate_s, e), tg, h, ff)
            Wu = _mke_dequant_rows(wu, _MKE._quant_expert_off0(up_s, e), tu, h, ff)
            g = Float64.(Wg) * Float64.(xin)
            u = Float64.(Wu) * Float64.(xin)
            hv = (g ./ (1 .+ exp.(-g))) .* u
            hq = _mke_dequant_act(_MKE._q8k_quantize_act(td, Float32.(hv)))
            Wd = _mke_dequant_rows(wd, _MKE._quant_expert_off0(down_s, e), td, ff, h)
            yref .+= weights[k] .* (Float64.(Wd) * Float64.(hq))
        end
        relrms = sqrt(sum(abs2, Float64.(yrow) .- yref) / sum(abs2, yref))
        println("moe ", label, " rel_rms_vs_dequant=", relrms)
        @test relrms <= 1.0e-3
    end
end

@testset "Kolibri-1 Q2_K expert rows vs dequant + F32" begin
    path = get(ENV, "GESSO_KOLIBRI_Q2K_GGUF", "")
    if !isfile(path)
        _skip(
            "set GESSO_KOLIBRI_Q2K_GGUF to the Kolibri-1 Q2_K GGUF — real expert rows skipped",
        )
    else
        hdr = _MKE.gguf_header(path)
        rng = Xoshiro(42)
        wanted = (
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        )
        for nm in wanted
            idx = findfirst(t -> t.name == nm, hdr.tensors)
            idx === nothing && continue
            info = hdr.tensors[idx]
            ty = info.ggml_type
            ty in (
                _MKE.GGML_TYPE_Q2_K,
                _MKE.GGML_TYPE_Q3_K,
                _MKE.GGML_TYPE_Q4_K,
                _MKE.GGML_TYPE_Q6_K,
            ) || continue
            ne0, ne1 = info.dims[1], info.dims[2]
            rb = _MKE._qrow_bytes(ty, ne0)
            base = Int(hdr.data_start) + Int(info.offset)
            rows = unique(round.(Int, range(1, ne1; length=min(16, ne1))))
            x = randn(rng, Float32, ne0)
            xq = _MKE._q8k_quantize_act(ty, x)
            xr = _mke_dequant_act(xq)
            for j in rows
                # the single-row byte view is the only file read
                wrow = view(hdr.data, (base+(j-1)*rb+1):(base+j*rb))
                y = zeros(Float32, 1)
                _MKE._qgemv_bytes!(y, x, wrow, ty, ne0, 1, 1.0f0, 0.0f0)
                wd = _mke_dequant_rows(hdr.data, base + (j - 1) * rb, ty, ne0, 1)
                yref = sum(Float64.(wd[1, :]) .* Float64.(xr))
                @test abs(Float64(y[1]) - yref) <= 1.0e-4 * max(abs(yref), 1.0e-3)
            end
            println(
                "kolibri ",
                nm,
                " type=",
                ty,
                " ne0=",
                ne0,
                " ne1=",
                ne1,
                " rows_checked=",
                length(rows),
            )
        end
    end
end
