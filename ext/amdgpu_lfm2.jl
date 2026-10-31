# LFM2 device ops: the gated short convolution and the router-biased MoE FFN.
#
# The short convolution is a recurrence over the sequence, so one thread owns a
# channel and walks time in order; there is no reduction to tree over. The MoE
# router and expert bank stay on the CPU reference path (see amdgpu_host.jl),
# and the device receives the mixed rows back.

@kernel function _ka_short_conv_kernel!(
    mixed,
    @Const(bcx),
    @Const(w),
    hist,
    seq::Int,
    dim::Int,
    width::Int,
)
    c = @index(Global, Linear)
    if c <= dim
        for t in 1:seq
            bx = bcx[t, c] * bcx[t, 2 * dim + c]
            y = 0.0f0
            for p in 1:(width-1)
                y += hist[c, p] * w[c, p]
            end
            y += bx * w[c, width]
            mixed[t, c] = bcx[t, dim + c] * y
            for p in 1:(width-2)
                hist[c, p] = hist[c, p+1]
            end
            hist[c, width-1] = bx
        end
    end
end

function _ka_short_conv!(ka, mixed, bcx, w, hist; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    seq, dim = size(mixed)
    width = size(w, 2)
    size(bcx) == (seq, 3 * dim) || throw(
        ArgumentError("short convolution needs bcx of size ($seq, $(3 * dim)), got $(size(bcx))"),
    )
    size(hist) == (dim, width - 1) || throw(
        ArgumentError("short convolution needs history of size ($dim, $(width - 1)), got $(size(hist))"),
    )
    width >= 2 || throw(ArgumentError("short convolution needs width >= 2, got $width"))
    (seq == 0 || dim == 0) && return mixed
    _ka_short_conv_kernel!(ka, wg)(mixed, bcx, w, hist, seq, dim, width; ndrange=dim)
    return mixed
end

function Gesso.Inference._short_conv_mix!(b::KADeviceBackend, mixed, bcx, w, hist)
    ms = _ka_device_storage!(:short_convolution, b, (; storage=mixed))
    bs = _ka_device_storage!(:short_convolution, b, (; storage=bcx))
    ws = _ka_device_storage!(:short_convolution, b, (; storage=w))
    hs = _ka_device_storage!(:short_convolution, b, (; storage=hist))
    return _ka_short_conv!(b.ka, ms, bs, ws, hs)
end

# The router's CPU reference: device weights go in as host views, the routed
# output comes back to the device in one copy.
_ka_host_view(w) = (; storage=_ka_host_storage(w.storage))

function Gesso.Inference._biased_moe_ffn!(b::KADeviceBackend, down, normed, bt)
    xh = Gesso.Inference.@decode_stage :moe_d2h b _ka_host_storage(normed.storage)
    yh = zeros(Float32, size(down.storage))
    hb = (;
        wrouter=_ka_host_view(bt.wrouter),
        wexp_gate=_ka_host_view(bt.wexp_gate),
        wexp_up=_ka_host_view(bt.wexp_up),
        wexp_down=_ka_host_view(bt.wexp_down),
        router_gating=bt.router_gating,
        n_expert_used=bt.n_expert_used,
    )
    if haskey(bt, :wrouter_bias)
        hb = merge(hb, (; wrouter_bias=_ka_host_view(bt.wrouter_bias)))
    end
    Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._biased_moe_ffn!(
        Gesso.CPUBackend(), (; storage=yh), (; storage=xh), hb,
    )
    Gesso.Inference.@decode_stage :moe_h2d b copyto!(down.storage, yh)
    return down
end
