# LFM2 / LFM2.5 on Lava: the gated short convolution and the biased MoE route.
# Norms, projections and attention use the shared Lava ops.

# One thread per channel. Time runs in order inside the thread because each
# step reads the history the step before it wrote.
@kernel cpu=false function _lava_short_conv_kernel!(
    mixed,
    bcx,
    w,
    hist,
    seq::Int,
    dim::Int,
    width::Int,
)
    c = @index(Global)
    if c <= dim
        for t in 1:seq
            bx = bcx[t, c] * bcx[t, 2dim+c]
            y = 0.0f0
            for p in 1:(width-1)
                y += hist[c, p] * w[c, p]
            end
            y += bx * w[c, width]
            mixed[t, c] = bcx[t, dim+c] * y
            for p in 1:(width-2)
                hist[c, p] = hist[c, p+1]
            end
            hist[c, width-1] = bx
        end
    end
end

function Gesso.Inference._short_conv_mix!(::LavaBackend, mixed, bcx, w, hist)
    seq, dim = size(mixed)
    width = size(w, 2)
    size(bcx) == (seq, 3dim) || error("short convolution: bcx must be ($seq, $(3dim)), got $(size(bcx))")
    size(hist) == (dim, width - 1) || error("short convolution: invalid history shape")
    m = _lh_device(mixed, :short_convolution)
    b = _lh_device(bcx, :short_convolution)
    ws = _lh_device(w, :short_convolution)
    hs = _lh_device(hist, :short_convolution)
    seq == 0 && return mixed
    k = _lava_short_conv_kernel!(KA.get_backend(b))
    k(m, b, ws, hs, Int(seq), Int(dim), Int(width); ndrange=dim, workgroupsize=_LH_WG)
    return mixed
end

# Routed experts on the host, as in the qwen35moe route: x comes down, the CPU
# method runs on host copies of the router, bias and banks, and y goes back up.
function Gesso.Inference._biased_moe_ffn!(backend::LavaBackend, down, normed, bt)
    host(w) = (; storage=_lh_host_weight(w.storage))
    xh = Gesso.Inference.@decode_stage :moe_d2h backend _lh_host(normed.storage)
    yh = zeros(Float32, size(down.storage))
    hb = (;
        wrouter=host(bt.wrouter),
        wexp_gate=host(bt.wexp_gate),
        wexp_up=host(bt.wexp_up),
        wexp_down=host(bt.wexp_down),
        router_gating=bt.router_gating,
        n_expert_used=bt.n_expert_used,
    )
    haskey(bt, :wrouter_bias) && (hb = merge(hb, (; wrouter_bias=host(bt.wrouter_bias))))
    Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._biased_moe_ffn!(
        Gesso.CPUBackend(),
        (; storage=yh),
        (; storage=xh),
        hb,
    )
    Gesso.Inference.@decode_stage :moe_h2d backend copyto!(down.storage, yh)
    return down
end
