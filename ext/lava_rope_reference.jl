# Imported Float64 frequencies use the same coefficient and rounding contract as CPU.
# Only position metadata crosses the host boundary; Q/K remain on Vulkan.
@kernel cpu=false function _lava_rope_reference_kernel!(
    x,
    @Const(C),
    @Const(S),
    seq::Int,
    nh::Int,
    rd::Int,
    interleaved::Bool,
)
    lin=@index(Global, Linear)
    if lin<=seq*nh
        t=(lin-1)%seq+1
        h=(lin-1)÷seq+1
        T=eltype(C)
        half=rd÷2
        @inbounds for i = 1:half
            c=C[t, i]
            s=S[t, i]
            a=interleaved ? 2i-1 : i
            b=interleaved ? 2i : i+half
            x1=T(x[t, h, a])
            x2=T(x[t, h, b])
            x[t, h, a]=Float32(x1*c-x2*s)
            x[t, h, b]=Float32(x1*s+x2*c)
        end
    end
end

function _lava_rope_reference!(q, k, positions, freq, rd, interleaved)
    seq = length(positions)
    seq == 0 && return nothing
    c = Matrix{Float64}(undef, seq, rd ÷ 2)
    s = similar(c)
    for t in eachindex(positions), i = 1:(rd÷2)
        phase = Float64(positions[t]) * Float64(freq[i])
        c[t, i] = cos(phase)
        s[t, i] = sin(phase)
    end
    # Per-call buffers avoid persistent context ownership, eviction and cache races.
    cd = Lava.LavaArray{Float64}(c)
    sd = Lava.LavaArray{Float64}(s)
    for x in (q, k)
        size(x, 2) == 0 && continue
        kernel = Lava.with_no_contraction(
            Lava.with_float_controls(_lava_rope_reference_kernel!(KA.get_backend(x)), (32, 64)),
        )
        kernel(
            x,
            cd,
            sd,
            size(x, 1),
            size(x, 2),
            rd,
            interleaved;
            ndrange = size(x, 1) * size(x, 2),
        )
    end
    return nothing
end
