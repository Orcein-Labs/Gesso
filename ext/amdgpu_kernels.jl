# Reference kernels for KADeviceBackend. Every kernel here follows the same
# shape, so ports copy it:
#
#   * the workgroup size is a launch parameter read with `@uniform @groupsize()`,
#     a power of two of at most `_KA_MAX_WG`; nothing assumes a wavefront size
#   * reductions fold through local memory as a tree with a uniform trip count
#   * any value needed after `@synchronize` is re-read from `@index`, local
#     memory or a kernel argument (KA's CPU backend splits the kernel at each
#     barrier and does not carry plain variables across it)

const _KA_MAX_WG = 256

_ka_check_wg(wg::Int) =
    ispow2(wg) && wg <= _KA_MAX_WG ||
    throw(gesso_error(ERR_INVALID_PLAN, "workgroup size $wg must be a power of two ≤ $_KA_MAX_WG"))

# Row RMS-norm over the LAST axis; one workgroup per flattened leading-axis row.
# The CPU contract: square in Float32, accumulate in Float64, store Float32. A
# row whose sum of squares is not finite stores NaN so the finite check fires.
@kernel function _ka_rmsnorm_rows_kernel!(
    dst,
    @Const(xs),
    @Const(scales),
    eps64::Float64,
    d::Int,
    lead::Int,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float64 (_KA_MAX_WG,)
    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    acc = 0.0
    j = lane
    while j <= d
        sq::Float32 = abs2(xs[g+(j-1)*lead])
        acc += Float64(sq)
        j += wg
    end
    part[lane] = acc
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for k in 1:steps
        lane = @index(Local, Linear)
        h = wg >> k
        if lane <= h
            part[lane] += part[lane+h]
        end
        @synchronize
    end
    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    total = part[1]
    invalid = !isfinite(total)
    rms = sqrt(total / Float64(d) + eps64)
    j = lane
    while j <= d
        idx = g + (j - 1) * lead
        v = Float32((Float64(xs[idx]) / rms) * Float64(scales[j]))
        dst[idx] = ifelse(invalid, NaN32, v)
        j += wg
    end
end

function _ka_rmsnorm_rows!(ka, dst, xs, scales, eps::Real; wg::Int=64)
    _ka_check_wg(wg)
    d = size(xs, ndims(xs))
    lead = div(length(xs), d)
    _ka_rmsnorm_rows_kernel!(ka, wg)(
        dst,
        xs,
        scales,
        Float64(eps),
        d,
        lead;
        ndrange=wg * lead,
    )
    return dst
end
