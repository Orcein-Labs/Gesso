using Test
using Gesso

@testset "branch-free Float32 exp equals Base.exp (bit-exact)" begin
    cexp = Gesso._cpu_exp
    bits(v) = reinterpret(UInt32, v)
    # Vectorized context: the loop the hot paths run.
    expall!(y, x) = (@inbounds for i in eachindex(x)
        y[i] = cexp(x[i])
    end; y)
    xs = Float32[reinterpret(Float32, u) for u in UInt32(0):UInt32(1009):typemax(UInt32)]
    append!(xs, Float32[0.0f0, -0.0f0, Inf32, -Inf32, NaN32, floatmin(Float32), -floatmin(Float32)])
    M = Base.Math
    for edge in (M.MAX_EXP(Val(:ℯ), Float32), M.MIN_EXP(Val(:ℯ), Float32),
                 -M.SUBNORM_EXP(Val(:ℯ), Float32), 88.0f0, -88.0f0, 0.5f0, -0.5f0)
        x = edge
        for _ in 1:4096
            x = prevfloat(x)
        end
        for _ in 1:8192
            push!(xs, x)
            x = nextfloat(x)
        end
    end
    ys = expall!(similar(xs), xs)
    @test count(i -> bits(ys[i]) != bits(exp(xs[i])), eachindex(xs)) == 0
    @test all(i -> bits(cexp(xs[i])) == bits(exp(xs[i])), 1:997:length(xs))
    @test cexp(1.0) === exp(1.0)
end

@testset "conv window row: branch-free rows equal the per-tap loop (bit-exact)" begin
    inf = Gesso.Inference
    function ref_row!(out, mixed, conv_in, weight, t, ksz)
        @inbounds for c in eachindex(out)
            acc = 0.0f0
            for p in 1:(ksz-1)
                s = t - ksz + p
                xs = s >= 1 ? mixed[c, s] : conv_in[c, t-1+p]
                acc += weight[c, 1, p] * xs
            end
            acc += weight[c, 1, ksz] * mixed[c, t]
            out[c] = acc / (1.0f0 + exp(-acc))
        end
        return out
    end
    for ksz in (2, 3, 4), C in (37, 64, 6144)
        T = 9
        mixed = randn(Float32, C, T) .* 6
        mixed[1:3:end, :] .= -0.0f0
        conv_in = randn(Float32, C, ksz - 1)
        weight = randn(Float32, C, 1, ksz)
        weight[2:5:end, 1, :] .= 0.0f0
        a = zeros(Float32, C)
        b = zeros(Float32, C)
        for t in 1:T
            inf._gdn_conv_window_row!(a, mixed, conv_in, weight, t, ksz)
            ref_row!(b, mixed, conv_in, weight, t, ksz)
            @test reinterpret(UInt32, a) == reinterpret(UInt32, b)
        end
    end
end
