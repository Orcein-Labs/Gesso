# Lava single-row rmsnorm (ext/lava_ops.jl) against the serial kernel it
# replaced. The reference below is a host transcription of that kernel's
# arithmetic: one accumulator summing the row in order, F64 rms, F32 write.
# The workgroup reduction sums in a different order, so the comparison is a
# relative tolerance, not bit equality.

using Random: MersenneTwister

using Test: AbstractTestSet, Broken, get_testset, record
if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava rmsnorm reduction tests skipped")
else
    const _RMS_REDUCE_EXT = Base.get_extension(Gesso, :GessoLavaExt)

    function _rms_serial_reference(x::Vector{Float32}, s::Vector{Float32}, eps::Float64)
        d = length(x)
        acc = Float32(0)
        for j in 1:d
            acc += abs2(x[j])
        end
        rms = sqrt(Float64(acc) / d + eps)
        return Float32[(Float64(x[j]) / rms) * Float64(s[j]) for j in 1:d]
    end

    @testset "lava rmsnorm! workgroup reduction matches the serial kernel" begin
        for d in (64, 2048, 4096, 5120)
            rng = MersenneTwister(0x5a + d)
            xh = randn(rng, Float32, d)
            sh = rand(rng, Float32, d) .+ 0.5f0
            want = _rms_serial_reference(xh, sh, 1e-6)
            x = (storage=Lava.LavaArray{Float32}(reshape(xh, 1, d)),)
            sc = (storage=Lava.LavaArray{Float32}(sh),)
            dst = (storage=Lava.LavaArray{Float32}(zeros(Float32, 1, d)),)
            _RMS_REDUCE_EXT._lava_rmsnorm!(dst, x, sc; check_finite=false)
            got = vec(Float64.(Array(dst.storage)))
            err = maximum(abs.(got .- Float64.(want)) ./ abs.(Float64.(want)))
            println("rmsnorm d=$d max rel err vs serial kernel: $err")
            @test err <= 1e-6
        end
    end
end
