# AMDGPU host-transfer cache: dense weights are copied to the host once per
# array and never for activations; the RoPE frequency table and the Gemma 4
# cos/sin pair are uploaded once per distinct content. The parity checks run
# the cached Gemma 4 RoPE on KA's CPU backend against the CPU oracle after a
# cache hit and after the positions change.
#
# Tolerance. The neox RoPE bound is 1e-5 * max(1, max|ref|), as in
# test_amdgpu_gemma4.jl. Nothing is loosened here.

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

# A dense array that is not an Array, so the cache path is taken; it counts
# how many times the weight is copied to the host.
mutable struct _HostCountArr{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    copies::Int
end
Base.size(a::_HostCountArr) = size(a.data)
Base.IndexStyle(::Type{<:_HostCountArr}) = IndexLinear()
Base.getindex(a::_HostCountArr, i::Int) = a.data[i]
Base.Array(a::_HostCountArr) = (a.copies += 1; copy(a.data))

_host_count_arr(x) = _HostCountArr(copy(x), 0)

if !AMDGPU_LOADED || AMDGPU_EXT === nothing
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU host cache tests skipped")
else
    const _AMDH_B = _AMDX.KADeviceBackend(_AMDKA.CPU())
    const _AMDH_CPU = Gesso.CPUBackend()
    _host_err(a, b) = maximum(abs, Float64.(a) .- Float64.(b))

    @testset "amdgpu host: a dense weight is copied to the host once per array" begin
        w = _host_count_arr(Float32[1, 2, 3, 4])
        h1 = _AMDX._ka_host_weight(w)
        h2 = _AMDX._ka_host_weight(w)
        @test h1 isa Array{Float32}
        @test h1 === h2
        @test w.copies == 1
    end

    @testset "amdgpu host: interleaved weights each keep their own copy" begin
        wa = _host_count_arr(Float32[1, 2])
        wb = _host_count_arr(Float32[3, 4])
        for _ in 1:3
            _AMDX._ka_host_weight(wa)
            _AMDX._ka_host_weight(wb)
        end
        @test wa.copies == 1
        @test wb.copies == 1
    end

    @testset "amdgpu host: an activation is never cached" begin
        x = _host_count_arr(Float32[5, 6])
        _AMDX._ka_host_storage(x)
        _AMDX._ka_host_storage(x)
        @test x.copies == 2
    end

    @testset "amdgpu host: a dropped weight's entry is pruned" begin
        function _make_and_cache()
            w = _host_count_arr(Float32[7, 8])
            _AMDX._ka_host_weight(w)
            return nothing
        end
        _make_and_cache()
        GC.gc(true)
        GC.gc(true)
        other = _host_count_arr(Float32[9])
        _AMDX._ka_host_weight(other)
        @test all(p -> p.second[1].value !== nothing, _AMDX._KA_HOST_DENSE)
    end

    @testset "amdgpu host: the RoPE frequency table is uploaded once per content" begin
        ka = _AMDH_B.ka
        f = Float64[1.0, 0.5, 0.25, 0.125]
        d1 = _AMDX._ka_rope_freq_device(ka, f)
        d2 = _AMDX._ka_rope_freq_device(ka, copy(f))
        @test d1 === d2
        d3 = _AMDX._ka_rope_freq_device(ka, Float64[1.0, 0.5, 0.25, 0.0625])
        @test d3 !== d1
        @test Array(d3) == Float64[1.0, 0.5, 0.25, 0.0625]
    end

    @testset "amdgpu host: Gemma 4 neox RoPE stays on the CPU oracle after a cache hit and a position change" begin
        inv_freq = [1.0 / 10000.0^(2i / 8) for i in 0:3]
        nheads, hd, n_rot, N = 3, 8, 8, 5
        p1 = collect(0:N-1) .+ 7
        p2 = collect(0:N-1) .+ 40
        A = randn(Float32, N, nheads * hd)
        for positions in (p1, p1, p2)
            ref = copy(A)
            Gesso.Inference._apply_rope_neox!(_AMDH_CPU, ref, positions, nheads, hd, n_rot, inv_freq[1:n_rot ÷ 2])
            dev = copy(A)
            Gesso.Inference._apply_rope_neox!(_AMDH_B, dev, positions, nheads, hd, n_rot, inv_freq[1:n_rot ÷ 2])
            @test _host_err(dev, ref) <= 1.0e-5 * max(1.0, maximum(abs, ref))
        end
    end
end
