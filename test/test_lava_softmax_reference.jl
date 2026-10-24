# Public checked softmax must preserve each causal row across prefill partitions.
using Test, Random

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — softmax partition regression skipped")
else
    _softmax_ref_act(x) = Gesso.TemporaryWorkspace(; shape=size(x), storage=x)
    _softmax_ref_bits(a, b) =
        size(a)==size(b) && reinterpret(UInt32, vec(a))==reinterpret(UInt32, vec(b))
    function _softmax_ref_gpu(scores)
        input=Lava.LavaArray{Float32}(scores)
        output=Lava.LavaArray{Float32}(undef, size(scores)...)
        ret=Gesso.softmax!(
            Gesso.LavaBackend(),
            _softmax_ref_act(output),
            _softmax_ref_act(input),
            Gesso.PrefillWorkload(),
        )
        @test ret.storage===output
        @test _softmax_ref_bits(Array(input), scores)
        Array(output)
    end

    @testset "Checked Float32 softmax partition contract" begin
        geometries=[
            (n, n) for n in (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257)
        ]
        append!(geometries, [(1, 129), (31, 127), (63, 127), (64, 129), (65, 257)])
        for (index, (L, K)) in enumerate(geometries)
            scores=3.0f0 .* randn(MersenneTwister(4919+index), Float32, L, K)
            cpu=zeros(Float32, L, K)
            Gesso.softmax!(
                Gesso.CPUBackend(),
                _softmax_ref_act(cpu),
                _softmax_ref_act(copy(scores)),
                Gesso.PrefillWorkload(),
            )
            whole=_softmax_ref_gpu(scores)
            @test maximum(abs.(Float64.(whole) .- Float64.(cpu)))<=1e-6
            @test all(isfinite, whole)
            @test all(whole .>= 0.0f0)
            @test maximum(abs.(sum(Float64.(whole); dims=2) .- 1))<=1e-5
            offset=K-L
            @test all(whole[i, j]==0.0f0 for i in 1:L for j in (offset+i+1):K)
            @test _softmax_ref_bits(_softmax_ref_gpu(scores), whole)
            chunks=unique(min.(L, L<=33 ? (1, 31, 64, 128) : (31, 64, 128)))
            for chunk in chunks
                assembled=zeros(Float32, L, K)
                for firstrow in 1:chunk:L
                    lastrow=min(firstrow+chunk-1, L)
                    rows=firstrow:lastrow
                    lastkey=offset+lastrow
                    assembled[rows, 1:lastkey]=_softmax_ref_gpu(
                        copy(scores[rows, 1:lastkey]),
                    )
                end
                @test _softmax_ref_bits(assembled, whole)
            end
        end
    end
end
