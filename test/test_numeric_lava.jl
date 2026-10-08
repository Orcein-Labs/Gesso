if !LAVA_LOADED || !VULKAN_OK
    @test _skip("no Vulkan device — native Lava numerical engine regression unavailable")
else
    @testset "Primary Lava rejects poisoned model output" begin
        b=Gesso.LavaBackend()
        for bad in (NaN32, Inf32, -Inf32)
            tensors=Gesso.to_device(b, toy2_tensors())
            fill!(tensors.embedding.storage, bad)
            sink=Gesso.InMemorySink()
            s=Gesso.Session(
                toy2_modelir(),
                tensors;
                backend=b,
                context_length=16,
                eos_token_id=2,
                sink,
            )
            err=try
                Gesso.generate(s, [1, 3, 4]; max_new_tokens=1)
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code==Gesso.ERR_NUMERICAL_INSTABILITY
            @test length(sink.buf)==1
            @test only(sink.buf).failure===err
        end
    end
    @testset "Decode planted NaN fails at logits after one-wait" begin
        b=Gesso.LavaBackend()
        tensors=Gesso.to_device(b, toy2_tensors())
        sink=Gesso.InMemorySink()
        s=Gesso.Session(
            toy2_modelir(),
            tensors;
            backend=b,
            context_length=16,
            eos_token_id=2,
            sink,
        )
        Gesso.prefill!(s, [1, 3, 4])
        fill!(s.h, NaN32)
        err=try
            Gesso.decode!(s)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_NUMERICAL_INSTABILITY
    end

    @testset "Primary normalization rejects unrepresentable epsilon" begin
        b=Gesso.LavaBackend()
        tensors=Gesso.to_device(b, toy2_tensors())
        for eps in (1e-100, 1e100), wl in (Gesso.PrefillWorkload(), Gesso.DecodeWorkload())
            x=Gesso.Activation(; shape=(1, 4), storage=Lava.LavaArray(ones(Float32, 1, 4)))
            out=Gesso.Activation(;
                shape=(1, 4),
                storage=Lava.LavaArray(zeros(Float32, 1, 4)),
            )
            scale=Gesso.FrozenParameter(;
                shape=(4,),
                storage=Lava.LavaArray(ones(Float32, 4)),
            )
            err=try
                Gesso.rmsnorm!(b, out, x, scale, wl; eps)
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
        end
    end

end
