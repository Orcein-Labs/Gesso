@testset "Nonfinite logits are typed failures" begin
    for values in ([NaN, 1.0], [Inf, 1.0], [-Inf, 1.0])
        err=try
            Gesso.Inference._greedy_id(values)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_NUMERICAL_INSTABILITY
    end
    @test Gesso.Inference._greedy_id([2.0, 2.0, 1.0])==0
end

@testset "Unsupported native CPU arithmetic is explicit" begin
    a(x) = Gesso.Activation(; shape=size(x), storage=x)
    w(x) = Gesso.TemporaryWorkspace(; shape=size(x), storage=x)
    p(x) = Gesso.ProjectionWeight(; shape=size(x), storage=x)
    frozen(x) = Gesso.FrozenParameter(; shape=size(x), storage=x)
    table(x) = Gesso.EmbeddingTable(; shape=size(x), storage=x)
    cpu=Gesso.CPUBackend()
    for wl in (Gesso.PrefillWorkload(), Gesso.DecodeWorkload())
        x=ones(Float16, 1, 4)
        dest=zeros(Float16, 1, 4)
        calls=(
            ()->Gesso.embedding_lookup!(cpu, a(dest), table(ones(Float16, 4, 4)), [0], wl),
            ()->Gesso.rmsnorm!(cpu, a(dest), a(x), frozen(ones(Float16, 4)), wl),
            ()->Gesso.rope!(
                cpu,
                a(reshape(x, 1, 1, 4)),
                a(reshape(copy(x), 1, 1, 4)),
                [0],
                wl,
            ),
            ()->Gesso.matmul!(cpu, a(dest), a(x), p(ones(Float16, 4, 4)), wl),
            ()->Gesso.softmax!(cpu, w(dest), w(x), wl),
            ()->Gesso.swiglu!(cpu, a(dest), a(x), a(x), wl),
        )
        for f in calls
            err=try
                f()
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
        end
    end
end
