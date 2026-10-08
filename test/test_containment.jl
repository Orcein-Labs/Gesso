struct RegimeFaultMatrix <: AbstractMatrix{Float64}
    data::Matrix{Float64}
    fail::Base.RefValue{Bool}
end
Base.size(x::RegimeFaultMatrix) = size(x.data)
Base.parent(x::RegimeFaultMatrix) = x.data
Base.getindex(x::RegimeFaultMatrix, i::Int, j::Int) =
    x.fail[] ? error("injected CPU projection failure") : x.data[i, j]
@testset "Typed malformed plans and containment" begin
    model=toy2_modelir()
    ts=toy2_tensors()
    make(; tensors=ts, m=model, eos=2, sink=Gesso.InMemorySink()) =
        Gesso.Session(m, tensors; context_length=32, eos_token_id=eos, sink)
    empty_model=Gesso.ModelIR.Model(;
        vocab_size=model.vocab_size,
        embedding=model.embedding,
        blocks=(),
    )
    bads=[
        ()->make(; m=empty_model),
        ()->make(; eos=-1),
        ()->make(; eos=model.vocab_size),
        ()->make(; tensors=merge(ts, (; blocks=()))),
        ()->make(;
            tensors=merge(
                ts,
                (; lm_head=Gesso.EmbeddingTable(; shape=(1, 1), storage=zeros(1, 1))),
            ),
        ),
    ]
    for f in bads
        err=try
            f()
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    end
    sink=Gesso.InMemorySink()
    s=make(; sink)
    prompt=[1, 3, 4]
    expected=Gesso.reference_generate(model, ts, prompt; max_new_tokens=2)
    for bad in (-1, model.vocab_size, typemax(Int))
        err=try
            Gesso.generate(s, [bad])
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
        @test !s.ready && sink.buf[end].failure===err
        @test Gesso.generate(s, prompt; max_new_tokens=2)==expected
    end
    table=ts.embedding
    dst=Gesso.Activation(;
        shape=(1, size(table.storage, 2)),
        storage=fill!(similar(table.storage, (1, size(table.storage, 2))), 0),
    )
    for wl in (Gesso.PrefillWorkload(), Gesso.DecodeWorkload()),
        id in (-1, model.vocab_size)

        err=try
            Gesso.embedding_lookup!(Gesso.CPUBackend(), dst, table, [id], wl)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    end
    for exception in (
        ErrorException("operator launch injected"),
        InterruptException(),
        OutOfMemoryError(),
    )
        err=try
            Gesso.generate(s, prompt; max_new_tokens=2, on_token=id->throw(exception))
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError &&
              err.code==(
            exception isa OutOfMemoryError ? Gesso.ERR_ALLOCATION : Gesso.ERR_RUNTIME
        )
        @test !s.ready && sink.buf[end].failure===err
        @test_throws Gesso.GessoError Gesso.decode!(s)
        @test Gesso.generate(s, prompt; max_new_tokens=2)==expected
    end
    # An actual projection fails after K/V has been written, then the same
    # model wrapper is repaired and reset/replayed through the real engine.
    trigger=Ref(true)
    bt=ts.blocks[1]
    weight=Gesso.ProjectionWeight(;
        shape=bt.wdown.shape,
        storage=RegimeFaultMatrix(bt.wdown.storage, trigger),
    )
    blocks=collect(ts.blocks)
    blocks[1]=merge(bt, (; wdown=weight))
    broken=make(; tensors=merge(ts, (; blocks=Tuple(blocks))))
    err=try
        Gesso.generate(broken, prompt; max_new_tokens=2)
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_RUNTIME
    @test occursin("projection failure", err.detail[:cause])
    @test !broken.ready && Gesso.Inference.page_count(broken.mgr)>0
    @test_throws Gesso.GessoError Gesso.decode!(broken)
    trigger[]=false
    @test Gesso.generate(broken, prompt; max_new_tokens=2)==expected
    err=try
        Gesso.generate(s, "missing tokenizer")
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    @test sink.buf[end].failure===err && !s.ready
    for f in (
        ()->Gesso.load_llama("/nonexistent/gesso-checkpoint"),
        ()->Gesso.load_safetensors("/nonexistent/gesso.safetensors"),
        ()->Gesso.load_llama_config("/nonexistent/gesso-config.json"),
        ()->Gesso.load_gpt2_tokenizer("/nonexistent/gesso-tokenizer"),
    )
        err=try
            f()
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
        @test haskey(err.detail, :cause) && occursin("nonexistent", err.detail[:cause])
    end
end
