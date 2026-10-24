# on_hidden on reference_generate: the callback sees the final-normed row the
# head reads, once per argmax step, and the ids do not change. Covered on the
# dense toy, the hybrid (GDN) toy, and a dense toy with a final RMSNorm, which
# the stock toy does not carry.

const _OH_PROMPT = [1, 3, 4, 5]
const _OH_NEW = 5
const _OH_EOS = 2   # the toy contract's EOS, the default eos_ids

# the stock toy has no final RMSNorm; a fixed non-trivial scale stands in for one
function _oh_with_final_norm(ts)
    dim = ts.model.embedding.dim
    scale = 1.0 .+ 0.25 .* sin.(1.0:dim)
    return merge(ts, (; final_rms=Gesso.FrozenParameter(; shape=(dim,), storage=scale)))
end

function _oh_check(ts)
    P = length(_OH_PROMPT)
    plain = reference_generate(ts.model, ts, _OH_PROMPT; max_new_tokens=_OH_NEW)
    calls = Tuple{Int,Vector{Float32}}[]
    ids = reference_generate(ts.model, ts, _OH_PROMPT; max_new_tokens=_OH_NEW,
        on_hidden=(pos, v) -> push!(calls, (pos, v)))
    # 1. the callback changes nothing about the output
    @test ids == plain
    # 2. one call per produced token: the loop runs the head once per new id
    #    (EOS included) and breaks without another head call
    n = length(ids) - P
    @test n == _OH_NEW || ids[end] == _OH_EOS
    @test length(calls) == n
    @test [pos for (pos, _) in calls] == collect(P:(P + n - 1))
    # 3. the head reads v: the argmax of lm_head * v is the next id
    #    (lm_head is tied to the embedding: storage is (vocab, dim))
    W = ts.lm_head.storage
    @test size(W, 2) == ts.model.embedding.dim
    for (pos, v) in calls
        @test length(v) == ts.model.embedding.dim
        @test argmax(W * v) - 1 == ids[pos + 1]
        # 4. v is the final-normed state at pos: the prefill oracle's last column
        #    for the sequence ending at that input
        ref = reference_prefill(ts.model, ts, ids[1:pos]; hidden=true)[:, end]
        @test isapprox(v, ref; rtol=1e-4, atol=1e-6)
    end
    # every call hands out its own vector
    @test length(unique(objectid.([v for (_, v) in calls]))) == length(calls)
end

@testset "on_hidden: dense toy" begin
    _oh_check(toy2_tensors())
end

@testset "on_hidden: hybrid toy" begin
    _oh_check(_hybrid_toy_tensors(toy2_tensors()))
end

@testset "on_hidden: dense toy with a final RMSNorm" begin
    ts = _oh_with_final_norm(toy2_tensors())
    @test haskey(ts, :final_rms)
    _oh_check(ts)
end

@testset "on_hidden: exclusive with a speculative draft" begin
    ts = toy2_tensors()
    @test_throws ArgumentError reference_generate(ts.model, ts, _OH_PROMPT;
        max_new_tokens=_OH_NEW, draft=SuffixSource(), on_hidden=(pos, v) -> nothing)
end
