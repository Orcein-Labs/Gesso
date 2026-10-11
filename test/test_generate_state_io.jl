# Park and resume: save_state / load_state against a live GenerateState.
#
# A loaded state must continue a conversation exactly as the live state it was
# saved from does, and every file that cannot be that state must be refused.

using .GessoTestHelpers: deterministic_rng

const _IO_PROMPT = [1, 3, 4, 5]

function _refused(f)
    try
        f()
        return false
    catch e
        return e isa Gesso.GessoError && e.code == Gesso.ERR_INVALID_PLAN
    end
end

# toy2 with block 1 turned into a Gated DeltaNet block, so the recurrent state
# and the snapshot take part in the round trip.
function _hybrid_toy_tensors(ts; seed=41)
    rng = deterministic_rng(seed)
    dim = ts.model.embedding.dim
    d_head = div(dim, ts.model.blocks[1].attention.n_heads)
    nk, nv, dk, dv, ksz = 2, 4, 2, 2, 4
    value_dim = nv * dv
    conv_dim = 2 * nk * dk + value_dim
    f(dims...) = 0.1 .* randn(rng, dims...)
    mx = (; num_key_heads=nk, num_value_heads=nv, key_head_dim=dk, value_head_dim=dv,
        conv_kernel_dim=ksz, v_head_tiled=false)
    gdn = (;
        mixer=:linear_attention,
        wgdn_qkv=Gesso.ProjectionWeight(; shape=(conv_dim, dim), storage=f(conv_dim, dim)),
        wgdn_z=Gesso.ProjectionWeight(; shape=(value_dim, dim), storage=f(value_dim, dim)),
        wgdn_a=Gesso.ProjectionWeight(; shape=(nv, dim), storage=f(nv, dim)),
        wgdn_b=Gesso.ProjectionWeight(; shape=(nv, dim), storage=f(nv, dim)),
        wgdn_o=Gesso.ProjectionWeight(; shape=(dim, value_dim), storage=f(dim, value_dim)),
        wgdn_conv=Gesso.FrozenParameter(; shape=(conv_dim, 1, ksz), storage=f(conv_dim, 1, ksz)),
        wgdn_A_log=Gesso.FrozenParameter(; shape=(nv,), storage=f(nv)),
        wgdn_dt_bias=Gesso.FrozenParameter(; shape=(nv,), storage=f(nv)),
        wgdn_norm=Gesso.FrozenParameter(; shape=(dv,), storage=ones(Float64, dv) .+ f(dv)),
    )
    blocks = (merge(ts.blocks[1], gdn), ts.blocks[2:end]...)
    return merge(ts, (; blocks, spec=(; head_dim=d_head, mixer=mx)))
end

# Turn one runs live and is parked; turn two runs on the live state, on the
# loaded state and cold. The loaded state's info shows how much it reused.
function _park_and_resume(ts, turn2::Function)
    mktempdir() do dir
        path = joinpath(dir, "state.gst")
        A = GenerateState(64)
        B = nothing
        try
            g1 = reference_generate(ts.model, ts, _IO_PROMPT; max_new_tokens=3, state=A)
            save_state(path, A)
            B = load_state(path, ts.model, ts)
            prompt2 = turn2(g1)
            infoA, infoB = Ref{Any}(nothing), Ref{Any}(nothing)
            gA = reference_generate(ts.model, ts, prompt2; max_new_tokens=4, state=A, info=infoA)
            gB = reference_generate(ts.model, ts, prompt2; max_new_tokens=4, state=B, info=infoB)
            cold = reference_generate(ts.model, ts, prompt2; max_new_tokens=4)
            return (; g1, gA, gB, cold, reusedA=infoA[].reused, reusedB=infoB[].reused,
                bytes=filesize(path))
        finally
            close(A)
            B === nothing || close(B)
        end
    end
end

@testset "save_state/load_state: attention round trip resumes without re-prefill" begin
    ts = toy2_tensors()
    r = _park_and_resume(ts, g1 -> [g1; 1])
    @test r.gB == r.gA
    @test r.gB == r.cold
    # the last generated token is never fed back, so it has no cache row yet
    @test r.reusedB == r.reusedA == length(r.g1) - 1
end

@testset "save_state/load_state: hybrid GDN round trip restores the snapshot" begin
    ts = _hybrid_toy_tensors(toy2_tensors())
    V = size(ts.embedding.storage, 1)
    # the reply diverges after the prompt, so only the snapshot can be resumed
    r = _park_and_resume(ts, g1 -> [_IO_PROMPT; mod(g1[5] + 1, V)])
    @test r.gB == r.gA
    @test r.gB == r.cold
    @test r.reusedB == r.reusedA == length(_IO_PROMPT)
end

@testset "load_state refuses what it cannot resume" begin
    ts = toy2_tensors()
    mktempdir() do dir
        path = joinpath(dir, "state.gst")
        A = GenerateState(64)
        try
            reference_generate(ts.model, ts, _IO_PROMPT; max_new_tokens=3, state=A)
            save_state(path, A)
        finally
            close(A)
        end
        bytes = read(path)

        other_wq = copy(ts.blocks[1].wq.storage)
        other_wq[1, 1] += 1.0
        other_block = merge(ts.blocks[1],
            (; wq=Gesso.ProjectionWeight(; shape=size(other_wq), storage=other_wq)))
        other = merge(ts, (; blocks=(other_block, ts.blocks[2:end]...)))
        @test _refused(() -> load_state(path, ts.model, other))

        flipped = copy(bytes)
        flipped[1] ⊻= 0x01
        flipped_path = joinpath(dir, "flipped.gst")
        write(flipped_path, flipped)
        @test _refused(() -> load_state(flipped_path, ts.model, ts))

        cut_path = joinpath(dir, "cut.gst")
        write(cut_path, bytes[1:(length(bytes) ÷ 2)])
        @test _refused(() -> load_state(cut_path, ts.model, ts))

        @test _refused() do
            withenv("GESSO_KV_FORMAT" => "f16") do
                load_state(path, ts.model, ts)
            end
        end
    end
end
