# Prefill runs the window in commit mode: each GDN block's conv and recurrent
# state are committed as soon as the block finishes, and the blocks share one
# set of record buffers. That must leave the bits a plain window followed by a
# full `_commit_window!` leaves. Included after test_verify_window.jl, which
# provides the hybrid Float32 toy.

using Random: Xoshiro

const _GIC = Gesso.Inference

function _vc_window!(model, ts, st, toks, pos; commit::Bool)
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = _GIC._tensor_head_dim(model, ts)
    theta = _GIC._resolved_rope_theta(ts, nothing)
    win = _GIC._decode_window!(
        model,
        ts,
        st.kc,
        st.vc,
        toks,
        pos,
        Gesso.CPUBackend(),
        _GIC.DecodeWorkload(),
        st.gdn_conv,
        st.gdn_rec,
        st.cpu_scratch;
        vocab=size(ts.embedding.storage, 1),
        n_heads,
        n_kv_heads,
        d_head,
        group=div(n_heads, n_kv_heads),
        eps=1e-6,
        theta,
        inv_freq=_GIC.tensors_rope_inv_freq(ts, d_head; theta),
        rotary_dim=_GIC._tensor_rotary_dim(ts),
        head=false,
        commit,
    )
    _GIC._commit_window!(win, length(toks), st.gdn_conv, st.gdn_rec, st.h)
    return win
end

function _vc_check(ts)
    model = ts.model
    vocab = size(ts.embedding.storage, 1)
    rng = Xoshiro(20261106)
    # passes above and below the window reuse limit, one after another, so the
    # second pass reads state the first committed
    for lens in ([40, 8], [8, 40], [5, 3, 33])
        A = GenerateState(_VW_CTX)
        B = GenerateState(_VW_CTX)
        try
            reference_generate(model, ts, _VW_PROMPT; max_new_tokens=0, state=A)
            reference_generate(model, ts, _VW_PROMPT; max_new_tokens=0, state=B)
            pos = length(_VW_PROMPT) + 1
            for n in lens
                toks = rand(rng, 0:(vocab-1), n)
                wa = _vc_window!(model, ts, A, toks, pos; commit=false)
                wb = _vc_window!(model, ts, B, toks, pos; commit=true)
                @test wa.hwin == wb.hwin
                append!(A.ids, toks)
                append!(B.ids, toks)
                pos += n
            end
            @test isempty(_vw_state_diff(A, B))
            # the committed state is distinct storage per block
            recs = [r for r in B.gdn_rec if r isa Array]
            @test length(unique(objectid, recs)) == length(recs)
        finally
            close(A)
            close(B)
        end
    end
    return nothing
end

_vc_ngdn(ts) = count(bt -> _GIC._block_mixer(bt) === :linear_attention, ts.blocks)

@testset "verify window: commit mode keeps the committed bits" begin
    # the hybrid toy (GDN then attention) and an all-GDN toy, where block 2's
    # state lands in the buffer block 1 displaced
    hybrid = _vw_hybrid_f32()
    allgdn = _vw_f32(_with_gdn_block(_hybrid_toy_tensors(toy2_tensors()), 2; seed=43))
    @test _vc_ngdn(hybrid) == 1
    @test _vc_ngdn(allgdn) == 2
    for ts in (hybrid, allgdn)
        _vc_check(ts)
    end
end
