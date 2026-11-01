# The decode loop takes the correction after a verify window from the window's
# own head instead of running the head again. That is only exact if every
# window row's logits are the bits `_last_logits` gives for the same hidden row,
# softcap included. Included after test_verify_window.jl, which provides the toys.

const _GIH = Gesso.Inference

_bits_vh(a::AbstractArray{Float64}) = reinterpret(UInt64, vec(collect(a)))
_bits_vh(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(collect(a)))

# The same spec with a final logit softcap. A tiny cap saturates tanh, so
# distinct logits collapse to ties and an argmax taken before the cap would
# pick a different id than decode does.
_vh_with_softcap(spec::NamedTuple, cap) = merge(
    spec,
    (features=unique(vcat(collect(get(spec, :features, Symbol[])), :logit_softcap)), final_logit_softcap=Float64(cap)),
)
function _vh_with_softcap(spec, cap)
    S = typeof(spec)
    feats = spec.features
    vals = map(fieldnames(S)) do f
        f === :features ? convert(fieldtype(S, f), unique(vcat(collect(feats), :logit_softcap))) :
        f === :final_logit_softcap ? Float64(cap) : getfield(spec, f)
    end
    return S(vals...)
end

function _vh_check(ts, prompt, toks)
    model = ts.model
    vocab = size(ts.embedding.storage, 1)
    st = GenerateState(_VW_CTX)
    try
        reference_generate(model, ts, prompt; max_new_tokens=0, state=st)
        cpu = Gesso.CPUBackend()
        wd = _GIH.DecodeWorkload()
        n_heads = model.blocks[1].attention.n_heads
        n_kv_heads = model.blocks[1].attention.n_kv_heads
        d_head = _GIH._tensor_head_dim(model, ts)
        theta = _GIH._resolved_rope_theta(ts, nothing)
        win = _GIH._decode_window!(
            model,
            ts,
            st.kc,
            st.vc,
            toks,
            length(prompt) + 1,
            cpu,
            wd,
            st.gdn_conv,
            st.gdn_rec,
            st.cpu_scratch;
            vocab,
            n_heads,
            n_kv_heads,
            d_head,
            group=div(n_heads, n_kv_heads),
            eps=1e-6,
            theta,
            inv_freq=_GIH.tensors_rope_inv_freq(ts, d_head; theta),
            rotary_dim=_GIH._tensor_rotary_dim(ts),
        )
        ties = 0
        dim = model.embedding.dim
        for m in eachindex(toks)
            h = Gesso.Activation(; shape=(1, dim), storage=Matrix(win.hwin[m:m, :]))
            ref = _GIH._last_logits(model, ts, h, cpu, wd, vocab, 1; eps=1e-6, buf=st.cpu_scratch)
            @test _bits_vh(win.logits[m, :]) == _bits_vh(ref)
            @test win.preds[m] == _GIH._greedy_id(ref)
            ties += count(==(maximum(ref)), ref) > 1
        end
        return ties
    finally
        close(st)
    end
end

@testset "verify window: head rows equal the decode head" begin
    toks = [2, 6, 1, 7, 3]
    _vh_check(toy2_tensors(), _VW_PROMPT, toks)
    ts = _vw_hybrid_f32()
    _vh_check(ts, _VW_PROMPT, toks)
    # with a saturating softcap the rows have tied maxima, so the check above
    # would fail if the window took its argmax before the cap
    capped = merge(ts, (spec=_vh_with_softcap(ts.spec, 1.0e-3),))
    @test _vh_check(capped, _VW_PROMPT, toks) > 0
end
