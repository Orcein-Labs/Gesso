# Lava hybrid-decode seams (L2): causal conv1d + SiLU with the carried conv
# state, the GDN gate params, the unfused gated delta rule with carried
# recurrent state (both head pairings, with and without L2 norm), and the
# fail-closed rule for host storage. Each device result is compared with the
# F64 CPU oracle. Named skip without a usable Vulkan device.
#
# Tolerance. The kernels run in F32 and the oracle in F64. Every output is an
# inner product of length dk (or k, or channels) computed in F32, and the
# recurrence applies that step seq times. Decay and beta are at most 1, so the
# state does not grow through the steps, and the error is linear in seq * dk.
# With dk <= 16, seq <= 8 and eps_F32 = 6e-8, the worst-case bound is about
# 8 * 16 * 6e-8 * a small constant (~10), i.e. ~8e-5 in absolute terms. The
# test uses 1e-4 times max(1, max|ref|). That is a bound on the scale, so it
# cannot pass a wrong head mapping (which gives O(1) differences).

using Test: AbstractTestSet, Broken, get_testset, record
using Random: MersenneTwister

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava hybrid seam tests skipped (§LXXXI skip law)")
else
    _lh_lava = Gesso.LavaBackend()
    _lh_inf = Gesso.Inference
    _LH_TOL = 1.0e-4

    _lh_tol(ref) = _LH_TOL * max(1.0, maximum(abs, ref; init=0.0))

    _lh_dev(a::AbstractArray) = Lava.LavaArray{Float32}(Float32.(a))

    # Device-side comparison: returns the max abs error against the F64 ref.
    function _lh_check(label, dev, ref)
        host = Float64.(Array(dev))
        err = maximum(abs, host .- ref; init=0.0)
        @test err <= _lh_tol(ref)
        println("  ", label, ": max|dev-ref| = ", err, " (tol ", _lh_tol(ref), ")")
        return err
    end

    @testset "Lava causal conv1d + SiLU with carried state" begin
        rng = MersenneTwister(0x1a1)
        channels, k = 6, 4
        w = randn(rng, Float64, channels, 1, k)
        st_ref = randn(rng, Float64, channels, k - 1)
        st_dev = _lh_dev(st_ref)
        # One step shorter than the window (state shift path), then a longer
        # chunk (state replaced from x). The state is carried across both.
        for seq in (2, 7)
            x = randn(rng, Float64, seq, channels)
            dst_ref = zeros(Float64, seq, channels)
            Gesso._cpu_causal_conv1d_silu!(dst_ref, x, w, st_ref)
            dst = _lh_dev(zeros(seq, channels))
            _lh_inf._causal_conv1d_silu!(_lh_lava, dst, _lh_dev(x), _lh_dev(w), st_dev)
            _lh_check("conv seq=$seq", dst, dst_ref)
            _lh_check("conv state after seq=$seq", st_dev, st_ref)
        end
    end

    @testset "Lava GDN gate params" begin
        rng = MersenneTwister(0x1a2)
        seq, nv = 5, 8
        a = randn(rng, Float64, seq, nv)
        b = randn(rng, Float64, seq, nv)
        A_log = randn(rng, Float64, nv)
        dt = randn(rng, Float64, nv)
        # The CPU body is inline in Inference._gdn_gate_params!: beta = sigmoid(b),
        # g = -exp(A_log) * softplus(a + dt_bias).
        beta_ref = 1 ./ (1 .+ exp.(-b))
        sp = log1p.(exp.(a .+ dt'))
        g_ref = -exp.(A_log)' .* sp
        g = _lh_dev(zeros(seq, nv))
        beta = _lh_dev(zeros(seq, nv))
        _lh_inf._gdn_gate_params!(_lh_lava, g, beta, _lh_dev(a), _lh_dev(b), _lh_dev(A_log), _lh_dev(dt))
        _lh_check("gate g", g, g_ref)
        _lh_check("gate beta", beta, beta_ref)
    end

    @testset "Lava gated delta rule with carried state" begin
        for tiled in (false, true), l2 in (false, true)
            rng = MersenneTwister(0x1a3 + 2 * tiled + l2)
            seq, nk, nv, dk, dv = 7, 2, 4, 8, 6
            q = randn(rng, Float64, seq, nk, dk)
            kk = randn(rng, Float64, seq, nk, dk)
            v = randn(rng, Float64, seq, nv, dv)
            g = -abs.(randn(rng, Float64, seq, nv)) .* 0.5
            beta = 1 ./ (1 .+ exp.(-randn(rng, Float64, seq, nv)))
            st_ref = 0.1 .* randn(rng, Float64, nv, dk, dv)
            st_dev = _lh_dev(st_ref)
            # Two chunks: the second call starts from the state the first left.
            for chunk in (1:4, 5:seq)
                qs, ks, vs = q[chunk, :, :], kk[chunk, :, :], v[chunk, :, :]
                gs, bs = g[chunk, :], beta[chunk, :]
                out_ref = zeros(Float64, length(chunk), nv, dv)
                Gesso._cpu_gated_delta_rule!(out_ref, qs, ks, vs, gs, bs, st_ref;
                    l2norm=l2, v_head_tiled=tiled)
                out = _lh_dev(zeros(length(chunk), nv, dv))
                _lh_inf._gated_delta_rule!(_lh_lava, out, _lh_dev(qs), _lh_dev(ks), _lh_dev(vs),
                    _lh_dev(gs), _lh_dev(bs), st_dev; l2norm=l2, v_head_tiled=tiled)
                _lh_check("gdn tiled=$tiled l2=$l2 chunk=$(first(chunk)):$(last(chunk)) out",
                    out, out_ref)
                _lh_check("gdn tiled=$tiled l2=$l2 chunk=$(first(chunk)):$(last(chunk)) state",
                    st_dev, st_ref)
            end
        end
    end

    @testset "Lava hybrid seams refuse host storage" begin
        q = zeros(Float32, 2, 1, 4)
        host_state = zeros(Float32, 1, 4, 4)
        @test_throws Gesso.GessoError _lh_inf._gated_delta_rule!(_lh_lava,
            _lh_dev(zeros(2, 1, 4)), _lh_dev(q), _lh_dev(q), _lh_dev(q),
            _lh_dev(zeros(2, 1)), _lh_dev(zeros(2, 1)), host_state)
        @test_throws Gesso.GessoError _lh_inf._causal_conv1d_silu!(_lh_lava,
            zeros(Float32, 2, 4), zeros(Float32, 2, 4), zeros(Float32, 4, 1, 4),
            zeros(Float32, 4, 3))
    end

    @testset "Lava rope! partial rotary and inv_freq match CPU" begin
        # rotary_dim < d: only the first rd features rotate and the tail passes
        # through. Covers the decode kernel (seq=1), the prefill kernel, the
        # scaled inv_freq table, and both pair layouts. rd == d is the
        # full-head regression case.
        cpu_b = Gesso.CPUBackend()
        pf = Gesso.PrefillWorkload()
        dc = Gesso.DecodeWorkload()
        _lh_act(a) = Gesso.Activation(; shape=size(a), storage=a)
        rng = MersenneTwister(0x1a4)
        d = 8
        cases = (
            (3, pf, 4, false, false),
            (3, pf, 4, true, false),
            (3, pf, 4, false, true),
            (1, dc, 4, false, false),
            (1, dc, 4, true, true),
            (3, pf, 8, true, false),
        )
        for (seq, wl_, rd, interleaved, scaled) in cases
            q0 = randn(rng, Float64, seq, 2, d)
            k0 = randn(rng, Float64, seq, 2, d)
            pos = seq == 1 ? [7] : collect(0:(seq - 1))
            inv = scaled ? Gesso.rope_inv_freq(Gesso.RoPEPolicy(; rotary_dim=rd, theta=10000.0), d) : nothing
            scaled && @test inv !== nothing
            cq, ck = _lh_act(copy(q0)), _lh_act(copy(k0))
            Gesso.rope!(cpu_b, cq, ck, pos, wl_;
                theta=10000.0, inv_freq=inv, interleaved, rotary_dim=rd)
            q = _lh_act(_lh_dev(q0))
            k = _lh_act(_lh_dev(k0))
            Gesso.rope!(_lh_lava, q, k, pos, wl_;
                theta=10000.0, inv_freq=inv, interleaved, rotary_dim=rd)
            tag = "rope seq=$seq rd=$rd interleaved=$interleaved scaled=$scaled"
            _lh_check("$tag q", q.storage, cq.storage)
            _lh_check("$tag k", k.storage, ck.storage)
            if rd < d
                @test Array(q.storage)[:, :, (rd + 1):d] == Float32.(q0[:, :, (rd + 1):d])
                @test Array(k.storage)[:, :, (rd + 1):d] == Float32.(k0[:, :, (rd + 1):d])
            end
        end
    end
end
