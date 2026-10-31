# AMDGPU hybrid seams (GDN conv, gates, recurrence, gated norm, fused decode,
# sigmoid and SiLU gates, shared-expert gate) on KA's CPU backend vs the core
# CPU F64 oracle, plus device-free AMD compile of every new kernel. GPU-UNTESTED:
# parity here proves the kernel bodies and the host wrappers; device behaviour
# is not measured. Requires test_amdgpu_seam.jl to have run first (it defines
# _AMDX and _AMDKA); tolerance rule copied from test_lava_hybrid.jl.

const _HYB_TOL = 1.0e-4
_hyb_tol(ref) = _HYB_TOL * max(1, maximum(abs, ref))
function _hyb_check(tag, got, ref)
    err = maximum(abs, Float64.(got) .- Float64.(ref))
    println("    [amdgpu hybrid] $tag max|err|=$(err) tol=$(_hyb_tol(ref))")
    @test err <= _hyb_tol(ref)
    return err
end
_hyb_arr(T, dims...; s=0.0) = reshape(T.(0.8 .* sin.(collect(1:prod(dims)) .* 0.7 .+ 0.3 .+ s)), dims)
_hyb_pos(T, dims...; s=0.0) = reshape(T.(0.5 .+ 0.3 .* sin.(collect(1:prod(dims)) .* 0.9 .+ s)), dims)
_hyb_sig(x) = 1 / (1 + exp(-x))
_hyb_silu(x) = x * _hyb_sig(x)
_hyb_softplus(x) = x > 20 ? x : log1p(exp(x))

if !(@isdefined(AMDGPU_EXT)) || AMDGPU_EXT === nothing
    @test _skip("AMDGPU hybrid tests skipped — the AMDGPU extension did not load")
else
    bk = _AMDX.KADeviceBackend(_AMDKA.CPU())

    @testset "amdgpu hybrid: causal_conv1d + SiLU and state carry (odd sizes, seq < k-1)" begin
        for (seq, ch, k) in ((5, 7, 4), (1, 3, 4), (2, 3, 4), (3, 5, 2), (1, 1, 1))
            x = _hyb_arr(Float32, seq, ch)
            w = _hyb_arr(Float32, ch, 1, k; s=1.0)
            st = _hyb_arr(Float32, ch, k - 1; s=2.0)
            ref = zeros(Float64, seq, ch)
            st_ref = Float64.(st)
            Gesso._cpu_causal_conv1d_silu!(ref, Float64.(x), Float64.(w), st_ref)
            dst = zeros(Float32, seq, ch)
            st_dev = copy(st)
            Gesso.Inference._causal_conv1d_silu!(bk, dst, x, w, st_dev)
            _hyb_check("conv seq=$seq ch=$ch k=$k", dst, ref)
            _hyb_check("conv state seq=$seq ch=$ch k=$k", st_dev, st_ref)
        end
    end

    @testset "amdgpu hybrid: gate params beta and g (odd sizes, size 1)" begin
        for (seq, nv) in ((3, 5), (1, 1))
            a = _hyb_arr(Float32, seq, nv; s=0.4)
            b = _hyb_arr(Float32, seq, nv; s=1.1)
            A = _hyb_arr(Float32, nv; s=2.2)
            dt = _hyb_arr(Float32, nv; s=3.3)
            g_ref = [-exp(Float64(A[h])) * _hyb_softplus(Float64(a[t, h]) + Float64(dt[h])) for t in 1:seq, h in 1:nv]
            beta_ref = [_hyb_sig(Float64(b[t, h])) for t in 1:seq, h in 1:nv]
            g = zeros(Float32, seq, nv)
            beta = zeros(Float32, seq, nv)
            Gesso.Inference._gdn_gate_params!(bk, g, beta, a, b, A, dt)
            _hyb_check("gate g seq=$seq nv=$nv", g, g_ref)
            _hyb_check("gate beta seq=$seq nv=$nv", beta, beta_ref)
        end
    end

    @testset "amdgpu hybrid: gated delta rule prefill, both head maps (odd dv, seq 1 and 3)" begin
        nk, nv, dk, dv = 2, 4, 3, 5
        for tiled in (false, true), seq in (1, 3)
            q = _hyb_arr(Float32, seq, nk, dk; s=0.1)
            k = _hyb_arr(Float32, seq, nk, dk; s=0.2)
            v = _hyb_arr(Float32, seq, nv, dv; s=0.3)
            g = -_hyb_pos(Float32, seq, nv; s=0.4) .* 0.3f0
            beta = _hyb_pos(Float32, seq, nv; s=0.5)
            st = _hyb_arr(Float32, nv, dk, dv; s=0.6)
            out_ref = zeros(Float64, seq, nv, dv)
            st_ref = Float64.(st)
            Gesso._cpu_gated_delta_rule!(
                out_ref, Float64.(q), Float64.(k), Float64.(v), Float64.(g), Float64.(beta),
                st_ref; l2norm=true, v_head_tiled=tiled,
            )
            out = zeros(Float32, seq, nv, dv)
            st_dev = copy(st)
            Gesso.Inference._gated_delta_rule!(
                bk, out, q, k, v, g, beta, st_dev; l2norm=true, v_head_tiled=tiled,
            )
            _hyb_check("gdn out tiled=$tiled seq=$seq", out, out_ref)
            _hyb_check("gdn state tiled=$tiled seq=$seq", st_dev, st_ref)
        end
    end

    @testset "amdgpu hybrid: gated RMSNorm merge, prefill (odd dv)" begin
        seq, nv, dv = 3, 4, 5
        core = _hyb_arr(Float32, seq, nv, dv; s=0.7)
        z = _hyb_arr(Float32, seq, nv * dv; s=0.8)
        nw = _hyb_pos(Float32, dv; s=0.9)
        eps = 1.0e-6
        ref = zeros(Float64, seq, nv * dv)
        for t in 1:seq, hh in 1:nv
            r = sqrt(sum(abs2, Float64.(core[t, hh, :])) / dv + eps)
            for j in 1:dv
                ref[t, (hh - 1) * dv + j] = core[t, hh, j] / r * nw[j] * _hyb_silu(Float64(z[t, (hh - 1) * dv + j]))
            end
        end
        merged = zeros(Float32, seq, nv * dv)
        @test Gesso.Inference._gdn_gnorm_merge!(bk, merged, core, z, (; storage=nw), eps) === true
        _hyb_check("gnorm prefill", merged, ref)
    end

    @testset "amdgpu hybrid: sigmoid and SiLU gates, shared-expert gate (odd lengths, strided rows)" begin
        for n in (7, 1)
            src = _hyb_arr(Float32, n; s=0.2)
            gt = _hyb_arr(Float32, n; s=0.4)
            d1 = zeros(Float32, n)
            Gesso.Inference._sigmoid_mul!(bk, d1, src, gt)
            _hyb_check("sigmoid_mul n=$n", d1, [Float64(src[i]) * _hyb_sig(Float64(gt[i])) for i in 1:n])
            d2 = zeros(Float32, n)
            Gesso.Inference._silu_mul!(bk, d2, src, gt)
            _hyb_check("silu_mul n=$n", d2, [Float64(src[i]) * _hyb_silu(Float64(gt[i])) for i in 1:n])
        end
        for (seq, h) in ((3, 70), (1, 1))
            x = _hyb_arr(Float32, seq, h; s=0.5)
            gvec = _hyb_arr(Float32, h; s=0.6) .* 0.2f0
            sd = _hyb_arr(Float32, seq, h; s=0.7)
            y = _hyb_arr(Float32, seq, h; s=0.9)
            ref = Float64.(y)
            for t in 1:seq
                acc = sum(Float64(x[t, i]) * Float64(gvec[i]) for i in 1:h)
                s = _hyb_sig(acc)
                for i in 1:h
                    ref[t, i] += s * Float64(sd[t, i])
                end
            end
            Gesso.Inference._shared_gate_add!(bk, y, x, sd, gvec)
            _hyb_check("shared_gate seq=$seq h=$h", y, ref)
        end
    end

    @testset "amdgpu hybrid: fused single-token decode core (both head maps, 3 steps)" begin
        for (nk, nv, dk, dv) in ((2, 4, 24, 100), (2, 6, 40, 70)), tiled in (false, true)
            ksz, eps = 4, 1.0e-6
            key_dim = nk * dk
            conv_dim = 2 * key_dim + nv * dv
            spec = (; mixer=(; num_key_heads=nk, num_value_heads=nv, key_head_dim=dk,
                value_head_dim=dv, conv_kernel_dim=ksz, v_head_tiled=tiled))
            w = _hyb_arr(Float32, conv_dim, 1, ksz; s=1.0)
            A = _hyb_arr(Float32, nv; s=1.2)
            dt = _hyb_arr(Float32, nv; s=1.4)
            nw = _hyb_pos(Float32, dv; s=1.6)
            bt = (; wgdn_conv=(; storage=w), wgdn_A_log=(; storage=A),
                wgdn_dt_bias=(; storage=dt), wgdn_norm=(; storage=nw))
            cst = _hyb_arr(Float32, conv_dim, ksz - 1; s=2.0)
            rst = _hyb_arr(Float32, nv, dk, dv; s=2.1)
            cst_ref = Float64.(cst)
            rst_ref = Float64.(rst)
            cst_dev = copy(cst)
            rst_dev = copy(rst)
            for step in 1:3
                mixed = _hyb_arr(Float32, 1, conv_dim; s=3.0 + step)
                z = _hyb_arr(Float32, 1, nv * dv; s=4.0 + step)
                a = _hyb_arr(Float32, 1, nv; s=5.0 + step)
                b = _hyb_arr(Float32, 1, nv; s=6.0 + step)
                conv_ref = zeros(Float64, 1, conv_dim)
                Gesso._cpu_causal_conv1d_silu!(conv_ref, Float64.(mixed), Float64.(w), cst_ref)
                q3 = zeros(Float64, 1, nk, dk)
                k3 = zeros(Float64, 1, nk, dk)
                for h in 1:nk, i in 1:dk
                    q3[1, h, i] = conv_ref[1, (h - 1) * dk + i]
                    k3[1, h, i] = conv_ref[1, key_dim + (h - 1) * dk + i]
                end
                v3 = zeros(Float64, 1, nv, dv)
                for hh in 1:nv, j in 1:dv
                    v3[1, hh, j] = conv_ref[1, 2 * key_dim + (hh - 1) * dv + j]
                end
                beta = [_hyb_sig(Float64(b[1, h])) for h in 1:nv]
                g = [-exp(Float64(A[h])) * _hyb_softplus(Float64(a[1, h]) + Float64(dt[h])) for h in 1:nv]
                core3 = zeros(Float64, 1, nv, dv)
                Gesso._cpu_gated_delta_rule!(
                    core3, q3, k3, v3, reshape(g, 1, nv), reshape(beta, 1, nv), rst_ref;
                    l2norm=true, v_head_tiled=tiled,
                )
                merged_ref = zeros(Float64, 1, nv * dv)
                for hh in 1:nv
                    r = sqrt(sum(abs2, core3[1, hh, :]) / dv + eps)
                    for j in 1:dv
                        merged_ref[1, (hh - 1) * dv + j] =
                            core3[1, hh, j] / r * Float64(nw[j]) * _hyb_silu(Float64(z[1, (hh - 1) * dv + j]))
                    end
                end
                s = (;
                    mixed=copy(mixed), conv=zeros(Float32, 1, conv_dim), z=copy(z),
                    a=copy(a), b=copy(b), core=zeros(Float32, 1, nv * dv),
                    merged=zeros(Float32, 1, nv * dv),
                )
                Gesso.Inference._gdn_decode_core!(bk, s, bt, spec, eps, cst_dev, rst_dev)
                tag = "decode nk=$nk nv=$nv dk=$dk dv=$dv tiled=$tiled step=$step"
                _hyb_check("$tag conv", s.conv, conv_ref)
                _hyb_check("$tag conv state", cst_dev, cst_ref)
                core_ref = zeros(Float64, 1, nv * dv)
                for hh in 1:nv, j in 1:dv
                    core_ref[1, (hh - 1) * dv + j] = core3[1, hh, j]
                end
                _hyb_check("$tag core", s.core, core_ref)
                _hyb_check("$tag rec state", rst_dev, rst_ref)
                _hyb_check("$tag merged", s.merged, merged_ref)
            end
        end
    end

    @testset "amdgpu hybrid: invalid workgroup sizes refused with ERR_INVALID_PLAN" begin
        ka = bk.ka
        for bad in (0, 3, 512)
            err = _ops_error(() -> _AMDX._ka_gdn_gnorm!(
                ka, zeros(Float32, 1, 4), zeros(Float32, 1, 2, 2), zeros(Float32, 1, 4),
                zeros(Float32, 2), 1.0e-6; wg=bad,
            ))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
            err = _ops_error(() -> _AMDX._ka_causal_conv1d_silu!(
                ka, zeros(Float32, 2, 2), zeros(Float32, 2, 2), zeros(Float32, 2, 1, 2),
                zeros(Float32, 2, 1); wg=bad,
            ))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
            err = _ops_error(() -> _AMDX._ka_shared_gate!(
                ka, zeros(Float32, 1, 2), zeros(Float32, 1, 2), zeros(Float32, 1, 2),
                zeros(Float32, 2); wg=bad,
            ))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
            err = _ops_error(() -> _AMDX._ka_gdn_delta!(
                ka, zeros(Float32, 1, 4), zeros(Float32, 1, 8), zeros(Float32, 1, 4),
                zeros(Float32, 1, 4), zeros(Float32, 4), zeros(Float32, 4),
                zeros(Float32, 4, 2, 2), 4, 2, 2, 2, false, 1.0e-6; wg=bad,
            ))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end
        err = _ops_error(() -> _AMDX._ka_gdn_delta!(
            ka, zeros(Float32, 1, 10), zeros(Float32, 1, 20), zeros(Float32, 1, 5),
            zeros(Float32, 1, 5), zeros(Float32, 5), zeros(Float32, 5),
            zeros(Float32, 5, 2, 2), 5, 1, 2, 2, false, 1.0e-6; wg=4,
        ))
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu kernels: compile and link hybrid kernels for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        for wg in (64, 256)
            ndr = (wg * 7,)
            cases = (
                (_AMDX._ka_gdn_conv_silu_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 3), G(Float32, 2), Int, Int, Int)),
                (_AMDX._ka_gdn_conv_state_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), Int, Int, Int, Int)),
                (_AMDX._ka_gdn_conv_decode_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 3), Int, Int)),
                (_AMDX._ka_gdn_gates_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 1), G(Float32, 1), Int, Int)),
                (_AMDX._ka_gdn_qk_kernel!, (G(Float32, 3), G(Float32, 3), G(Float32, 3), G(Float32, 3), Int, Int, Int, Bool, Int, Float64, Float64, Bool, Int)),
                (_AMDX._ka_gdn_recur_kernel!, (G(Float32, 3), G(Float32, 3), G(Float32, 3), G(Float32, 3), G(Float32, 3), G(Float32, 2), G(Float32, 2), Int, Int, Int, Int, Int)),
                (_AMDX._ka_gdn_gnorm_kernel!, (G(Float32, 2), G(Float32, 3), G(Float32, 2), G(Float32, 1), Int, Int, Float64)),
                (_AMDX._ka_gdn_gnorm_decode_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 1), Int, Float64)),
                (_AMDX._ka_gdn_delta_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 1), G(Float32, 1), G(Float32, 3), Int, Int, Int, Bool, Int, Int, Int, Float64, Float64)),
                (_AMDX._ka_gate_sigmoid_mul_kernel!, (G(Float32, 1), G(Float32, 1), G(Float32, 1), Int)),
                (_AMDX._ka_gate_silu_mul_kernel!, (G(Float32, 1), G(Float32, 1), G(Float32, 1), Int)),
                (_AMDX._ka_shared_gate_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 1), Int)),
            )
            for (kernel, argtypes) in cases
                res = amd_compile(kernel, wg, ndr, argtypes)
                @test length(res) == length(AMD_TARGETS)
                @test all(r -> r.bytes > 0, res)
            end
        end
    end
end
