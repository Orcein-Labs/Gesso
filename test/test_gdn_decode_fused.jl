# Fused single-token Gated DeltaNet decode vs the unfused block, on device.
# Random weights, both V-head layouts, N decode steps with persistent conv and
# recurrent state. Named skip without a functional NVIDIA device.

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

using .GessoTestHelpers: approx_eq, deterministic_rng

if !CUDA_LOADED || !CUDA.functional()
    _skip("no NVIDIA device — fused GDN decode test skipped (§LXXVII skip law)")
else
    cuda = Gesso.CUDABackend()
    wl = Gesso.DecodeWorkload()
    Inf_ = Gesso.Inference

    # Dimensions are small enough that every projection stays F32 on device.
    # nk < nv so the grouped and tiled pairings differ.
    function _gdn_fixture(rng; tiled::Bool)
        dim, nk, nv, dk, dv, ksz = 32, 2, 4, 16, 16, 4
        key_dim = nk * dk
        value_dim = nv * dv
        conv_dim = 2 * key_dim + value_dim
        f32(dims...) = CUDA.CuArray{Float32}(0.1f0 .* randn(rng, Float32, dims...))
        spec = (;
            mixer=(;
                num_key_heads=nk,
                num_value_heads=nv,
                key_head_dim=dk,
                value_head_dim=dv,
                conv_kernel_dim=ksz,
                v_head_tiled=tiled,
            )
        )
        bt = (;
            attn_rms=Gesso.FrozenParameter(;
                shape=(dim,),
                storage=CUDA.CuArray{Float32}(ones(Float32, dim)),
            ),
            wgdn_qkv=Gesso.ProjectionWeight(;
                shape=(conv_dim, dim),
                storage=f32(conv_dim, dim),
            ),
            wgdn_z=Gesso.ProjectionWeight(;
                shape=(value_dim, dim),
                storage=f32(value_dim, dim),
            ),
            wgdn_a=Gesso.ProjectionWeight(; shape=(nv, dim), storage=f32(nv, dim)),
            wgdn_b=Gesso.ProjectionWeight(; shape=(nv, dim), storage=f32(nv, dim)),
            wgdn_o=Gesso.ProjectionWeight(;
                shape=(dim, value_dim),
                storage=f32(dim, value_dim),
            ),
            wgdn_conv=Gesso.FrozenParameter(;
                shape=(conv_dim, 1, ksz),
                storage=f32(conv_dim, 1, ksz),
            ),
            wgdn_A_log=Gesso.FrozenParameter(; shape=(nv,), storage=f32(nv)),
            wgdn_dt_bias=Gesso.FrozenParameter(; shape=(nv,), storage=f32(nv)),
            wgdn_norm=Gesso.FrozenParameter(;
                shape=(dv,),
                storage=CUDA.CuArray{Float32}(
                    ones(Float32, dv) .+ 0.1f0 .* randn(rng, Float32, dv),
                ),
            ),
        )
        conv0 = f32(conv_dim, ksz - 1)
        rec0 = f32(nv, dk, dv)
        x0 = f32(1, dim)
        return (; spec, bt, conv0, rec0, x0, dim, conv_dim, value_dim, nv, dk, dv, ksz)
    end

    zl = dims -> CUDA.zeros(Float32, dims)

    for tiled in (false, true)
        @testset "fused GDN decode matches unfused, $(tiled ? "tiled" : "grouped") V heads" begin
            rng = deterministic_rng(tiled ? 23 : 19)
            fx = _gdn_fixture(rng; tiled)
            tensors = (; spec=fx.spec)
            scratch = Inf_._gdn_decode_scratch(zl, tensors)
            @test scratch !== nothing
            @test Inf_._gdn_fused_decode(cuda)

            conv_u, rec_u = copy(fx.conv0), copy(fx.rec0)
            conv_f, rec_f = copy(fx.conv0), copy(fx.rec0)
            hu = Gesso.Activation(; shape=(1, fx.dim), storage=copy(fx.x0))
            hf = Gesso.Activation(; shape=(1, fx.dim), storage=copy(fx.x0))
            normed = Gesso.Activation(; shape=(1, fx.dim), storage=zl((1, fx.dim)))
            sub_u = Gesso.Activation(; shape=(1, fx.dim), storage=zl((1, fx.dim)))
            sub_f = Gesso.Activation(; shape=(1, fx.dim), storage=zl((1, fx.dim)))
            N = 6
            for step in 1:N
                Inf_._hybrid_linear_block!(
                    cuda,
                    hu,
                    normed,
                    sub_u,
                    fx.bt,
                    fx.spec,
                    wl,
                    1e-6,
                    zl;
                    conv_state=conv_u,
                    rec_state=rec_u,
                    gdn_fused=false,
                )
                hu.storage .+= sub_u.storage
                Inf_._hybrid_linear_block!(
                    cuda,
                    hf,
                    normed,
                    sub_f,
                    fx.bt,
                    fx.spec,
                    wl,
                    1e-6,
                    zl;
                    conv_state=conv_f,
                    rec_state=rec_f,
                    gdn_scratch=scratch,
                    gdn_fused=true,
                )
                hf.storage .+= sub_f.storage
                @test approx_eq(Array(sub_f.storage), Array(sub_u.storage); atol=1e-3)
                @test approx_eq(Array(hf.storage), Array(hu.storage); atol=1e-3)
                @test approx_eq(Array(conv_f), Array(conv_u); atol=1e-3)
                @test approx_eq(Array(rec_f), Array(rec_u); atol=2e-3)
            end
            # the state actually evolved, so the comparison is not vacuous
            @test !approx_eq(Array(rec_f), Array(fx.rec0); atol=1e-6)
        end
    end

    @testset "fused GDN decode rejects a state of the wrong shape" begin
        rng = deterministic_rng(29)
        fx = _gdn_fixture(rng; tiled=false)
        scratch = Inf_._gdn_decode_scratch(zl, (; spec=fx.spec))
        bad_rec = CUDA.zeros(Float32, fx.nv, fx.dk, fx.dv + 1)
        hx = Gesso.Activation(; shape=(1, fx.dim), storage=copy(fx.x0))
        normed = Gesso.Activation(; shape=(1, fx.dim), storage=zl((1, fx.dim)))
        sub = Gesso.Activation(; shape=(1, fx.dim), storage=zl((1, fx.dim)))
        @test_throws ErrorException Inf_._hybrid_linear_block!(
            cuda,
            hx,
            normed,
            sub,
            fx.bt,
            fx.spec,
            wl,
            1e-6,
            zl;
            conv_state=copy(fx.conv0),
            rec_state=bad_rec,
            gdn_scratch=scratch,
            gdn_fused=true,
        )
    end
end
