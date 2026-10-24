# CPU single-token (seq = 1) decode: the preallocated F32 Gated DeltaNet mixer
# against the reference block. Random dense weights, both V-head layouts,
# N decode steps with persistent conv and recurrent state. Outputs and states
# are compared at each step, so the state is checked as well as the output.

using Test: @testset, @test
using .GessoTestHelpers: approx_eq, deterministic_rng

const _CD = Gesso.Inference

function _cpu_gdn_fixture(rng; tiled::Bool)
    dim, nk, nv, dk, dv, ksz = 32, 2, 4, 16, 16, 4
    key_dim = nk * dk
    value_dim = nv * dv
    conv_dim = 2 * key_dim + value_dim
    f32(dims...) = Array{Float32}(0.1f0 .* randn(rng, Float32, dims...))
    mx = (;
        num_key_heads=nk,
        num_value_heads=nv,
        key_head_dim=dk,
        value_head_dim=dv,
        conv_kernel_dim=ksz,
        v_head_tiled=tiled,
    )
    spec = (; mixer=mx)
    bt = (;
        attn_rms=Gesso.FrozenParameter(; shape=(dim,), storage=ones(Float32, dim)),
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
            storage=ones(Float32, dv) .+ 0.1f0 .* randn(rng, Float32, dv),
        ),
    )
    conv0 = f32(conv_dim, ksz - 1)
    rec0 = f32(nv, dk, dv)
    x0 = f32(1, dim)
    return (; spec, bt, conv0, rec0, x0, dim, conv_dim, value_dim, nv, dk, dv, ksz)
end

zeros32(dims::Tuple) = zeros(Float32, dims)

for tiled in (false, true)
    @testset "CPU seq=1 GDN decode matches reference, $(tiled ? "tiled" : "grouped") V heads" begin
        rng = deterministic_rng(tiled ? 31 : 37)
        fx = _cpu_gdn_fixture(rng; tiled)
        cpu = Gesso.CPUBackend()
        wl = Gesso.DecodeWorkload()
        sc = _CD._cpu_decode_scratch(fx.spec.mixer, fx.dim)
        @test _CD._cpu_gdn_fast_ok(
            fx.bt,
            copy(fx.conv0),
            copy(fx.rec0),
            Gesso.Activation(; shape=(1, fx.dim), storage=copy(fx.x0)),
        )

        conv_r, rec_r = copy(fx.conv0), copy(fx.rec0)
        conv_f, rec_f = copy(fx.conv0), copy(fx.rec0)
        hr = Gesso.Activation(; shape=(1, fx.dim), storage=copy(fx.x0))
        hf = Gesso.Activation(; shape=(1, fx.dim), storage=copy(fx.x0))
        normed = Gesso.Activation(; shape=(1, fx.dim), storage=zeros32((1, fx.dim)))
        sub_r = Gesso.Activation(; shape=(1, fx.dim), storage=zeros32((1, fx.dim)))
        sub_f = Gesso.Activation(; shape=(1, fx.dim), storage=zeros32((1, fx.dim)))
        N = 6
        rec_moved = false
        for step in 1:N
            _CD._hybrid_linear_block!(
                cpu,
                hr,
                normed,
                sub_r,
                fx.bt,
                fx.spec,
                wl,
                1e-6,
                zeros32;
                conv_state=conv_r,
                rec_state=rec_r,
            )
            hr.storage .+= sub_r.storage
            _CD._hybrid_linear_block!(
                cpu,
                hf,
                normed,
                sub_f,
                fx.bt,
                fx.spec,
                wl,
                1e-6,
                zeros32;
                conv_state=conv_f,
                rec_state=rec_f,
                cpu_scratch=sc,
            )
            hf.storage .+= sub_f.storage
            @test approx_eq(sub_f.storage, sub_r.storage; atol=1e-4)
            @test approx_eq(hf.storage, hr.storage; atol=1e-4)
            @test approx_eq(conv_f, conv_r; atol=1e-5)
            @test approx_eq(rec_f, rec_r; atol=1e-4)
            rec_moved = rec_moved || !approx_eq(rec_f, fx.rec0; atol=1e-6)
        end
        # the state evolved, so the comparison is not vacuous
        @test rec_moved
    end
end
