# Public imported-policy RoPE: exact CPU rounding and partition independence.
using Test, Random

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — imported-policy RoPE regression skipped")
else
    function _rope_ref_bits(a, b)
        size(a) == size(b) && reinterpret(UInt32, vec(a)) == reinterpret(UInt32, vec(b))
    end
    _rope_ref_act(x) = Gesso.Activation(; shape=size(x), storage=x)

    @testset "Imported Float64 RoPE partition contract" begin
        cases = (
            (
                positions=collect(0:63),
                parts=[1, 30, 1, 1, 30, 1],
                d=256,
                rd=64,
                nq=16,
                nk=2,
            ),
            (
                positions=[0, 1, 31, 32, 63, 1023, 32767, 65535],
                parts=[1, 3, 1, 3],
                d=72,
                rd=64,
                nq=4,
                nk=1,
            ),
            (positions=[0, 5, 37], parts=[1, 1, 1], d=12, rd=8, nq=2, nk=3),
        )
        for (ci, case) in enumerate(cases), interleaved in (false, true)
            pos = case.positions
            n, d, rd = length(pos), case.d, case.rd
            freq = [1000000.0^(-2i/rd) for i in 0:(rd÷2-1)]
            rng = MersenneTwister(1730 + ci)
            q0 = 0.25f0 .* randn(rng, Float32, n, case.nq, d)
            k0 = 0.25f0 .* randn(rng, Float32, n, case.nk, d)
            qc, kc = copy(q0), copy(k0)
            Gesso.rope!(
                Gesso.CPUBackend(),
                _rope_ref_act(qc),
                _rope_ref_act(kc),
                pos,
                Gesso.PrefillWorkload();
                inv_freq=freq,
                rotary_dim=rd,
                interleaved,
            )
            baseline = nothing
            for rep in 1:3
                qwhole = Lava.LavaArray{Float32}(q0)
                kwhole = Lava.LavaArray{Float32}(k0)
                ret = Gesso.rope!(
                    Gesso.LavaBackend(),
                    _rope_ref_act(qwhole),
                    _rope_ref_act(kwhole),
                    pos,
                    Gesso.PrefillWorkload();
                    inv_freq=freq,
                    rotary_dim=rd,
                    interleaved,
                )
                qw, kw = Array(qwhole), Array(kwhole)
                @test ret.storage === qwhole
                @test _rope_ref_bits(qw, qc)
                @test _rope_ref_bits(kw, kc)
                @test _rope_ref_bits(qw[:, :, (rd+1):end], q0[:, :, (rd+1):end])
                @test _rope_ref_bits(kw[:, :, (rd+1):end], k0[:, :, (rd+1):end])
                qp, kp = similar(q0), similar(k0)
                offset = 0
                for count in case.parts
                    rows = (offset+1):(offset+count)
                    qd = Lava.LavaArray{Float32}(copy(q0[rows, :, :]))
                    kd = Lava.LavaArray{Float32}(copy(k0[rows, :, :]))
                    Gesso.rope!(
                        Gesso.LavaBackend(),
                        _rope_ref_act(qd),
                        _rope_ref_act(kd),
                        pos[rows],
                        Gesso.PrefillWorkload();
                        inv_freq=freq,
                        rotary_dim=rd,
                        interleaved,
                    )
                    qp[rows, :, :] = Array(qd)
                    kp[rows, :, :] = Array(kd)
                    offset += count
                end
                @test _rope_ref_bits(qp, qc)
                @test _rope_ref_bits(kp, kc)
                @test _rope_ref_bits(qp, qw)
                @test _rope_ref_bits(kp, kw)
                if baseline !== nothing
                    @test _rope_ref_bits(qw, baseline[1])
                    @test _rope_ref_bits(kw, baseline[2])
                end
                baseline = (qw, kw)
            end
        end
        # Validation still precedes the imported-frequency path.
        q = Lava.LavaArray{Float32}(zeros(Float32, 1, 2, 8))
        k = Lava.LavaArray{Float32}(zeros(Float32, 1, 1, 8))
        for (positions, freq, rd) in
            (([-1], ones(4), 8), ([0], ones(3), 8), ([0], ones(4), 7))
            @test_throws Exception Gesso.rope!(
                Gesso.LavaBackend(),
                _rope_ref_act(q),
                _rope_ref_act(k),
                positions,
                Gesso.PrefillWorkload();
                inv_freq=freq,
                rotary_dim=rd,
            )
        end
        @test _rope_ref_bits(Array(q), zeros(Float32, 1, 2, 8))
        @test _rope_ref_bits(Array(k), zeros(Float32, 1, 1, 8))
    end
end
