# Exact finite Float32 semantics, including cases that a scale tolerance hides.
# Included by test_lava_hybrid.jl only after usable-device admission.
@testset "Lava shared gate preserves Float32 operation boundaries" begin
    bitsequal(a, b) =
        size(a) == size(b) && reinterpret(UInt32, vec(a)) == reinterpret(UInt32, vec(b))
    function check_case(x, g, y0, shared)
        xd = Lava.LavaArray(x)
        gd = Lava.LavaArray(g)
        sd = Lava.LavaArray(shared)
        expected = copy(y0)
        _lh_inf._shared_gate_add!(Gesso.CPUBackend(), expected, x, shared, g)
        for _ in 1:3
            output = Lava.LavaArray(y0)
            @test _lh_inf._shared_gate_add!(_lh_lava, output, xd, sd, gd) === output
            @test bitsequal(Array(output), expected)
        end
        output = Lava.LavaArray(y0)
        carried = copy(y0)
        for _ in 1:3
            _lh_inf._shared_gate_add!(Gesso.CPUBackend(), carried, x, shared, g)
            _lh_inf._shared_gate_add!(_lh_lava, output, xd, sd, gd)
            @test bitsequal(Array(output), carried)
        end
    end
    rng = MersenneTwister(20261019)
    for (seq, h) in (
        (1, 1),
        (3, 63),
        (3, 64),
        (3, 65),
        (2, 127),
        (2, 129),
        (3, 2048),
        (2, 2049),
        (0, 64),
        (1, 0),
        (4, 257),
    )
        check_case(
            randn(rng, Float32, seq, h),
            randn(rng, Float32, h),
            randn(rng, Float32, seq, h),
            randn(rng, Float32, seq, h),
        )
    end
    for xv in (
        Float32[1.0e10, 1, -1.0e10, 1],
        Float32[100],
        Float32[-100],
        Float32[floatmax(Float32), floatmax(Float32)],
        Float32[-floatmax(Float32), -floatmax(Float32)],
    )
        h = length(xv)
        check_case(
            reshape(xv, 1, h),
            ones(Float32, h),
            fill(0.25f0, 1, h),
            fill(0.5f0, 1, h),
        )
    end
    for a in Float32[
        -87,
        -87.3,
        -87.33655,
        -87.5,
        -88,
        -88.5,
        -88.7,
        -88.72,
        -88.72283,
        -88.72284,
        -88.8,
        -89,
        -90,
        -0.0,
        0.0,
        87,
    ]
        check_case(
            reshape([a], 1, 1),
            ones(Float32, 1),
            zeros(Float32, 1, 1),
            ones(Float32, 1, 1),
        )
    end
    minsub = reinterpret(Float32, UInt32(1))
    sdvalues =
        reinterpret(
            Float32,
            UInt32[
                1,
                0x80000001,
                0x007fffff,
                0x00800000,
                0x00800001,
                0x00ffffff,
                0x80000000,
                0,
            ],
        ) |> collect
    append!(sdvalues, Float32[1.2345678, -1.2345678, floatmax(Float32), -floatmax(Float32)])
    for a in Float32[0, 1, 87], sdval in sdvalues
        sigmoid = 1.0f0 / (1.0f0 + exp(-a))
        term = sigmoid * sdval
        for yval in Float32[0.0f0, -0.0f0, -term, minsub]
            check_case(
                reshape([a], 1, 1),
                ones(Float32, 1),
                reshape([yval], 1, 1),
                reshape([sdval], 1, 1),
            )
        end
    end
    dotcases = (
        (Float32[minsub], Float32[floatmax(Float32)]),
        (Float32[-minsub], Float32[floatmax(Float32)]),
        (Float32[floatmax(Float32)], Float32[minsub]),
        (Float32[floatmax(Float32)], Float32[-minsub]),
        (Float32[2.0^-140], Float32[2.0^127]),
        (Float32[1, -1, minsub], Float32[1, 1, floatmax(Float32)]),
        (Float32[minsub, 1, -1], Float32[floatmax(Float32), 1, 1]),
        (Float32[2.0^-140, 2.0^-140], Float32[2.0^127, -2.0^127]),
    )
    for (xv, gv) in dotcases
        h = length(xv)
        check_case(reshape(xv, 1, h), gv, zeros(Float32, 1, h), ones(Float32, 1, h))
    end
    good = Lava.LavaArray(zeros(Float32, 1, 1))
    gate = Lava.LavaArray(ones(Float32, 1))
    @test_throws Gesso.GessoError _lh_inf._shared_gate_add!(
        _lh_lava,
        good,
        zeros(Float32, 1, 1),
        good,
        gate,
    )
    @test_throws Gesso.GessoError _lh_inf._shared_gate_add!(
        _lh_lava,
        good,
        good,
        good,
        Lava.LavaArray(ones(Float32, 2)),
    )
end
