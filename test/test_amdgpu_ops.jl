# AMDGPU operator methods on KA's CPU backend vs the core CPU F64 oracle, plus
# device-free AMD compile of every new kernel. GPU-UNTESTED: parity here proves
# the kernel bodies and the host wrappers; device behaviour is not measured.
# Requires test_amdgpu_seam.jl to have run first (it defines _AMDX and _AMDKA).

_ops_act(F, arr) = F(; shape=size(arr), storage=arr)
_ops_maxerr(a, b) = maximum(abs, a .- b)
_ops_within(a, b; atol) = _ops_maxerr(a, b) <= atol
_ops_report(name, err, atol) = println("    [amdgpu parity] $name max|err|=$(err) tol=$(atol)")

function _ops_error(f)
    try
        f()
        return nothing
    catch e
        return e
    end
end

if !(@isdefined(AMDGPU_EXT)) || AMDGPU_EXT === nothing
    @test _skip("AMDGPU ops tests skipped — the AMDGPU extension did not load")
else
    @testset "amdgpu ops: embedding_lookup! (exact row copy, 0-based ids)" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        wl = Gesso.PrefillWorkload()
        tab = Float32.(reshape(1:15, 5, 3))
        toks = [4, 0, 2, 2, 1]
        dst = _ops_act(Gesso.Activation, zeros(Float32, 5, 3))
        Gesso.embedding_lookup!(b, dst, _ops_act(Gesso.EmbeddingTable, tab), toks, wl)
        @test dst.storage == tab[toks .+ 1, :]
        _ops_report("embedding_lookup!", 0.0, 0.0)
    end

    @testset "amdgpu ops: rmsnorm! vs CPU F64 oracle" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        wl = Gesso.PrefillWorkload()
        cpu = Gesso.CPUBackend()
        rng = Random.MersenneTwister(0x52a1)
        for shape in ((1, 1000), (7, 33), (2, 3, 40))
            x32 = randn(rng, Float32, shape...)
            w32 = randn(rng, Float32, shape[end])
            dst = _ops_act(Gesso.Activation, similar(x32))
            Gesso.rmsnorm!(b, dst, _ops_act(Gesso.Activation, x32), _ops_act(Gesso.FrozenParameter, w32), wl; eps=1e-6)
            ref = _ops_act(Gesso.Activation, zeros(Float64, shape...))
            Gesso.rmsnorm!(
                cpu,
                ref,
                _ops_act(Gesso.Activation, Float64.(x32)),
                _ops_act(Gesso.FrozenParameter, Float64.(w32)),
                wl;
                eps=1e-6,
            )
            err = _ops_maxerr(dst.storage, ref.storage)
            _ops_report("rmsnorm! $(shape)", err, 1e-5)
            @test err <= 1e-5
        end

        x = randn(rng, Float32, 3, 16)
        x[2, 5] = Inf32
        dst = _ops_act(Gesso.Activation, similar(x))
        ones16 = _ops_act(Gesso.FrozenParameter, ones(Float32, 16))
        err = _ops_error(() -> Gesso.rmsnorm!(b, dst, _ops_act(Gesso.Activation, x), ones16, wl; eps=1e-6))
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_NUMERICAL_INSTABILITY
        dec = _ops_act(Gesso.Activation, similar(x))
        Gesso.rmsnorm!(b, dec, _ops_act(Gesso.Activation, x), ones16, Gesso.DecodeWorkload(); eps=1e-6)
        @test all(isnan, dec.storage[2, :]) && all(isfinite, dec.storage[[1, 3], :])
    end

    @testset "amdgpu ops: rope! vs CPU F64 oracle (interleaved, half-split, partial, custom inv_freq)" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        wl = Gesso.PrefillWorkload()
        cpu = Gesso.CPUBackend()
        rng = Random.MersenneTwister(0x70e)
        cases = (
            (; name="interleaved d=8 pos 0,1,5000", seq=5, hq=3, hk=2, d=8, pos=[0, 1, 5000, 7, 2], kw=(;)),
            (; name="half-split d=6 seq=1", seq=1, hq=2, hk=1, d=6, pos=[3], kw=(; interleaved=false)),
            (;
                name="half-split partial rotary_dim=4",
                seq=4, hq=2, hk=2, d=8, pos=[0, 9, 10, 11],
                kw=(; interleaved=false, rotary_dim=4),
            ),
            (;
                name="interleaved partial rotary_dim=6 theta=500000",
                seq=3, hq=2, hk=2, d=8, pos=[1, 2, 3],
                kw=(; theta=500000.0, rotary_dim=6),
            ),
            (;
                name="custom inv_freq (llama3-style table)",
                seq=3, hq=2, hk=2, d=8, pos=[1, 2, 3],
                kw=(; inv_freq=[1.0, 0.5, 0.25, 0.125]),
            ),
        )
        for c in cases
            q32 = randn(rng, Float32, c.seq, c.hq, c.d)
            k32 = randn(rng, Float32, c.seq, c.hk, c.d)
            qd = _ops_act(Gesso.Activation, copy(q32))
            kd = _ops_act(Gesso.Activation, copy(k32))
            Gesso.rope!(b, qd, kd, c.pos, wl; c.kw...)
            q64 = _ops_act(Gesso.Activation, Float64.(q32))
            k64 = _ops_act(Gesso.Activation, Float64.(k32))
            Gesso.rope!(cpu, q64, k64, c.pos, wl; c.kw...)
            eq = _ops_maxerr(qd.storage, q64.storage)
            ek = _ops_maxerr(kd.storage, k64.storage)
            _ops_report("rope! q, $(c.name)", eq, 1e-3)
            _ops_report("rope! k, $(c.name)", ek, 1e-3)
            @test eq <= 1e-3 && ek <= 1e-3
        end

        q = _ops_act(Gesso.Activation, randn(rng, Float32, 2, 1, 4))
        k = _ops_act(Gesso.Activation, randn(rng, Float32, 2, 1, 4))
        err = _ops_error(() -> Gesso.rope!(b, q, k, [0, Int(typemax(Int32)) + 1], wl))
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu ops: softmax! vs CPU F64 oracle (causal, prefix, stability, NaN rows)" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        wl = Gesso.PrefillWorkload()
        cpu = Gesso.CPUBackend()
        rng = Random.MersenneTwister(0x50f7)
        for (L, K) in ((3, 3), (2, 5), (1, 4), (4, 4))
            s32 = 3f0 .* randn(rng, Float32, L, K)
            out = _ops_act(Gesso.TemporaryWorkspace, zeros(Float32, L, K))
            Gesso.softmax!(b, out, _ops_act(Gesso.TemporaryWorkspace, copy(s32)), wl)
            ref = _ops_act(Gesso.TemporaryWorkspace, zeros(Float64, L, K))
            Gesso.softmax!(cpu, ref, _ops_act(Gesso.TemporaryWorkspace, Float64.(s32)), wl)
            err = _ops_maxerr(out.storage, ref.storage)
            _ops_report("softmax! $(L)x$(K)", err, 1e-6)
            @test err <= 1e-6
        end

        big = fill(1.0f6, 3, 3)
        ob = _ops_act(Gesso.TemporaryWorkspace, zeros(Float32, 3, 3))
        Gesso.softmax!(b, ob, _ops_act(Gesso.TemporaryWorkspace, big), wl)
        @test all(isfinite, ob.storage)
        @test ob.storage[1, 2] == 0 && ob.storage[1, 3] == 0 && ob.storage[2, 3] == 0

        nanrow = Float32[NaN 0; 1 2]
        out = _ops_act(Gesso.TemporaryWorkspace, zeros(Float32, 2, 2))
        err = _ops_error(() -> Gesso.softmax!(b, out, _ops_act(Gesso.TemporaryWorkspace, nanrow), wl))
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_NUMERICAL_INSTABILITY
        dec = _ops_act(Gesso.TemporaryWorkspace, zeros(Float32, 2, 2))
        Gesso.softmax!(b, dec, _ops_act(Gesso.TemporaryWorkspace, nanrow), Gesso.DecodeWorkload())
        @test isnan(dec.storage[1, 1])
    end

    @testset "amdgpu ops: swiglu! vs CPU F64 oracle" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        wl = Gesso.PrefillWorkload()
        cpu = Gesso.CPUBackend()
        rng = Random.MersenneTwister(0x5719)
        g32 = 2f0 .* randn(rng, Float32, 3, 17)
        u32 = randn(rng, Float32, 3, 17)
        dst = _ops_act(Gesso.Activation, zeros(Float32, 3, 17))
        Gesso.swiglu!(b, dst, _ops_act(Gesso.Activation, g32), _ops_act(Gesso.Activation, u32), wl)
        ref = _ops_act(Gesso.Activation, zeros(Float64, 3, 17))
        Gesso.swiglu!(
            cpu,
            ref,
            _ops_act(Gesso.Activation, Float64.(g32)),
            _ops_act(Gesso.Activation, Float64.(u32)),
            wl,
        )
        err = _ops_maxerr(dst.storage, ref.storage)
        _ops_report("swiglu!", err, 1e-6)
        @test err <= 1e-6

        bad = _ops_error(() -> Gesso.swiglu!(
            b,
            _ops_act(Gesso.Activation, zeros(Float32, 2, 2)),
            _ops_act(Gesso.Activation, g32),
            _ops_act(Gesso.Activation, u32),
            wl,
        ))
        @test bad isa Gesso.GessoError && bad.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu ops: GEGLU (Gesso.Inference._geglu!) vs CPU F64 oracle" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        cpu = Gesso.CPUBackend()
        rng = Random.MersenneTwister(0x6e61)
        g32 = 3f0 .* randn(rng, Float32, 4, 33)
        u32 = randn(rng, Float32, 4, 33)
        dst = _ops_act(Gesso.Activation, zeros(Float32, 4, 33))
        Gesso.Inference._geglu!(b, dst, _ops_act(Gesso.Activation, g32), _ops_act(Gesso.Activation, u32))
        ref = _ops_act(Gesso.Activation, zeros(Float64, 4, 33))
        Gesso.Inference._geglu!(
            cpu,
            ref,
            _ops_act(Gesso.Activation, Float64.(g32)),
            _ops_act(Gesso.Activation, Float64.(u32)),
        )
        err = _ops_maxerr(dst.storage, ref.storage)
        _ops_report("geglu (_geglu!)", err, 1e-6)
        @test err <= 1e-6
    end

    @testset "amdgpu ops: matmul! vs CPU F64 oracle (projection and tied embedding head)" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        wl = Gesso.PrefillWorkload()
        cpu = Gesso.CPUBackend()
        rng = Random.MersenneTwister(0x3a7)
        x32 = randn(rng, Float32, 4, 5)
        w32 = randn(rng, Float32, 3, 5)
        e32 = randn(rng, Float32, 6, 5)
        for (w, F, outw) in (
            (w32, Gesso.ProjectionWeight, 3),
            (e32, Gesso.EmbeddingTable, 6),
        )
            dst = _ops_act(Gesso.Activation, zeros(Float32, 4, outw))
            Gesso.matmul!(b, dst, _ops_act(Gesso.Activation, x32), _ops_act(F, w), wl)
            ref = _ops_act(Gesso.Activation, zeros(Float64, 4, outw))
            Gesso.matmul!(cpu, ref, _ops_act(Gesso.Activation, Float64.(x32)), _ops_act(F, Float64.(w)), wl)
            err = _ops_maxerr(dst.storage, ref.storage)
            _ops_report("matmul! $(F)", err, 1e-6)
            @test err <= 1e-6
        end
    end

    @testset "amdgpu ops: host storage and invalid arguments stay typed" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        wl = Gesso.PrefillWorkload()
        host64 = _ops_act(Gesso.Activation, ones(Float64, 2, 2))
        ok32 = _ops_act(Gesso.Activation, ones(Float32, 2, 2))
        w = _ops_act(Gesso.FrozenParameter, ones(Float32, 2))
        for f in (
            () -> Gesso.rmsnorm!(b, ok32, host64, w, wl),
            () -> Gesso.swiglu!(b, ok32, host64, ok32, wl),
            () -> Gesso.matmul!(b, ok32, host64, _ops_act(Gesso.ProjectionWeight, ones(Float32, 2, 2)), wl),
        )
            err = _ops_error(f)
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end

        x = _ops_act(Gesso.Activation, ones(Float32, 2, 4))
        w4 = _ops_act(Gesso.FrozenParameter, ones(Float32, 4))
        for eps in (0.0, -1.0, NaN, 1e-60)
            err = _ops_error(() -> Gesso.rmsnorm!(b, _ops_act(Gesso.Activation, zeros(Float32, 2, 4)), x, w4, wl; eps))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end

        err = _ops_error(() -> Gesso.embedding_lookup!(
            b,
            _ops_act(Gesso.Activation, zeros(Float32, 1, 3)),
            _ops_act(Gesso.EmbeddingTable, zeros(Float32, 4, 3)),
            [4],
            wl,
        ))
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu kernels: compile and link new kernels for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        for wg in (64, 256)
            ndr = (wg * 7,)
            cases = (
                (_AMDX._ka_embedding_kernel!, (G(Float32, 2), G(Float32, 2), G(Int32, 1), Int)),
                (_AMDX._ka_rope_kernel!, (G(Float32, 3), G(Int32, 1), G(Float64, 1), Bool, Int, Int)),
                (_AMDX._ka_softmax_rows_kernel!, (G(Float32, 2), G(Float32, 2), Int, Int)),
                (_AMDX._ka_swiglu_kernel!, (G(Float32, 1), G(Float32, 1), G(Float32, 1))),
                (_AMDX._ka_geglu_kernel!, (G(Float32, 1), G(Float32, 1), G(Float32, 1))),
            )
            for (kernel, argtypes) in cases
                res = amd_compile(kernel, wg, ndr, argtypes)
                @test length(res) == length(AMD_TARGETS)
                @test all(r -> r.bytes > 0, res)
            end
        end
    end
end
