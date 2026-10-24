# Exact Float32 storage/Float64 recurrence semantics through the public seam.
using Test, Random, JSON, SHA

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — CPU-order GDN device regression skipped")
else
    function _gdn_ref_bits(a, b)
        size(a) == size(b) && reinterpret(UInt32, vec(a)) == reinterpret(UInt32, vec(b))
    end

    function _gdn_ref_case(
        q,
        k,
        v,
        g,
        beta,
        initial;
        tiled = false,
        l2norm = true,
        chunks = nothing,
    )
        seq, nv, dv = size(v)
        chunks === nothing && (chunks = filter(!isempty, (1:max(1, seq÷2), (max(1, seq÷2)+1):seq)))
        baseline = nothing
        for rep = 1:3
            state_cpu = copy(initial)
            state_gpu = Lava.LavaArray{Float32}(initial)
            collected = Array{Float32,3}[]
            for chunk in chunks
                inputs = (
                    copy(q[chunk, :, :]),
                    copy(k[chunk, :, :]),
                    copy(v[chunk, :, :]),
                    copy(g[chunk, :]),
                    copy(beta[chunk, :]),
                )
                originals = map(copy, inputs)
                devices = map(x -> Lava.LavaArray{Float32}(x), inputs)
                expected = fill(NaN32, length(chunk), nv, dv)
                Gesso.Inference._gated_delta_rule!(
                    Gesso.CPUBackend(),
                    expected,
                    inputs...,
                    state_cpu;
                    l2norm,
                    v_head_tiled = tiled,
                )
                output = Lava.LavaArray{Float32}(fill(NaN32, size(expected)))
                returned = Gesso.Inference._gated_delta_rule!(
                    Gesso.LavaBackend(),
                    output,
                    devices...,
                    state_gpu;
                    l2norm,
                    v_head_tiled = tiled,
                )
                actual = Array(output)
                @test returned === output
                @test all(isfinite, actual)
                @test _gdn_ref_bits(actual, expected)
                @test _gdn_ref_bits(Array(state_gpu), state_cpu)
                @test all(_gdn_ref_bits(a, b) for (a, b) in zip(inputs, originals))
                @test all(_gdn_ref_bits(Array(a), b) for (a, b) in zip(devices, originals))
                push!(collected, actual)
            end
            full = cat(collected...; dims = 1)
            rep == 1 ? (baseline = full) : (@test _gdn_ref_bits(full, baseline))
            GC.gc(true)
        end
    end

    @testset "CPU-order Lava GDN recurrence" begin
        for geometry in ((1, 2, 4, 8, 3), (7, 2, 4, 16, 6), (64, 2, 4, 32, 8)),
            tiled in (false, true),
            l2norm in (false, true)

            seq, nk, nv, dk, dv = geometry
            rng = MersenneTwister(0x20261018 + seq + 2tiled + l2norm)
            q = 0.1f0 .* randn(rng, Float32, seq, nk, dk)
            k = 0.1f0 .* randn(rng, Float32, seq, nk, dk)
            v = randn(rng, Float32, seq, nv, dv)
            g = -0.1f0 .- abs.(randn(rng, Float32, seq, nv))
            beta = 0.25f0 .+ 0.5f0 .* rand(rng, Float32, seq, nv)
            initial = 0.01f0 .* randn(rng, Float32, nv, dk, dv)
            @testset "geometry=$geometry tiled=$tiled l2=$l2norm" begin
                _gdn_ref_case(q, k, v, g, beta, initial; tiled, l2norm)
            end
        end
    end

    @testset "GDN decay Float32 exponential boundary contract" begin
        ext = Base.get_extension(Gesso, :GessoLavaExt)
        # Constructor presence alone is never the numeric acceptance check.
        if isdefined(ext, :_lava_gdn_exp32)
            values = Float32[
                -Inf,
                -104,
                -103.97208,
                -100,
                -90,
                -88,
                -20,
                -1,
                -eps(Float32),
                -0.0,
                0.0,
                eps(Float32),
                1,
                20,
                80,
                88,
                88.72284,
                89,
                Inf,
                NaN,
            ]
            expected = exp.(values)
            host = ext._lava_gdn_exp32.(values)
            @test all(
                isnan(b) ? isnan(a) : reinterpret(UInt32, a)==reinterpret(UInt32, b) for
                (a, b) in zip(host, expected)
            )
            input = Lava.LavaArray{Float32}(values)
            output = Lava.LavaArray{Float32}(fill(NaN32, length(values)))
            kb = Lava.KernelAbstractions.get_backend(input)
            ext._gdn_reference_kernel(ext._lava_gdn_decay_reference_kernel!, kb)(
                output,
                input,
                length(values);
                ndrange = length(values),
            )
            actual = Array(output)
            @test all(
                isnan(b) ? isnan(a) : reinterpret(UInt32, a)==reinterpret(UInt32, b) for
                (a, b) in zip(actual, expected)
            )
            @test isequal(Array(input), values)
        else
            @test isdefined(ext, :_lava_gdn_exp32)
        end
    end

    # Campaign execution supplies the immutable original captured first block.
    fixture = get(ENV, "GESSO_GDN_REFERENCE_FIXTURE", "")
    if !isempty(fixture)
        @testset "Original captured GDN recurrence" begin
            events = read(joinpath(fixture, "events.jsonl"))
            @test bytes2hex(sha256(events)) ==
                  "509fac64aea23007a397bddf97b9e2c9b2ef5ddad6450bc779d124cd84897c5d"
            rows = [JSON.parse(line) for line in eachline(IOBuffer(events))]
            stage(name) =
                only(filter(x -> x["kind"]=="stage" && x["label"]=="cpu" && x["stage"]==name, rows))
            function raw(rec)
                bytes = read(joinpath(fixture, rec["path"]))
                @test bytes2hex(sha256(bytes)) == rec["sha256"]
                @test length(bytes) == rec["bytes"]
                reshape(copy(reinterpret(Float32, bytes)), Tuple(Int.(rec["shape"])))
            end
            q, k, v = [raw(f["raw"]) for f in stage("gdn_split")["fields"]]
            g, beta = [raw(f["raw"]) for f in stage("gdn_gates")["fields"]]
            _gdn_ref_case(
                q,
                k,
                v,
                g,
                beta,
                zeros(Float32, size(v, 2), size(q, 3), size(v, 3));
                tiled = true,
                chunks = (1:size(q, 1),),
            )
        end
    end
end
