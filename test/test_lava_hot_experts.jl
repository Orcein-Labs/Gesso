# Lava hot host experts: the per-layer device pool in ext/lava_hot_experts.jl.
# Checks a zero-slot pool and an unregistered slice against the host expert route
# bit for bit, a full pool against an F64 reference, a partial pool against an
# independent model of the residency policy with per-token counts derived by
# hand, slot fills byte for byte, and two negative controls (a hit read from the
# wrong slot, and a routing weight applied twice) that must exceed the tolerance.
# Uses F32 expert banks so the host route and the device GEMV read the same
# numbers. Named skip without a Vulkan device.

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
    _skip("no usable Vulkan device — Lava hot-expert tests skipped")
else
    const _HOT_EXT = Base.get_extension(Gesso, :GessoLavaExt)
    const _HOT_INF = Gesso.Inference
    const _HOT_H = 512
    const _HOT_FF = 256
    const _HOT_NEXP = 6
    const _HOT_NUSED = 2
    # Each expert is 3 banks of 256*512*4 bytes, 1.5 MiB together: a 4 MiB pool
    # holds two experts. The repeated (5, 6) lets its decayed count clear the
    # resident pair by the swap margin, so the two-slot pool evicts; the (3, 4)
    # at position 6 is the miss that has to be refused.
    const _HOT_SCRIPT = [(1, 2), (5, 6), (5, 6), (5, 6), (5, 6), (3, 4), (5, 6), (5, 6), (5, 6)]

    function _hot_pack(A3::Array{Float32,3})
        parts = UInt8[]
        for e in 1:size(A3, 1)
            g = vec(permutedims(A3[e, :, :], (2, 1)))
            append!(parts, reinterpret(UInt8, g))
        end
        return parts
    end

    # A3[e, r, c] is one expert with c innermost, as the GGUF layout has it.
    function _hot_bank(A3::Array{Float32,3})
        n, r, c = size(A3)
        return _HOT_INF.GgufQuantSlice(
            _hot_pack(A3), 0, Int32(0), n, c, r, c * r * 4, C_NULL, Int64(1),
        )
    end

    # Each feature f routes to the pair script[f]: logit 2 for the first expert,
    # 1 for the second, 0 for the rest. A one-hot input at f then picks that pair.
    function _hot_router(script)
        R = zeros(Float32, _HOT_NEXP, _HOT_H)
        for (f, (p1, p2)) in enumerate(script)
            R[p1, f] = 2.0f0
            R[p2, f] = 1.0f0
        end
        return R
    end

    _hot_onehot(f::Int) = (x = zeros(Float32, 1, _HOT_H); x[1, f] = 1.0f0; x)

    function _hot_route_ref(R, xrow)
        l = Float64.(R) * Float64.(vec(xrow))
        p = exp.(l .- maximum(l))
        p ./= sum(p)
        ord = sortperm(p; rev=true)[1:_HOT_NUSED]
        w = p[ord] ./ sum(p[ord])
        return ord, w
    end

    _hot_silu(v) = v ./ (1 .+ exp.(-v))

    function _hot_expert_ref(Wg, Wu, Wd, e, xrow)
        x = Float64.(vec(xrow))
        g = Float64.(Wg[e, :, :]) * x
        u = Float64.(Wu[e, :, :]) * x
        return Float64.(Wd[e, :, :]) * (_hot_silu(g) .* u)
    end

    function _hot_ref(Wg, Wu, Wd, xrow, experts, wts)
        y = zeros(Float64, _HOT_H)
        for (k, e) in enumerate(experts)
            y .+= Float64(wts[k]) .* _hot_expert_ref(Wg, Wu, Wd, e, xrow)
        end
        return y
    end

    _hot_tol(ref) = 1e-4 * max(1.0, maximum(abs, ref))

    # One decode token through the layer, as the hybrid and qwen4exp routes call it.
    function _hot_decode(gate, up, down, R, xrow)
        x = Lava.LavaArray{Float32}(xrow)
        y = Lava.LavaArray{Float32}(zeros(Float32, 1, _HOT_H))
        _HOT_EXT._lhot_moe_decode!(y, x, xrow, R, gate, up, down, _HOT_NUSED)
        return Array(y)
    end

    # The host route the hot path must reproduce with no pool: the same routing
    # and host expert calls that _moe_topk_swiglu! makes for one decode token.
    function _hot_host_ref(gate, up, down, R, xrow)
        experts = Matrix{Int}(undef, _HOT_NUSED, 1)
        wts = Matrix{Float32}(undef, _HOT_NUSED, 1)
        _HOT_INF._moe_route_token!(
            experts, wts, 1, xrow, R, zeros(Float32, _HOT_NEXP), zeros(Int, _HOT_NEXP), _HOT_NUSED,
        )
        yref = zeros(Float32, _HOT_H)
        _HOT_INF._moe_host_experts_auto!(
            yref, vec(xrow), view(experts, :, 1), view(wts, :, 1), _HOT_NUSED, gate, up, down,
        )
        return yref
    end

    # One expert's device output with the given slot and alpha.
    function _hot_expert_dev(layer, slot::Int, xrow; alpha::Float32=1.0f0)
        x = Lava.LavaArray{Float32}(vec(xrow))
        g = Lava.LavaArray{Float32}(zeros(Float32, _HOT_FF))
        u = Lava.LavaArray{Float32}(zeros(Float32, _HOT_FF))
        y = Lava.LavaArray{Float32}(zeros(Float32, _HOT_H))
        _HOT_INF._qgemv_expert!(g, x, layer.pg, slot; alpha=1.0f0, beta=0.0f0)
        _HOT_INF._qgemv_expert!(u, x, layer.pu, slot; alpha=1.0f0, beta=0.0f0)
        a = Lava.LavaArray{Float32}(zeros(Float32, _HOT_FF))
        a .= (g ./ (1.0f0 .+ exp.(-g))) .* u
        _HOT_INF._qgemv_expert!(y, a, layer.pd, slot; alpha=alpha, beta=0.0f0)
        return Array(y)
    end

    # Runs f with a fresh pool budget of mib MiB and the switch on, then drops every pool.
    function _hot_with(f, mib::Int)
        _HOT_EXT.release_lava_hot_experts!()
        # A model loaded by an earlier test file leaves its layer count behind.
        nl = _HOT_EXT._LHOT_NLAYERS[]
        _HOT_EXT._LHOT_NLAYERS[] = 0
        try
            withenv("GESSO_GPU_HOT_EXPERTS" => "1", "GESSO_GPU_HOT_EXPERTS_MIB" => string(mib)) do
                f()
            end
        finally
            _HOT_EXT.release_lava_hot_experts!()
            _HOT_EXT._LHOT_NLAYERS[] = nl
        end
    end

    # The residency policy as the module states it, with the pool's slot order:
    # free slots fill lowest first, and a tie for the coldest resident goes to the
    # lower slot. Keeping slot order here makes the victim choice reproducible; a
    # Set would leave it to hash order.
    mutable struct _HotModel
        slots::Vector{Int}
        score::Vector{Float64}
        seen::Vector{Int}
        t::Int
        fills::Int
        evictions::Int
    end

    _HotModel(n::Int, nslots::Int) = _HotModel(zeros(Int, nslots), zeros(n), zeros(Int, n), 0, 0, 0)

    # Decayed pick counts, a margin of 1 before a swap, at most one eviction per
    # token, and two fills per token.
    function _hot_model_step!(m::_HotModel, picks)
        m.t += 1
        dec(e) = m.score[e] * 0.99^(m.t - m.seen[e])
        for e in picks
            m.score[e] = dec(e) + 1.0
            m.seen[e] = m.t
        end
        evicted = false
        for _ in 1:2
            best, bs = 0, -1.0
            for e in picks
                e in m.slots && continue
                v = dec(e)
                if v > bs
                    best, bs = e, v
                end
            end
            best == 0 && break
            slot = findfirst(==(0), m.slots)
            if slot === nothing
                evicted && break
                slot, vs = 0, Inf
                for s in eachindex(m.slots)
                    r = m.slots[s]
                    m.seen[r] == m.t && continue
                    v = dec(r)
                    if v < vs
                        slot, vs = s, v
                    end
                end
                (slot == 0 || vs + 1.0 >= bs) && break
                evicted = true
                m.evictions += 1
            end
            m.slots[slot] = best
            m.fills += 1
        end
        return m
    end

    rng = MersenneTwister(20261027)
    Wg = randn(rng, Float32, _HOT_NEXP, _HOT_FF, _HOT_H) .* 0.05f0
    Wu = randn(rng, Float32, _HOT_NEXP, _HOT_FF, _HOT_H) .* 0.05f0
    Wd = randn(rng, Float32, _HOT_NEXP, _HOT_H, _HOT_FF) .* 0.05f0
    gate = _hot_bank(Wg)
    up = _hot_bank(Wu)
    down = _hot_bank(Wd)
    R = _hot_router(_HOT_SCRIPT)

    @testset "zero-slot pool is the host route, bit for bit" begin
        _hot_with(0) do
            layer = _HOT_EXT._lhot_layer!(gate, up, down, 0)
            @test layer.nslots == 0
            for (f, _) in enumerate(_HOT_SCRIPT)
                xrow = _hot_onehot(f)
                # With no slots the hybrid and qwen4exp routes never call the pool.
                @test !_HOT_EXT._lhot_wants(xrow, gate, up, down, 0)
                y = _hot_decode(gate, up, down, R, xrow)
                @test y[1, :] == _hot_host_ref(gate, up, down, R, xrow)
            end
        end
    end

    @testset "unregistered slice takes the host route, bit for bit" begin
        _hot_with(4) do
            gate_u = _hot_bank(Wg)
            up_u = _hot_bank(Wu)
            down_u = _hot_bank(Wd)
            for f in 1:3
                xrow = _hot_onehot(f)
                y = _hot_decode(gate_u, up_u, down_u, R, xrow)
                @test y[1, :] == _hot_host_ref(gate_u, up_u, down_u, R, xrow)
            end
            # The fallback must not register the slice: that would fix its slot count.
            @test !haskey(_HOT_EXT._LHOT, gate_u)
        end
    end

    @testset "full pool matches the F64 reference" begin
        _hot_with(16) do
            layer = _HOT_EXT._lhot_layer!(gate, up, down, 0)
            @test layer.nslots == _HOT_NEXP
            for e in 1:_HOT_NEXP
                _HOT_EXT._lhot_fill!(layer, e, e)
                layer.slot_of[e] = e
                layer.expert_in[e] = e
            end
            layer.free = Int[]
            for (f, (p1, p2)) in enumerate(_HOT_SCRIPT)
                xrow = _hot_onehot(f)
                y = _hot_decode(gate, up, down, R, xrow)
                ord, w = _hot_route_ref(R, xrow)
                @test Set(ord) == Set((p1, p2))
                ref = _hot_ref(Wg, Wu, Wd, xrow, ord, w)
                @test maximum(abs, y[1, :] .- ref) <= _hot_tol(ref)
            end
            @test layer.hits == layer.picks == _HOT_NUSED * length(_HOT_SCRIPT)
            @test layer.fills == 0
        end
    end

    @testset "partial pool: hit, miss, eviction and fill, counts derived by hand" begin
        # Two slots for six experts. dec(e) = score * 0.99^(t - seen). A miss swaps
        # in only when it beats the coldest resident by more than the margin 1.0,
        # at most one swap per token, and free slots fill first.
        #   t1 {1,2}: 2 misses; both fill the free slots.
        #   t2 {5,6}: 2 misses; best 5 (1.00) against coldest 1 (0.99): 1.00 is not
        #             above 0.99 + 1.0, no swap.
        #   t3 {5,6}: 2 misses; best 5 (1.99) against coldest 1 (0.9801, tied with
        #             2, lowest slot wins): 1.99 > 0.9801 + 1.0, so 5 takes slot 1
        #             and 1 is evicted. 6 waits, one swap per token.
        #   t4 {5,6}: 5 is a hit (slot 1), 6 a miss; best 6 (2.9701) against coldest
        #             2 (0.9703): swap, 6 takes slot 2, 2 is evicted.
        #   t5 {5,6}: both hits.
        #   t6 {3,4}: 2 misses; coldest resident 5 or 6 (3.90) is far above 3 (1.00).
        #   t7-t9 {5,6}: both hits each token.
        # Cumulative totals: hits 9, misses 9, fills 4, evictions 2.
        cum_hits = [0, 0, 0, 1, 3, 3, 5, 7, 9]
        cum_misses = [2, 4, 6, 7, 7, 9, 9, 9, 9]
        cum_fills = [2, 2, 3, 4, 4, 4, 4, 4, 4]
        cum_evict = [0, 0, 1, 2, 2, 2, 2, 2, 2]
        _hot_with(4) do
            layer = _HOT_EXT._lhot_layer!(gate, up, down, 0)
            @test layer.nslots == 2
            model = _HotModel(_HOT_NEXP, 2)
            for (f, _) in enumerate(_HOT_SCRIPT)
                xrow = _hot_onehot(f)
                y = _hot_decode(gate, up, down, R, xrow)
                ord, w = _hot_route_ref(R, xrow)
                ref = _hot_ref(Wg, Wu, Wd, xrow, ord, w)
                @test maximum(abs, y[1, :] .- ref) <= _hot_tol(ref)
                _hot_model_step!(model, ord)
                @test Set(e for e in layer.expert_in if e > 0) == Set(e for e in model.slots if e > 0)
                @test layer.hits == cum_hits[f]
                @test layer.picks - layer.hits == cum_misses[f]
                @test layer.fills == cum_fills[f]
                @test layer.evictions == cum_evict[f]
            end
            @test model.fills == layer.fills
            @test model.evictions == layer.evictions
            @test Set(e for e in layer.expert_in if e > 0) == Set((5, 6))

            # Every resident slot holds its expert's bytes exactly.
            for (s, e) in enumerate(layer.expert_in)
                e == 0 && continue
                for (pool, bank) in ((layer.pg, gate), (layer.pu, up), (layer.pd, down))
                    b = bank.bytes_per_expert
                    off = _HOT_INF._quant_expert_off0(bank, e)
                    host = bank.data[(off + 1):(off + b)]
                    dev = Array(pool.data)[((s - 1) * b + 1):(s * b)]
                    @test dev == host
                end
            end
        end
    end

    @testset "negative controls exceed the tolerance" begin
        _hot_with(16) do
            layer = _HOT_EXT._lhot_layer!(gate, up, down, 0)
            for e in 1:_HOT_NEXP
                _HOT_EXT._lhot_fill!(layer, e, e)
            end
            e = 3
            xrow = _hot_onehot(1)
            ref = _hot_expert_ref(Wg, Wu, Wd, e, xrow)
            w = 0.37f0

            good = _hot_expert_dev(layer, e, xrow; alpha=w)
            @test maximum(abs, good .- w .* ref) <= _hot_tol(w .* ref)

            wrong = _hot_expert_dev(layer, e + 1, xrow; alpha=w)
            @test maximum(abs, wrong .- w .* ref) > _hot_tol(w .* ref)

            doubled = _hot_expert_dev(layer, e, xrow; alpha=w * w)
            @test maximum(abs, doubled .- w .* ref) > _hot_tol(w .* ref)
        end
    end
    @testset "registered layers split the budget evenly" begin
        # Three expert layers and a budget of two slots each, plus half a slot.
        # Before the count was taken at load, the first layer seen took every
        # expert and the other two got nothing.
        banks = [(_hot_bank(Wg), _hot_bank(Wu), _hot_bank(Wd)) for _ in 1:3]
        blocks = Any[(wexp_gate=(storage=g,), wexp_up=(storage=u,), wexp_down=(storage=d,)) for (g, u, d) in banks]
        push!(blocks, (wq=(storage=zeros(Float32, 2, 2),),))
        bpe = sum(b.bytes_per_expert for b in banks[1])
        _hot_with(0) do
            @test _HOT_EXT._lhot_register!(blocks) == 3
            _HOT_EXT._LHOT_BUDGET[] = _HOT_EXT._LHOT_LEFT[] = 6 * bpe + div(bpe, 2)
            for (g, u, d) in banks
                @test _HOT_EXT._lhot_layer!(g, u, d, 0).nslots == 2
            end

            # The negative control: the same budget with no count is greedy.
            _HOT_EXT.release_lava_hot_experts!()
            _HOT_EXT._LHOT_NLAYERS[] = 0
            _HOT_EXT._LHOT_BUDGET[] = _HOT_EXT._LHOT_LEFT[] = 6 * bpe + div(bpe, 2)
            @test [_HOT_EXT._lhot_layer!(g, u, d, 0).nslots for (g, u, d) in banks] == [_HOT_NEXP, 0, 0]

            # A new load drops the old pools.
            @test !isempty(_HOT_EXT._LHOT)
            _HOT_EXT._lhot_register!(blocks[1:2])
            @test isempty(_HOT_EXT._LHOT)
            @test _HOT_EXT._LHOT_NLAYERS[] == 2
        end
    end

    @testset "a prefill pass frees the pool and decode restores the hottest experts" begin
        lava = Gesso.LavaBackend()
        _hot_with(4) do
            layer = _HOT_EXT._lhot_layer!(gate, up, down, 0)
            for f in 1:5
                _hot_decode(gate, up, down, R, _hot_onehot(f))
            end
            @test Set(layer.expert_in) == Set((5, 6))
            score = copy(layer.score)

            # A one-row pass is a decode token's shape and keeps the pool.
            _HOT_INF._backend_prefill_pass!(lava, 1)
            @test layer.live && layer.nslots == 2 && layer.pg !== nothing

            _HOT_INF._backend_prefill_pass!(lava, 8)
            @test !layer.live
            @test layer.nslots == 0
            @test layer.pg === nothing && layer.pu === nothing && layer.pd === nothing
            @test all(==(0), layer.slot_of)
            @test _HOT_EXT._LHOT_BUDGET[] == -1
            @test layer.score == score
            @test !_HOT_EXT._lhot_suspend!()

            # The next decode token gets a pool back, holding 5 and 6 byte for byte.
            @test _HOT_EXT._lhot_wants(_hot_onehot(7), gate, up, down, 0)
            @test layer.live && layer.nslots == 2
            @test layer.restores == 1
            @test Set(layer.expert_in) == Set((5, 6))
            for (s, e) in enumerate(layer.expert_in)
                @test layer.slot_of[e] == s
                for (pool, bank) in ((layer.pg, gate), (layer.pu, up), (layer.pd, down))
                    b = bank.bytes_per_expert
                    off = _HOT_INF._quant_expert_off0(bank, e)
                    @test Array(pool.data)[((s - 1) * b + 1):(s * b)] == bank.data[(off + 1):(off + b)]
                end
            end

            # Both picks hit at once, with no fill, and the output holds.
            hits, fills = layer.hits, layer.fills
            xrow = _hot_onehot(7)
            y = _hot_decode(gate, up, down, R, xrow)
            ord, w = _hot_route_ref(R, xrow)
            ref = _hot_ref(Wg, Wu, Wd, xrow, ord, w)
            @test maximum(abs, y[1, :] .- ref) <= _hot_tol(ref)
            @test layer.hits == hits + 2
            @test layer.fills == fills
        end
    end
end
