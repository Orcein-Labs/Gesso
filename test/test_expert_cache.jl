# Per-expert device residency: host residency logic (always runs), routing
# profile and trace format, and the device cache against the uncached path
# (exact) and the CPU oracle, for all-hit, all-miss and mixed residency.

using Test: AbstractTestSet, Broken, get_testset, record
using .GessoTestHelpers: approx_eq, deterministic_rng

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _EC_Inf = Gesso.Inference

# F32-packed expert bank in GGUF (ne0 innermost) layout, as in test_cuda_moe.
function _ec_f32_bank(A3::AbstractArray{Float32,3}, ne0::Int, ne1::Int)
    parts = UInt8[]
    for e in 1:size(A3, 1)
        W = A3[e, :, :]
        append!(parts, reinterpret(UInt8, vec(permutedims(W, (2, 1)))))
    end
    return _EC_Inf.GgufQuantSlice(parts, 0, Int32(0), size(A3, 1), ne0, ne1, ne0 * ne1 * 4, C_NULL, Int64(1))
end

@testset "expert residency: quota, fill order, profile, trace (host)" begin
    rng = deterministic_rng(31)
    n_exp, h, ff = 6, 4, 3
    g = _ec_f32_bank(randn(rng, Float32, n_exp, ff, h), h, ff)
    u = _ec_f32_bank(randn(rng, Float32, n_exp, ff, h), h, ff)
    d = _ec_f32_bank(randn(rng, Float32, n_exp, h, ff), ff, h)
    banks = (g, u, d)
    slot = g.bytes_per_expert + u.bytes_per_expert + d.bytes_per_expert
    @test _EC_Inf.expert_slot_bytes(banks) == slot

    # equal bytes per layer: exactly k experts fit in k*slot bytes, capped at n_exp
    @test _EC_Inf.expert_layer_quota(0, banks, n_exp) == 0
    @test _EC_Inf.expert_layer_quota(3 * slot - 1, banks, n_exp) == 2
    @test _EC_Inf.expert_layer_quota(3 * slot, banks, n_exp) == 3
    @test _EC_Inf.expert_layer_quota(typemax(Int32) * Int64(slot), banks, n_exp) == n_exp

    # no profile: ascending id; profile: descending count, ties by id
    @test _EC_Inf.expert_fill_order(nothing, n_exp) == collect(1:n_exp)
    counts = Dict(4 => 9, 2 => 5, 6 => 5)
    @test _EC_Inf.expert_fill_order(counts, n_exp)[1:4] == [4, 2, 6, 1]

    cache = _EC_Inf.ExpertLayerCache(1, banks, nothing, n_exp, 2)
    _EC_Inf.expert_fill!(cache, _EC_Inf.expert_fill_order(counts, n_exp))
    @test cache.slot_expert == Int32[4, 2]
    @test cache.slot_of == Int32[0, 2, 0, 1, 0, 0]
    _EC_Inf.expert_fill!(cache, collect(1:n_exp))
    @test cache.slot_expert == Int32[1, 2]
    @test count(!iszero, cache.slot_of) == 2

    # profile file: `layer expert count`, comments and blank lines ignored, counts summed
    mktempdir() do dir
        p = joinpath(dir, "profile.tsv")
        write(p, "# comment\n1\t4\t9\n1\t2\t5\n\n2 3 7\n1 4 1\n")
        prof = _EC_Inf.read_expert_profile(p)
        @test prof[1] == Dict(4 => 10, 2 => 5)
        @test prof[2] == Dict(3 => 7)
        bad = joinpath(dir, "bad.tsv")
        write(bad, "1 2\n")
        @test_throws ErrorException _EC_Inf.read_expert_profile(bad)
    end

    # trace line: layer, per-layer row index, ids in rank order
    io = IOBuffer()
    c2 = _EC_Inf.ExpertLayerCache(7, banks, nothing, n_exp, 0)
    _EC_Inf.routing_trace_record!(io, c2, [5, 1])
    _EC_Inf.routing_trace_record!(io, c2, [3, 6])
    @test String(take!(io)) == "7 1 5 1\n7 2 3 6\n"
    @test c2.traced == 2

    # bank wrapper reports the host bank's shape, so the MoE checks apply
    wrapped = _EC_Inf.CachedExpertBank(cache, 1)
    @test size(wrapped) == (n_exp, ff, h)
    @test size(_EC_Inf.CachedExpertBank(cache, 3)) == (n_exp, h, ff)

    # the CPU MoE path refuses cached banks instead of mis-indexing them
    x = ones(Float32, 1, h)
    router = zeros(Float32, n_exp, h)
    y = zeros(Float32, 1, h)
    @test_throws ErrorException _EC_Inf._moe_topk_swiglu!(
        Gesso.CPUBackend(), y, x, router, wrapped, wrapped, wrapped, 2,
    )
end

if !CUDA_LOADED || !CUDA.functional()
    _skip("no NVIDIA device — device expert residency tests skipped")
else
    cuda = Gesso.CUDABackend()
    cpu = Gesso.CPUBackend()
    ext = Base.get_extension(Gesso, :GessoCUDAExt)

    # Routing is fixed by the router, not by random draws: with x = 1 every
    # token sees the same logits c_e, so the top-2 are experts 1 and 3 here.
    n_exp, h, ff, n_used, seq = 6, 8, 4, 2, 2
    rng = deterministic_rng(37)
    A_g = randn(rng, Float32, n_exp, ff, h)
    A_u = randn(rng, Float32, n_exp, ff, h)
    A_d = randn(rng, Float32, n_exp, h, ff)
    gate = _ec_f32_bank(A_g, h, ff)
    up = _ec_f32_bank(A_u, h, ff)
    down = _ec_f32_bank(A_d, ff, h)
    slot_bytes = gate.bytes_per_expert + up.bytes_per_expert + down.bytes_per_expert
    c = Float32[3.0, 1.0, 2.0, 0.5, 0.25, 0.1]
    router = repeat(c ./ Float32(h), 1, h)
    x = ones(Float32, seq, h)
    x[2, :] .= 0.5f0

    y_cpu = zeros(Float32, seq, h)
    Gesso.Inference._cpu_moe_topk_swiglu!(
        y_cpu, x, router, A_g, A_u, A_d, n_used,
    )

    # uncached reference: plain host GgufQuantSlice banks, the existing quant path
    y_ref = CUDA.zeros(Float32, seq, h)
    Gesso.Inference._moe_topk_swiglu!(
        cuda, y_ref, CuArray{Float32}(x), CuArray{Float32}(router), gate, up, down, n_used,
    )
    y_ref_h = Array(y_ref)
    @test approx_eq(y_ref_h, y_cpu; atol=1e-3)

    # Build the cache through the real placement, with an explicit byte budget
    # and an optional profile file that picks the resident set.
    function cached_run(budget_bytes::Int64, resident::Vector{Int})
        profile_path = ""
        mktempdir() do dir
            if !isempty(resident)
                profile_path = joinpath(dir, "profile.tsv")
                open(profile_path, "w") do io
                    for (r, e) in enumerate(resident)
                        println(io, 1, ' ', e, ' ', 1000 - r)
                    end
                end
            end
            conv = IdDict{Any,Any}()
            withenv("GESSO_EXPERT_PROFILE" => (isempty(profile_path) ? nothing : profile_path)) do
                ext._place_expert_caches!(conv, [(1, gate, up, down)]; budget=budget_bytes)
            end
            return conv
        end
    end

    function run_cached(conv)
        y = CUDA.zeros(Float32, seq, h)
        Gesso.Inference._moe_topk_swiglu!(
            cuda, y, CuArray{Float32}(x), CuArray{Float32}(router),
            conv[gate], conv[up], conv[down], n_used,
        )
        return Array(y), conv[gate].layer
    end

    @testset "device expert cache: all-hit matches uncached exactly" begin
        conv = cached_run(Int64(n_exp) * slot_bytes, collect(1:n_exp))
        y, layer = run_cached(conv)
        @test layer.slot_of == Int32.(1:n_exp)
        @test y == y_ref_h
        @test approx_eq(y, y_cpu; atol=1e-3)
        @test layer.hits == seq * n_used
        @test layer.misses == 0
    end

    @testset "device expert cache: all-miss matches uncached exactly" begin
        conv = cached_run(Int64(0), Int[])
        y, layer = run_cached(conv)
        @test isempty(layer.slot_expert)
        @test y == y_ref_h
        @test layer.hits == 0
        @test layer.misses == seq * n_used
    end

    @testset "device expert cache: mixed residency matches uncached exactly" begin
        # expert 1 resident (rank-1 hit every token), expert 3 missed (rank 2)
        conv = cached_run(Int64(1) * slot_bytes, [1])
        y, layer = run_cached(conv)
        @test layer.slot_of[1] == 1 && layer.slot_of[3] == 0
        @test y == y_ref_h
        @test approx_eq(y, y_cpu; atol=1e-3)
        @test layer.hits == seq
        @test layer.misses == seq
    end

    @testset "device expert cache: accepted by the CUDA storage check" begin
        conv = cached_run(Int64(n_exp) * slot_bytes, collect(1:n_exp))
        t = (storage=conv[gate],)
        @test Gesso.Inference._check_device_storage(:probe, cuda, t) === nothing
    end

    @testset "device expert cache: budget arithmetic is free minus reserve" begin
        MiB = Int64(1024 * 1024)
        @test ext._expert_budget_from_free(Int64(3) * 1024 * MiB, 1024) == Int64(2) * 1024 * MiB
        @test ext._expert_budget_from_free(Int64(7300) * MiB, 1024) == Int64(6276) * MiB
        @test ext._expert_budget_from_free(Int64(500) * MiB, 1024) == 0
        @test ext._expert_budget_from_free(Int64(500) * MiB, 0) == Int64(500) * MiB
        @test ext._expert_budget_from_free(Int64(500) * MiB, -5) == Int64(500) * MiB
        @test_throws ErrorException ext._expert_budget_from_free(-1, 0)
    end

    @testset "device expert cache: budget env and fallbacks" begin
        withenv("GESSO_GPU_EXPERT_CACHE" => nothing) do
            @test !ext._expert_cache_enabled()   # opt-in: whole-slab placement is the default
        end
        withenv("GESSO_GPU_EXPERT_CACHE" => "1") do
            @test ext._expert_cache_enabled()
        end
        withenv("GESSO_GPU_EXPERT_CACHE_MIB" => "0") do
            @test ext._expert_cache_budget_bytes() == 0
        end
        withenv("GESSO_GPU_EXPERT_CACHE_MIB" => "3") do
            @test ext._expert_cache_budget_bytes() == Int64(3) * 1024 * 1024
        end
        withenv("GESSO_GPU_EXPERT_CACHE_MIB" => "not-a-number") do
            @test_throws ErrorException ext._expert_cache_budget_bytes()
        end
        # with no override the budget is free memory minus the reserve, never negative
        withenv("GESSO_GPU_EXPERT_CACHE_MIB" => nothing, "GESSO_GPU_RESERVE_MIB" => "1048576") do
            @test ext._expert_cache_budget_bytes() == 0
        end
    end
end
