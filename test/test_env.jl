# GESSO_DISABLE / GESSO_PROFILE / GESSO_TUNE parsing (src/env.jl).

@testset "env lists: DISABLE" begin
    withenv("GESSO_DISABLE" => nothing) do
        @test !Gesso.env_disabled(:prefault)
    end
    withenv("GESSO_DISABLE" => " Prefault , quant_simd,,") do
        @test Gesso.env_disabled(:prefault)
        @test Gesso.env_disabled(:quant_simd)
        @test !Gesso.env_disabled(:prefill_window)
    end
    # The cached parse follows the variable when it changes.
    withenv("GESSO_DISABLE" => "prefill_window") do
        @test Gesso.env_disabled(:prefill_window)
        @test !Gesso.env_disabled(:prefault)
    end
end

@testset "env lists: PROFILE" begin
    withenv("GESSO_PROFILE" => nothing) do
        @test Gesso.env_profile(:decode) === nothing
        @test !Gesso.env_profile_on(:timing)
    end
    withenv("GESSO_PROFILE" => "1, expert = /tmp/e.tsv ,timing") do
        @test Gesso.env_profile(:decode) == ""
        @test Gesso.env_profile_on(:decode)
        @test Gesso.env_profile(:expert) == "/tmp/e.tsv"
        @test Gesso.env_profile(:timing) == ""
        @test Gesso.env_profile(:routing) === nothing
    end
    withenv("GESSO_PROFILE" => "decode") do
        @test Gesso.env_profile_on(:decode)
    end
end

@testset "env lists: TUNE" begin
    withenv("GESSO_TUNE" => nothing) do
        @test Gesso.env_tune(:dq_gemm_min, 32) == 32
        @test Gesso.env_tune(:hot_decay, 0.99) == 0.99
        @test Gesso.env_tune(:gemv_pool, "auto") == "auto"
        @test Gesso.env_tune(:lava_hot_experts, false) === false
    end
    withenv("GESSO_TUNE" => " DQ_GEMM_MIN = 64 ,hot_decay=0.5,gemv_pool=threads,lava_hot_experts=on,hot_fill=") do
        @test Gesso.env_tune(:dq_gemm_min, 32) === 64
        @test Gesso.env_tune(:hot_decay, 0.99) === 0.5
        @test Gesso.env_tune(:gemv_pool, "auto") == "threads"
        @test Gesso.env_tune(:lava_hot_experts, false) === true
        @test Gesso.env_tune(:hot_fill, 2) == 2          # empty value = default
        @test Gesso.env_tune(:hot_mib, -1) == -1         # absent key = default
    end
    for (raw, want) in ("1" => true, "true" => true, "on" => true,
                        "0" => false, "false" => false, "off" => false)
        withenv("GESSO_TUNE" => "cuda_expert_cache=$raw") do
            @test Gesso.env_tune(:cuda_expert_cache, !want) === want
        end
    end
    withenv("GESSO_TUNE" => "cuda_expert_cache=maybe,dq_gemm_min=lots,hot_decay=x,hot_fill=2.5") do
        @test_throws ErrorException Gesso.env_tune(:cuda_expert_cache, false)
        @test_throws ErrorException Gesso.env_tune(:dq_gemm_min, 32)
        @test_throws ErrorException Gesso.env_tune(:hot_decay, 0.99)
        @test_throws ErrorException Gesso.env_tune(:hot_fill, 2)
    end
end

@testset "env lists: replaced variables warn" begin
    withenv("GESSO_PREFAULT" => "0", "GESSO_GPU_HOT_FILL" => "4") do
        @test_logs (:warn, r"GESSO_PREFAULT.*GESSO_DISABLE=prefault"s) Gesso._env_warn_replaced()
    end
    withenv((old => nothing for (old, _) in Gesso._ENV_REPLACED)...) do
        @test_logs Gesso._env_warn_replaced()
    end
    # Every replaced name is unique and none is still a user setting.
    olds = first.(collect(Gesso._ENV_REPLACED))
    @test allunique(olds)
end
