# Context budget and compaction policy (src/Inference/context_budget.jl). Pure
# CPU arithmetic over a synthetic 35B-shaped spec and hand-made cost curves, so
# every expected value below is derived by hand in the comment beside it.

using Gesso: kv_bytes_per_token, memory_ceiling_tokens, CostModel, prefill_seconds,
    decode_seconds, ContextBudget, latency_ceiling_tokens, CompactionPolicy,
    CompactionTracker, reset!, decide_compaction

const _CB = Gesso.Inference

# Ornith-1.5-35B shape: 40 layers, every 4th full attention (10 KV layers),
# 2 KV heads, head_dim 256.
function _spec35(; n_layers::Int=40, full_every::Int=4, head_dim::Int=256)
    types = [i % full_every == full_every - 1 ? :full_attention : :linear_attention
             for i in 0:(n_layers - 1)]
    mixer = _CB.LinearMixerSpec(layer_types=types, num_key_heads=16, key_head_dim=128,
        num_value_heads=32, value_head_dim=128, conv_kernel_dim=4)
    return _CB.ArchitectureSpec(family=:qwen35moe, hidden_size=2048, num_layers=n_layers,
        n_heads=16, n_kv_heads=2, head_dim=head_dim, vocab_size=1000,
        intermediate_size=512, mixer=mixer)
end

@testset "context budget: kv_bytes_per_token" begin
    spec = _spec35()
    # Per attention layer: K and V, each 2 heads x 256 values.
    #   f32: 256 * 4 B per head                    -> 2 * 2 * 1024    = 4096
    #   f16: 256 * 2 B per head                    -> 2 * 2 * 512     = 2048
    #   q8:  256 Int8 + 8 Float16 scales (16 B)    -> 272 B per head  -> 2 * 2 * 272 = 1088
    #   q4h: 128 bytes (two codes a byte) + 16 B   -> 144 B per head  -> 2 * 2 * 144 = 576
    # Times 10 KV layers.
    @test kv_bytes_per_token(spec, :f32) == 40960
    @test kv_bytes_per_token(spec, :f16) == 20480
    @test kv_bytes_per_token(spec, :q8) == 10880
    @test kv_bytes_per_token(spec, :q4h) == 5760

    # Red control: the GDN layers must not be counted. Making every layer a KV
    # layer quadruples the figure, because 40 layers is 4x 10.
    all_attn = _spec35(full_every=1)
    @test kv_bytes_per_token(all_attn, :q8) == 4 * 10880
    @test kv_bytes_per_token(all_attn, :q4h) == 4 * 5760

    # A dense model has no mixer; every layer carries KV.
    dense = _CB.ArchitectureSpec(family=:llama, hidden_size=2048, num_layers=40,
        n_heads=16, n_kv_heads=2, head_dim=256, vocab_size=1000, intermediate_size=512)
    @test kv_bytes_per_token(dense, :q8) == 4 * 10880

    @test_throws Gesso.GessoError kv_bytes_per_token(spec, :q3)
    # Quantized formats need whole 32-value scale blocks; q4h also needs a
    # power-of-two head for the Hadamard rotation.
    @test_throws Gesso.GessoError kv_bytes_per_token(_spec35(head_dim=250), :q8)
    @test_throws Gesso.GessoError kv_bytes_per_token(_spec35(head_dim=96), :q4h)
    @test kv_bytes_per_token(_spec35(head_dim=96), :q8) > 0
end

@testset "context budget: memory_ceiling_tokens" begin
    @test memory_ceiling_tokens(budget_bytes=1000, resident_bytes=200,
        scratch_bytes=300, kv_bytes_per_token=100) == 5
    # floor: 550 free bytes at 100 per token
    @test memory_ceiling_tokens(budget_bytes=1050, resident_bytes=200,
        scratch_bytes=300, kv_bytes_per_token=100) == 5
    @test memory_ceiling_tokens(budget_bytes=1000, resident_bytes=0,
        kv_bytes_per_token=10) == 100
    # nothing fits: zero, not negative
    @test memory_ceiling_tokens(budget_bytes=1000, resident_bytes=1000,
        kv_bytes_per_token=10) == 0
    @test memory_ceiling_tokens(budget_bytes=1000, resident_bytes=5000,
        kv_bytes_per_token=10) == 0
    @test_throws ArgumentError memory_ceiling_tokens(budget_bytes=1000,
        resident_bytes=0, kv_bytes_per_token=0)
end

@testset "context budget: CostModel" begin
    cm = CostModel(prefill=[100 => 1.0, 200 => 3.0], decode=[0 => 1.0, 1000 => 2.0])
    # linear between samples: at 150 the prefill rate is 2.0 s/token
    @test prefill_seconds(cm, 10, 150) == 20.0
    @test decode_seconds(cm, 10, 500) == 15.0
    # flat below the first sample and past the last
    @test prefill_seconds(cm, 10, 50) == 10.0
    @test prefill_seconds(cm, 10, 5000) == 30.0
    @test decode_seconds(cm, 10, 5000) == 20.0
    @test decode_seconds(cm, 0, 500) == 0.0

    @test_throws ArgumentError CostModel(prefill=Pair{Int,Float64}[], decode=[0 => 1.0])
    @test_throws ArgumentError CostModel(prefill=[200 => 1.0, 100 => 1.0], decode=[0 => 1.0])
    @test_throws ArgumentError CostModel(prefill=[100 => 1.0, 100 => 2.0], decode=[0 => 1.0])
    @test_throws ArgumentError CostModel(prefill=[100 => 0.0], decode=[0 => 1.0])
    @test_throws ArgumentError CostModel(prefill=[100 => NaN], decode=[0 => 1.0])
    @test_throws ArgumentError CostModel(prefill=[-1 => 1.0], decode=[0 => 1.0])
    @test_throws ArgumentError CostModel(prefill=[0 => 1.0], decode=[0 => -2.0])
end

@testset "context budget: latency_ceiling_tokens" begin
    # decode(L) = 1.0 + L/1000 s/token up to L = 1000, so decode tok/s >= 0.5 means
    # decode <= 2.0, i.e. L <= 1000. The third sample is needed: decode is held flat
    # past the last sample, so without it no length would ever exceed 2.0.
    # Cold prefill of L tokens = 0.5 * L seconds.
    cm = CostModel(prefill=[0 => 0.5], decode=[0 => 1.0, 1000 => 2.0, 2000 => 3.0])
    @test latency_ceiling_tokens(cm; min_decode_tok_s=0.5, max_len=5000) == 1000
    @test latency_ceiling_tokens(cm; max_reprefill_s=400.0, max_len=5000) == 800
    # both: the tighter one wins (min(1000, 800))
    @test latency_ceiling_tokens(cm; min_decode_tok_s=0.5, max_reprefill_s=400.0,
        max_len=5000) == 800
    @test latency_ceiling_tokens(cm; min_decode_tok_s=0.5, max_reprefill_s=600.0,
        max_len=5000) == 1000
    # nothing constrained: the cap itself
    @test latency_ceiling_tokens(cm; max_len=5000) == 5000
    # the cap binds before either constraint
    @test latency_ceiling_tokens(cm; min_decode_tok_s=0.5, max_len=500) == 500
    # no positive length qualifies
    @test latency_ceiling_tokens(cm; max_reprefill_s=0.1, max_len=5000) == 0

    # The crossing falls between samples: decode = 0.5 + L/1000, and 1.0 s/token
    # is reached at L = 500, which no sample holds.
    mid = CostModel(prefill=[0 => 0.5], decode=[0 => 0.5, 1000 => 1.5])
    @test latency_ceiling_tokens(mid; min_decode_tok_s=1.0, max_len=5000) == 500
end

@testset "context budget: ContextBudget" begin
    b = ContextBudget(memory_tokens=100000, quality_tokens=32768, latency_tokens=40000)
    @test b.safe_tokens == 32768
    @test b.binding === :quality
    @test ContextBudget(memory_tokens=50000, latency_tokens=40000).binding === :latency
    @test ContextBudget(memory_tokens=1000).binding === :memory
    @test ContextBudget(memory_tokens=1000, quality_tokens=1000).binding === :memory
    u = ContextBudget()
    @test u.safe_tokens == typemax(Int)
    @test u.binding === :none
    @test_throws ArgumentError ContextBudget(memory_tokens=-1)
end

@testset "context budget: CompactionPolicy and tracker" begin
    p = CompactionPolicy()
    @test p.rule === :rent_or_buy
    @test p.reserve_tokens == 4096
    @test p.target_fraction == 0.5
    @test p.threshold_fraction == 0.9
    @test p.summary_tokens == 1024
    @test_throws ArgumentError CompactionPolicy(rule=:bogus)
    @test_throws ArgumentError CompactionPolicy(target_fraction=0.9, threshold_fraction=0.9)
    @test_throws ArgumentError CompactionPolicy(target_fraction=0.95)
    @test_throws ArgumentError CompactionPolicy(threshold_fraction=1.5)
    @test_throws ArgumentError CompactionPolicy(summary_tokens=0)
    @test_throws ArgumentError CompactionPolicy(reserve_tokens=-1)

    t = CompactionTracker()
    t.accumulated_slowdown_s = 12.5
    t.turns_since_compaction = 7
    reset!(t, 50000)
    @test t.accumulated_slowdown_s == 0.0
    @test t.turns_since_compaction == 0
    @test t.last_compacted_len == 50000
end

@testset "context budget: decide_compaction" begin
    budget = ContextBudget(memory_tokens=100000)   # safe 100000, target 50000, threshold 90000

    # Rent: decode(L) = 0.01 + 1e-6 * L s/token, prefill 0.002 s/token.
    # At ctx 60000 a turn of 256 new tokens pays
    #   rent per turn = 256 * (d(60000) - d(50000)) = 256 * (0.07 - 0.06) = 2.56 s.
    # Compaction costs
    #   C = 1024 * d(60000) + (50000 + 1024) * 0.002 = 71.68 + 102.048 = 173.728 s.
    # Rent paid after n turns is 2.56 n. The first n with 2.56 n >= 173.728 is
    # n = 68 (67 turns give 171.52, 68 give 174.08), so compaction fires on turn 68.
    rent_cm = CostModel(prefill=[0 => 0.002], decode=[0 => 0.01, 100000 => 0.11])
    policy = CompactionPolicy()
    tracker = CompactionTracker()
    ctx = 60000
    decisions = [decide_compaction(policy, budget, rent_cm, tracker, ctx) for _ in 1:70]
    @test !decisions[1].compact
    @test decisions[1].reason === :wait
    @test findfirst(d -> d.compact, decisions) == 68
    @test all(d -> d.reason === :wait, decisions[1:67])
    @test decisions[68].reason === :rent_or_buy
    @test decisions[68].target_tokens == 50000
    @test decisions[68].target_tokens < ctx
    @test isapprox(decisions[68].compaction_cost_s, 173.728; atol=1e-6)
    @test isapprox(decisions[68].accumulated_slowdown_s, 68 * 2.56; atol=1e-6)

    # The caller clears the rent after compacting, so the next turn waits again.
    reset!(tracker, 50000)
    d = decide_compaction(policy, budget, rent_cm, tracker, 50000 + 256)
    @test !d.compact
    @test d.reason === :below_target || d.reason === :wait

    # Hard ceiling: 96000 + 4096 > 100000 overflows, so it fires under every rule.
    for rule in (:rent_or_buy, :threshold, :never)
        h = decide_compaction(CompactionPolicy(rule=rule), budget, rent_cm,
            CompactionTracker(), 96000)
        @test h.compact
        @test h.reason === :hard_ceiling
        @test h.target_tokens == 50000
    end
    # just under the ceiling, :never stays put
    never = decide_compaction(CompactionPolicy(rule=:never), budget, rent_cm,
        CompactionTracker(), 95000)
    @test !never.compact
    @test never.reason === :never

    # Threshold: 0.9 * 100000 = 90000 exactly. The boundary fires; one below waits.
    thr = CompactionPolicy(rule=:threshold)
    below = decide_compaction(thr, budget, rent_cm, CompactionTracker(), 89999)
    at = decide_compaction(thr, budget, rent_cm, CompactionTracker(), 90000)
    @test !below.compact && below.reason === :wait
    @test at.compact && at.reason === :threshold
    @test at.target_tokens == 50000 && at.target_tokens < 90000

    # Below target: a context already at 10000 has nothing to give back.
    bt = decide_compaction(policy, budget, rent_cm, CompactionTracker(), 10000)
    @test !bt.compact && bt.reason === :below_target

    # Cannot fit: at 50000 (= target) plus a 60000-token turn overflows, and
    # compacting would not shrink the context, so the reply budget must shrink.
    cf = decide_compaction(policy, budget, rent_cm, CompactionTracker(), 50000;
        next_turn_tokens=60000)
    @test !cf.compact && cf.reason === :cannot_fit

    # Red control: decode cost flat in length, so rent never accrues. Rent-or-buy
    # must not compact before the hard ceiling, which is where 96000 first overflows.
    flat_cm = CostModel(prefill=[0 => 0.002], decode=[0 => 0.01])
    tr = CompactionTracker()
    seen = [(ctx=c, d=decide_compaction(policy, budget, flat_cm, tr, c))
            for c in 0:1000:100000]
    @test !any(s -> s.d.reason === :rent_or_buy, seen)
    first_compact = findfirst(s -> s.d.compact, seen)
    @test seen[first_compact].ctx == 96000
    @test seen[first_compact].d.reason === :hard_ceiling
    @test all(s -> !s.d.compact || s.d.target_tokens < s.ctx, seen)
end
