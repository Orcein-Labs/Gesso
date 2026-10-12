# Context checkpoints on a hybrid model: a conversation that shares only a
# system prompt with a parked one resumes at the newest checkpoint inside the
# shared rows, and splitting prefill at the stride boundaries changes nothing.

const _CK_SYS = [1, 3, 4, 5, 2, 6, 7]
const _CK_CONV_A = [8, 9, 10]
const _CK_CONV_B = [11, 12]

function _with_env(f, pairs::Pair{String,String}...)
    saved = [k => get(ENV, k, nothing) for (k, _) in pairs]
    try
        for (k, v) in pairs
            ENV[k] = v
        end
        return f()
    finally
        for (k, v) in saved
            v === nothing ? delete!(ENV, k) : (ENV[k] = v)
        end
    end
end

function _ck_run(ts, prompt; every::Int, window::Bool)
    st = GenerateState(64)
    g = _with_env("GESSO_CTX_CHECKPOINT_EVERY" => string(every),
        "GESSO_PREFILL_WINDOW" => window ? "1" : "0") do
        reference_generate(ts.model, ts, prompt; max_new_tokens=3, state=st)
    end
    return (; st, g)
end

_ck_positions(st) = [c.pos for c in st.ckpts]

@testset "ctx checkpoints: a new conversation resumes at the newest checkpoint in the shared prefix" begin
    ts = _hybrid_toy_tensors(toy2_tensors())
    @test size(ts.embedding.storage, 1) > maximum(_CK_CONV_B)
    A = [_CK_SYS; _CK_CONV_A]
    B = [_CK_SYS; _CK_CONV_B]
    _with_env("GESSO_CTX_CHECKPOINTS" => "8", "GESSO_CTX_CHECKPOINT_EVERY" => "3") do
        st = GenerateState(64)
        fresh = GenerateState(64)
        try
            reference_generate(ts.model, ts, A; max_new_tokens=3, state=st)
            @test _ck_positions(st) == [3, 6, 9, 10]
            info = Ref{Any}(nothing)
            gB = reference_generate(ts.model, ts, B; max_new_tokens=3, state=st, info)
            # the shared rows are 1:7, so the checkpoint at 6 is the newest usable one
            @test info[].reused == 6
            @test _ck_positions(st) == [3, 6, 9]
            gF = reference_generate(ts.model, ts, B; max_new_tokens=3, state=fresh)
            @test gB == gF
            @test _vw_state_diff(st, fresh) == String[]
        finally
            close(st)
            close(fresh)
        end
    end
end

@testset "ctx checkpoints: GESSO_CTX_CHECKPOINTS=0 keeps only the end-of-prompt snapshot" begin
    ts = _hybrid_toy_tensors(toy2_tensors())
    A = [_CK_SYS; _CK_CONV_A]
    B = [_CK_SYS; _CK_CONV_B]
    _with_env("GESSO_CTX_CHECKPOINTS" => "0", "GESSO_CTX_CHECKPOINT_EVERY" => "3") do
        st = GenerateState(64)
        try
            reference_generate(ts.model, ts, A; max_new_tokens=3, state=st)
            @test _ck_positions(st) == [10]
            info = Ref{Any}(nothing)
            reference_generate(ts.model, ts, B; max_new_tokens=3, state=st, info)
            @test info[].reused == 0
        finally
            close(st)
        end
    end
end

@testset "ctx checkpoints: a full slot set keeps the oldest and the newest" begin
    ts = _hybrid_toy_tensors(toy2_tensors())
    A = [_CK_SYS; _CK_CONV_A]
    _with_env("GESSO_CTX_CHECKPOINTS" => "2", "GESSO_CTX_CHECKPOINT_EVERY" => "3") do
        st = GenerateState(64)
        try
            reference_generate(ts.model, ts, A; max_new_tokens=3, state=st)
            @test _ck_positions(st) == [3, 10]
        finally
            close(st)
        end
    end
end

# Split prefill must leave the same state as the unsplit pass, on the per-row
# path and on the verify window (the window needs the Float32 fast path).
@testset "ctx checkpoints: stride-split prefill equals unsplit prefill" for window in (false, true)
    ts = window ? _vw_hybrid_f32() : _hybrid_toy_tensors(toy2_tensors())
    prompt = [_CK_SYS; _CK_CONV_A]
    split = _ck_run(ts, prompt; every=3, window)
    whole = _ck_run(ts, prompt; every=0, window)
    try
        @test _ck_positions(split.st) == [3, 6, 9, 10]
        @test split.g == whole.g
        @test _vw_state_diff(split.st, whole.st) == String[]
    finally
        close(split.st)
        close(whole.st)
    end
end
