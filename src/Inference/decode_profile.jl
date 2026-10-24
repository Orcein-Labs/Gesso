# Opt-in decode stage profiler (env GESSO_PROFILE=1, read once at load).
#
# Off (the default): each stage site is one read of a const Ref and one
# branch. Nothing is allocated and no backend sync is issued.
#
# On: wall time per named stage, accumulated only inside a token window.
# `reference_generate` opens a window per generated token, so prefill is
# never counted. GPU stages bracket their work with a backend sync so the
# time belongs to the stage that launched it. Host<->device copies, bytes
# and page faults are counted where the code performs them.
#
# Stages are disjoint and summed into the token total; `other` is what is
# left. Sub-stages (`DECODE_SUBSTAGES`) and counters are nested inside a
# stage and are reported beside the table, never added to it.

const DECODE_STAGES = (
    :embed,
    :gdn_mixer,
    :attn_mixer,
    :norm,
    :dense_ffn,
    :moe_router,
    :moe_alloc,
    :moe_qupload,
    :moe_qgemv,
    :moe_d2h,
    :moe_host,
    :moe_h2d,
    :shared_expert,
    :lm_head,
    :sample,
)

# GDN mixer sub-stages (inside :gdn_mixer, nested like the host sub-stages).
const DECODE_GDN_SUBSTAGES = (
    :gdn_norm,
    :gdn_proj,
    :gdn_conv,
    :gdn_split,
    :gdn_gates,
    :gdn_qk,
    :gdn_recur,
    :gdn_gnorm,
    :gdn_outproj,
)

# Full-attention decode sub-stages (inside :attn_mixer).
const DECODE_ATTN_SUBSTAGES =
    (:attn_proj, :attn_headnorm_rope, :attn_kv_append, :attn_core, :attn_outproj)

# Dense FFN decode sub-stages (inside :dense_ffn).
const DECODE_FFN_SUBSTAGES = (:ffn_gateup, :ffn_act, :ffn_down)

const DECODE_SUBSTAGES = (
    :host_fork_join,
    :host_critical_path,
    :host_cpu_work,
    :upload_gather,
    DECODE_GDN_SUBSTAGES...,
    DECODE_ATTN_SUBSTAGES...,
    DECODE_FFN_SUBSTAGES...,
)

const DECODE_COUNTERS = (
    :expert_h2d_bytes,
    :expert_hits,
    :expert_misses,
    :cuda_alloc_calls,
    :cuda_alloc_bytes,
    :host_threaded_calls,
    :minflt,
    :majflt,
)

const TokenLogEntry =
    NamedTuple{(:ns, :minflt, :majflt, :decoded), Tuple{Int64, Int, Int, Bool}}

mutable struct DecodeProfile
    ns::Dict{Symbol, Int64}
    calls::Dict{Symbol, Int}
    sub_ns::Dict{Symbol, Int64}
    counters::Dict{Symbol, Int64}
    d2h::Int
    h2d::Int
    token_ns::Int64
    tokens::Int
    window::Bool
    fault0::Tuple{Int, Int}
    tok_log::Vector{TokenLogEntry}
    chunk_ns::Vector{Int64}
end

function DecodeProfile()
    ns = Dict{Symbol, Int64}(s => 0 for s in DECODE_STAGES)
    calls = Dict{Symbol, Int}(s => 0 for s in DECODE_STAGES)
    sub_ns = Dict{Symbol, Int64}(s => 0 for s in DECODE_SUBSTAGES)
    counters = Dict{Symbol, Int64}(s => 0 for s in DECODE_COUNTERS)
    return DecodeProfile(
        ns,
        calls,
        sub_ns,
        counters,
        0,
        0,
        0,
        0,
        false,
        (0, 0),
        TokenLogEntry[],
        Int64[],
    )
end

const _PROFILE = Ref{Union{Nothing, DecodeProfile}}(nothing)

function __init__()
    get(ENV, "GESSO_PROFILE", "") == "1" && profile_enable!()
    get(ENV, "GESSO_PREFILL_PROFILE", "") == "1" && prefill_profile_enable!()
    return nothing
end

"""
    profile_enable!()

Start a fresh decode profile. Stages and copies are recorded from the next
token window on.
"""
profile_enable!() = (_PROFILE[]=DecodeProfile(); nothing)

profile_disable!() = (_PROFILE[]=nothing; nothing)

function _profile_current()
    p = _PROFILE[]
    p === nothing &&
        error("decode profile is not enabled (set GESSO_PROFILE=1 or call profile_enable!)")
    return p
end

function profile_reset!()
    p = _profile_current()
    for s in DECODE_STAGES
        p.ns[s] = 0
        p.calls[s] = 0
    end
    for s in DECODE_SUBSTAGES
        p.sub_ns[s] = 0
    end
    for s in DECODE_COUNTERS
        p.counters[s] = 0
    end
    p.d2h = 0
    p.h2d = 0
    p.token_ns = 0
    p.tokens = 0
    empty!(p.tok_log)
    return nothing
end

function profile_snapshot()
    p = _profile_current()
    return DecodeProfile(
        copy(p.ns),
        copy(p.calls),
        copy(p.sub_ns),
        copy(p.counters),
        p.d2h,
        p.h2d,
        p.token_ns,
        p.tokens,
        false,
        (0, 0),
        copy(p.tok_log),
        Int64[],
    )
end

"""
    profile_delta(later, earlier) -> DecodeProfile

Field-wise difference of two snapshots. Used to cancel the head-only token
that a 1-token run contributes, so N decode tokens are measured exactly.
"""
function profile_delta(b::DecodeProfile, a::DecodeProfile)
    d = DecodeProfile()
    for s in DECODE_STAGES
        d.ns[s] = b.ns[s] - a.ns[s]
        d.calls[s] = b.calls[s] - a.calls[s]
    end
    for s in DECODE_SUBSTAGES
        d.sub_ns[s] = b.sub_ns[s] - a.sub_ns[s]
    end
    for s in DECODE_COUNTERS
        d.counters[s] = b.counters[s] - a.counters[s]
    end
    d.d2h = b.d2h - a.d2h
    d.h2d = b.h2d - a.h2d
    d.token_ns = b.token_ns - a.token_ns
    d.tokens = b.tokens - a.tokens
    return d
end

# Backend sync used to bracket GPU stages. Core has no device, so each GPU
# extension (CUDA, Lava) adds its own method.
_profile_sync(::Any) = nothing

@inline function _profile_add!(p::DecodeProfile, name::Symbol, dt::Int64)
    p.ns[name] += dt
    p.calls[name] += 1
    return nothing
end

"""
    @decode_stage :name backend expr

Evaluate `expr` and, when a profile window is open, add its wall time to
stage `:name`. `backend` is synced before and after the timed region; pass
`nothing` for host-only work. Returns the value of `expr`.
"""
macro decode_stage(name, backend, ex)
    nm = name isa QuoteNode ? name.value : name
    nm isa Symbol || error("@decode_stage: stage name must be a literal symbol, got $name")
    return quote
        local p = _PROFILE[]
        if p === nothing || !p.window
            $(esc(ex))
        else
            local b = $(esc(backend))
            _profile_sync(b)
            local t0 = time_ns()
            local r = $(esc(ex))
            _profile_sync(b)
            _profile_add!(p, $(QuoteNode(nm)), Int64(time_ns() - t0))
            r
        end
    end
end

"""
    @decode_sub :name backend expr

Like `@decode_stage`, but the time is added to the nested sub-stage `:name`
(reported beside the table, never summed into it).
"""
macro decode_sub(name, backend, ex)
    nm = name isa QuoteNode ? name.value : name
    nm isa Symbol ||
        error("@decode_sub: sub-stage name must be a literal symbol, got $name")
    return quote
        local p = _PROFILE[]
        if p === nothing || !p.window
            $(esc(ex))
        else
            local b = $(esc(backend))
            _profile_sync(b)
            local t0 = time_ns()
            local r = $(esc(ex))
            _profile_sync(b)
            _profile_sub!($(QuoteNode(nm)), Int64(time_ns() - t0))
            r
        end
    end
end

# Page faults of this process from /proc/self/stat (fields 10 minflt and
# 12 majflt). Zero where /proc is unavailable.
function _proc_faults()
    s = try
        read("/proc/self/stat", String)
    catch
        return (0, 0)
    end
    fields = split(s[(findlast(')', s)+2):end])
    length(fields) >= 10 || return (0, 0)
    return (parse(Int, fields[8]), parse(Int, fields[10]))
end

# Token window, opened by `reference_generate` around each generated token.
# Returns the start stamp, or zero when profiling is off.
@inline function _profile_token_begin()
    p = _PROFILE[]
    p === nothing && return UInt64(0)
    p.window = true
    p.fault0 = _proc_faults()
    return time_ns()
end

@inline function _profile_token_end(t0::UInt64, decoded::Bool)
    p = _PROFILE[]
    p === nothing && return nothing
    p.window = false
    ns = Int64(time_ns() - t0)
    f = _proc_faults()
    dmin = f[1] - p.fault0[1]
    dmaj = f[2] - p.fault0[2]
    p.token_ns += ns
    p.counters[:minflt] += dmin
    p.counters[:majflt] += dmaj
    push!(p.tok_log, (ns=ns, minflt=dmin, majflt=dmaj, decoded=decoded))
    decoded && (p.tokens += 1)
    return nothing
end

# Explicit host<->device copy: `d2h=true` for device-to-host.
@inline function _profile_copy!(d2h::Bool)
    p = _PROFILE[]
    (p === nothing || !p.window) && return nothing
    d2h ? (p.d2h += 1) : (p.h2d += 1)
    return nothing
end

@inline function _profile_count!(name::Symbol, n::Integer)
    p = _PROFILE[]
    (p === nothing || !p.window) && return nothing
    p.counters[name] += Int64(n)
    return nothing
end

@inline _profile_window_open() = (p=_PROFILE[]; p !== nothing && p.window)

@inline function _profile_sub!(name::Symbol, dt::Integer)
    p = _PROFILE[]
    (p === nothing || !p.window) && return nothing
    p.sub_ns[name] += Int64(dt)
    return nothing
end

"""
    print_decode_profile(io, p::DecodeProfile; label="")

Table of ms/token, share of the token total, and calls per token for each
stage, then `other` (token total minus the stages), then the nested
sub-stages, counters and copies per token.
"""
function print_decode_profile(io::IO, p::DecodeProfile; label::AbstractString="")
    p.tokens > 0 || error("print_decode_profile: no decode tokens in the profile")
    n = p.tokens
    tot_ms = p.token_ns / n / 1e6
    fmt(x; d=3) = string(round(x; digits=d))
    println(io, "decode profile $label  decode_tokens=$n  token_total_ms=$(fmt(tot_ms))")
    println(
        io,
        rpad("stage", 16),
        lpad("ms/token", 10),
        lpad("%", 8),
        lpad("calls/token", 13),
    )
    staged_ms = 0.0
    for s in DECODE_STAGES
        ms = p.ns[s] / n / 1e6
        staged_ms += ms
        println(
            io,
            rpad(string(s), 16),
            lpad(fmt(ms), 10),
            lpad(fmt(100 * ms / tot_ms; d=1), 8),
            lpad(fmt(p.calls[s] / n; d=2), 13),
        )
    end
    other_ms = tot_ms - staged_ms
    println(
        io,
        rpad("other", 16),
        lpad(fmt(other_ms), 10),
        lpad(fmt(100 * other_ms / tot_ms; d=1), 8),
    )
    println(io, rpad("TOTAL", 16), lpad(fmt(tot_ms), 10), lpad("100.0", 8))
    println(io, "nested (inside a stage above, not summed):")
    for s in DECODE_SUBSTAGES
        ms = p.sub_ns[s] / n / 1e6
        println(
            io,
            rpad("  " * string(s), 22),
            lpad(fmt(ms), 10),
            lpad(fmt(100 * ms / tot_ms; d=1), 8),
        )
    end
    println(io, "counters/token:")
    for s in DECODE_COUNTERS
        println(io, rpad("  " * string(s), 22), lpad(fmt(p.counters[s] / n; d=1), 14))
    end
    println(
        io,
        "copies/token  d2h=$(fmt(p.d2h / n; d=2))  h2d=$(fmt(p.h2d / n; d=2))  total=$(fmt((p.d2h + p.h2d) / n; d=2))",
    )
    return nothing
end

"""
    print_token_log(io, p::DecodeProfile)

One line per token window in the order they ran: wall ms, minor and major
page faults during the window, and whether the token ran a decode step.
"""
function print_token_log(io::IO, p::DecodeProfile)
    println(io, "token  ms        minflt   majflt  decoded")
    for (i, e) in enumerate(p.tok_log)
        println(
            io,
            lpad(string(i), 5),
            lpad(string(round(e.ns / 1e6; digits=2)), 10),
            lpad(string(e.minflt), 9),
            lpad(string(e.majflt), 9),
            lpad(string(e.decoded), 9),
        )
    end
    return nothing
end

# Prefill profiler (env GESSO_PREFILL_PROFILE=1, read once at load).
#
# Each prefill pass is its own window. The stage macros above accumulate into
# `_PROFILE[]`, so a pass swaps that Ref to a private DecodeProfile for its
# duration and restores the previous value on close. The off path is therefore
# the same single Ref read as a decode stage site, and a decode profile enabled
# alongside this one never sees prefill time.
#
# Stages are the prefill-relevant subset of DECODE_STAGES. `other` is the pass
# wall minus the stage sum: the norms, embedding, residual adds and allocation
# that the pass runs outside any stage, plus any GPU tail that a `nothing`
# backend stage did not wait for.

const PREFILL_STAGES = (
    :attn_mixer,
    :gdn_mixer,
    :moe_router,
    :moe_alloc,
    :moe_qupload,
    :moe_qgemv,
    :moe_host,
    :moe_h2d,
    :moe_d2h,
    :shared_expert,
    :dense_ffn,
)

const PREFILL_STAGE_NS = NamedTuple{PREFILL_STAGES, NTuple{length(PREFILL_STAGES), Int64}}

struct PrefillPassRecord
    pass::Int
    s0::Int
    n::Int
    K::Int
    wall_ns::Int64
    stage_ns::PREFILL_STAGE_NS
    other_ns::Int64
    h2d_bytes::Int64
end

mutable struct PrefillProfile
    dp::DecodeProfile
    passes::Vector{PrefillPassRecord}
    done::Bool
end

struct PrefillPassToken
    pp::PrefillProfile
    backend::Any
    prev::Union{Nothing, DecodeProfile}
    t0::UInt64
    s0::Int
    n::Int
    P::Int
end

const _PREFILL_PROFILE = Ref{Union{Nothing, PrefillProfile}}(nothing)

"""
    prefill_profile_enable!()

Start a fresh prefill profile. Each prefill pass run after this call records
one `PrefillPassRecord` and prints one PFPASS line; the last pass prints PFSUM.
"""
prefill_profile_enable!() = (
    _PREFILL_PROFILE[]=PrefillProfile(DecodeProfile(), PrefillPassRecord[], false);
    nothing
)

prefill_profile_disable!() = (_PREFILL_PROFILE[]=nothing; nothing)

"""
    prefill_profile_records() -> Vector{PrefillPassRecord}

Copy of the records from the most recent prefill, empty when profiling is off.
"""
prefill_profile_records() =
    (pp=_PREFILL_PROFILE[]; pp === nothing ? PrefillPassRecord[] : copy(pp.passes))

# Called once per pass by `reference_generate`. Off: one Ref read and a branch,
# and `nothing` is returned so nothing is allocated.
@inline function _prefill_pass_begin(backend, s0::Int, n::Int, P::Int)
    pp = _PREFILL_PROFILE[]
    pp === nothing && return nothing
    return _prefill_pass_open(pp, backend, s0, n, P)
end

@inline function _prefill_pass_end(tok)
    tok === nothing && return nothing
    return _prefill_pass_close(tok)
end

function _prefill_clear!(dp::DecodeProfile)
    for s in DECODE_STAGES
        dp.ns[s] = 0
        dp.calls[s] = 0
    end
    for s in DECODE_SUBSTAGES
        dp.sub_ns[s] = 0
    end
    for s in DECODE_COUNTERS
        dp.counters[s] = 0
    end
    dp.d2h = 0
    dp.h2d = 0
    return nothing
end

function _prefill_pass_open(pp::PrefillProfile, backend, s0::Int, n::Int, P::Int)
    # A prefill that finished, or one that starts at row 1, begins a new record
    # list. A resumed prefill (s0 > 1, after a prefix snapshot) follows the
    # pass it resumes on from, so its records join the same list.
    if pp.done || s0 == 1
        empty!(pp.passes)
        pp.done = false
    end
    dp = pp.dp
    _prefill_clear!(dp)
    # Drain work queued by earlier stages so the wall starts at the pass boundary
    _profile_sync(backend)
    prev = _PROFILE[]
    # A pass that threw left the prefill profile installed; never save it as the
    # decode profile to restore
    prev === dp && (prev = nothing)
    _PROFILE[] = dp
    dp.window = true
    return PrefillPassToken(pp, backend, prev, time_ns(), s0, n, P)
end

function _prefill_pass_close(tok::PrefillPassToken)
    pp = tok.pp
    dp = pp.dp
    _profile_sync(tok.backend)
    wall = Int64(time_ns() - tok.t0)
    dp.window = false
    _PROFILE[] = tok.prev
    stage_ns =
        PREFILL_STAGE_NS(ntuple(i -> dp.ns[PREFILL_STAGES[i]], length(PREFILL_STAGES)))
    other = wall - sum(values(stage_ns))
    K = tok.s0 + tok.n - 1
    rec = PrefillPassRecord(
        length(pp.passes) + 1,
        tok.s0,
        tok.n,
        K,
        wall,
        stage_ns,
        other,
        dp.counters[:expert_h2d_bytes],
    )
    push!(pp.passes, rec)
    _print_prefill_pass(stdout, rec)
    if K == tok.P
        pp.done = true
        _print_prefill_summary(stdout, pp)
    end
    return nothing
end

_prefill_ms(ns::Integer) = round(ns / 1e6; digits=3)

function _print_prefill_pass(io::IO, r::PrefillPassRecord)
    print(
        io,
        "PFPASS i=",
        r.pass,
        " n=",
        r.n,
        " K=",
        r.K,
        " wall_ms=",
        _prefill_ms(r.wall_ns),
    )
    for s in PREFILL_STAGES
        print(io, " ", s, "=", _prefill_ms(getfield(r.stage_ns, s)))
    end
    print(io, " other=", _prefill_ms(r.other_ns))
    println(
        io,
        " h2d_mb=",
        round(r.h2d_bytes / 2^20; digits=2),
        " h2d_ms=",
        _prefill_ms(getfield(r.stage_ns, :moe_h2d)),
    )
    flush(io)
    return nothing
end

function _print_prefill_summary(io::IO, pp::PrefillProfile)
    rows = sum(r.n for r in pp.passes; init=0)
    wall = sum(r.wall_ns for r in pp.passes; init=0)
    print(
        io,
        "PFSUM passes=",
        length(pp.passes),
        " rows=",
        rows,
        " wall_ms=",
        _prefill_ms(wall),
    )
    for s in PREFILL_STAGES
        print(
            io,
            " ",
            s,
            "=",
            _prefill_ms(sum(getfield(r.stage_ns, s) for r in pp.passes; init=0)),
        )
    end
    other = sum(r.other_ns for r in pp.passes; init=0)
    h2d = sum(r.h2d_bytes for r in pp.passes; init=0)
    println(io, " other=", _prefill_ms(other), " h2d_mb=", round(h2d / 2^20; digits=2))
    flush(io)
    return nothing
end
