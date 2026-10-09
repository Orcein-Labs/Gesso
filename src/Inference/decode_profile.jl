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
    :embed, :gdn_mixer, :attn_mixer, :norm, :dense_ffn,
    :moe_router, :moe_alloc, :moe_qupload, :moe_qgemv,
    :moe_d2h, :moe_host, :moe_h2d, :shared_expert,
    :lm_head, :sample,
)

# GDN mixer sub-stages (inside :gdn_mixer, nested like the host sub-stages).
const DECODE_GDN_SUBSTAGES = (
    :gdn_norm, :gdn_proj, :gdn_conv, :gdn_split, :gdn_gates,
    :gdn_qk, :gdn_recur, :gdn_gnorm, :gdn_outproj,
)

const DECODE_SUBSTAGES = (
    :host_fork_join, :host_critical_path, :host_cpu_work, :upload_gather,
    DECODE_GDN_SUBSTAGES...,
)

const DECODE_COUNTERS = (
    :expert_h2d_bytes, :expert_hits, :expert_misses,
    :cuda_alloc_calls, :cuda_alloc_bytes,
    :host_threaded_calls, :minflt, :majflt,
)

const TokenLogEntry = NamedTuple{(:ns, :minflt, :majflt, :decoded),Tuple{Int64,Int,Int,Bool}}

mutable struct DecodeProfile
    ns::Dict{Symbol,Int64}
    calls::Dict{Symbol,Int}
    sub_ns::Dict{Symbol,Int64}
    counters::Dict{Symbol,Int64}
    d2h::Int
    h2d::Int
    token_ns::Int64
    tokens::Int
    window::Bool
    fault0::Tuple{Int,Int}
    tok_log::Vector{TokenLogEntry}
    chunk_ns::Vector{Int64}
end

function DecodeProfile()
    ns = Dict{Symbol,Int64}(s => 0 for s in DECODE_STAGES)
    calls = Dict{Symbol,Int}(s => 0 for s in DECODE_STAGES)
    sub_ns = Dict{Symbol,Int64}(s => 0 for s in DECODE_SUBSTAGES)
    counters = Dict{Symbol,Int64}(s => 0 for s in DECODE_COUNTERS)
    return DecodeProfile(
        ns, calls, sub_ns, counters, 0, 0, 0, 0, false, (0, 0),
        TokenLogEntry[], Int64[],
    )
end

const _PROFILE = Ref{Union{Nothing,DecodeProfile}}(nothing)

function __init__()
    get(ENV, "GESSO_PROFILE", "") == "1" && profile_enable!()
    return nothing
end

"""
    profile_enable!()

Start a fresh decode profile. Stages and copies are recorded from the next
token window on.
"""
profile_enable!() = (_PROFILE[] = DecodeProfile(); nothing)

profile_disable!() = (_PROFILE[] = nothing; nothing)

function _profile_current()
    p = _PROFILE[]
    p === nothing && error("decode profile is not enabled (set GESSO_PROFILE=1 or call profile_enable!)")
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
        copy(p.ns), copy(p.calls), copy(p.sub_ns), copy(p.counters),
        p.d2h, p.h2d, p.token_ns, p.tokens, false, (0, 0),
        copy(p.tok_log), Int64[],
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

# Backend sync used to bracket GPU stages. Core has no device, so the
# extension adds the CUDA methods; every other backend needs none.
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
    nm isa Symbol || error("@decode_sub: sub-stage name must be a literal symbol, got $name")
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
    fields = split(s[(findlast(')', s) + 2):end])
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

@inline _profile_window_open() = (p = _PROFILE[]; p !== nothing && p.window)

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
    println(io, rpad("stage", 16), lpad("ms/token", 10), lpad("%", 8), lpad("calls/token", 13))
    staged_ms = 0.0
    for s in DECODE_STAGES
        ms = p.ns[s] / n / 1e6
        staged_ms += ms
        println(io, rpad(string(s), 16), lpad(fmt(ms), 10), lpad(fmt(100 * ms / tot_ms; d=1), 8), lpad(fmt(p.calls[s] / n; d=2), 13))
    end
    other_ms = tot_ms - staged_ms
    println(io, rpad("other", 16), lpad(fmt(other_ms), 10), lpad(fmt(100 * other_ms / tot_ms; d=1), 8))
    println(io, rpad("TOTAL", 16), lpad(fmt(tot_ms), 10), lpad("100.0", 8))
    println(io, "nested (inside a stage above, not summed):")
    for s in DECODE_SUBSTAGES
        ms = p.sub_ns[s] / n / 1e6
        println(io, rpad("  " * string(s), 22), lpad(fmt(ms), 10), lpad(fmt(100 * ms / tot_ms; d=1), 8))
    end
    println(io, "counters/token:")
    for s in DECODE_COUNTERS
        println(io, rpad("  " * string(s), 22), lpad(fmt(p.counters[s] / n; d=1), 14))
    end
    println(io, "copies/token  d2h=$(fmt(p.d2h / n; d=2))  h2d=$(fmt(p.h2d / n; d=2))  total=$(fmt((p.d2h + p.h2d) / n; d=2))")
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
        println(io, lpad(string(i), 5), lpad(string(round(e.ns / 1e6; digits=2)), 10),
            lpad(string(e.minflt), 9), lpad(string(e.majflt), 9), lpad(string(e.decoded), 9))
    end
    return nothing
end
