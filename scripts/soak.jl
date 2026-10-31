# Soak client for the OpenAI-compatible chat server.
#
#   julia --project scripts/soak.jl --port 8080 --duration 600 --clients 4 --pid <server pid>
#
# Drives a running server with a fixed mix of requests (short and long prompts,
# streaming and not, repeated greedy prompts, multi-turn conversations,
# disconnects mid-stream, malformed requests, oversize bodies and prompts over
# the context limit) and checks every reply. One JSON line per event goes to --log (default soak-events.jsonl); a
# summary is printed at the end. The exit code is 1 when any check failed.
#
# Checks: every reply is well formed for its type; errors carry the documented
# error object; a greedy prompt returns the same text every time, streamed or
# not; /health answers throughout; the server's resident memory (read from
# /proc/<pid>/status when --pid is given) and, with --vram, its GPU memory do
# not keep growing; the p99 time-to-first-token does not trend upward. Memory
# and latency are judged only on windows that start after --warmup seconds.
#
# A conversation sends its whole history every turn, greedy, and stops early
# (not a failure) before the history could reach n_ctx. With probability
# --replay-frac a turn is re-sent unchanged on a new connection and the two
# replies are compared. An unrelated one-token request goes first, so the replay
# is prefilled from scratch rather than served from the prompt cache it would
# otherwise hit whole; a replay that still reports a long cached prefix is
# counted as not fresh. Greedy output can legitimately flip at a near-tie when
# the cached rows came from decode instead of prefill, so a divergence is
# counted and reported (among all replays and among the drift-eligible ones: fresh
# replays of turns whose original was itself served from a cached prefix);
# --replay-strict makes it a failure. scripts/soak_margin.jl judges the logged
# divergences offline.
#
# Conversation turns, replays and the evicting request send enable_thinking=false:
# a thinking model spends the small token budget on reasoning, so the reply has no
# content and the conversation would end at turn 1.
#
# Two kinds call a tool (get_weather), streamed and not: the call is validated
# (id, type, name, JSON-object arguments with a string city) and answered with a
# tool message, and the follow-up reply is checked too. A model that answers in
# text instead of calling is counted as no_call, not a failure. A server whose
# chat format has no tool-call parser answers the start-up probe with a 400; the
# tool kinds are then switched off with a note.
#
# Every --metrics-interval seconds a "metrics" event gives the window's request
# and error counts, p50/p99 time-to-first-token and mean decode tok/s of the
# streamed requests, the latest RSS and, with --vram, GPU memory used.
#
# With several clients the server's one prompt-cache slot rarely still holds a
# conversation when its next turn arrives, so originals run uncached and no
# replay can show cache-origin drift. --solo-every S gives every S seconds a
# window of --solo-len seconds in which client 1 runs only conversations and the
# others pause, ending any conversation of theirs at its next turn. The windows
# carry no cancellations, so keep them short next to the mixed time.
#
# Options that take a value: --host --port --duration --clients --pid --seed
#   --log --request-timeout --health-timeout --health-interval --rss-interval
#   --rss-growth-mb --rss-growth-frac --long-words --max-turns --replay-frac
#   --metrics-interval --warmup --p99-growth-ms --p99-growth-frac
#   --vram-growth-mb --solo-every --solo-len
# Flags: --vram (sample nvidia-smi) --replay-strict
#
# The module is also used by test/test_soak.jl, so the client cannot rot.

module GessoSoak

using Sockets
using JSON
using Random

Base.@kwdef struct Config
    host::String = "127.0.0.1"
    port::Int = 8080
    duration::Float64 = 60.0            # seconds in which new requests start
    clients::Int = 4
    pid::Union{Nothing, Int} = nothing  # server process, for RSS sampling
    seed::Int = 1
    request_timeout::Float64 = 120.0
    health_timeout::Float64 = 10.0      # a slower /health answer is a failure
    health_interval::Float64 = 1.0
    rss_interval::Float64 = 10.0
    rss_growth_mb::Float64 = 64.0       # growth must exceed both of these
    rss_growth_frac::Float64 = 0.15     # to be flagged
    long_words::Int = 300
    max_turns::Int = 30                 # a conversation has 2..max_turns turns
    replay_frac::Float64 = 0.25         # share of conversation turns replayed
    replay_strict::Bool = false         # a replay divergence is a failure
    metrics_interval::Float64 = 60.0
    warmup::Float64 = 300.0             # windows starting earlier are not judged
    p99_growth_ms::Float64 = 2000.0     # p99 TTFT rise must exceed both of these
    p99_growth_frac::Float64 = 0.5      # (the fraction is of the first window's p99)
    vram::Bool = false                  # sample GPU memory with nvidia-smi
    vram_growth_mb::Float64 = 256.0
    solo_every::Float64 = 0.0           # period of the single-client windows; 0 turns them off
    solo_len::Float64 = 600.0
end

# Whether `elapsed` seconds after the clients start falls in a single-client
# window. A window opens each period, so a run starts with one.
in_solo(cfg::Config, elapsed) = cfg.solo_every > 0 && mod(elapsed, cfg.solo_every) < cfg.solo_len

# Request kinds and their weights in the mix.
const MIX = [
    "short" => 20,
    "short_stream" => 20,
    "long" => 4,
    "long_stream" => 4,
    "greedy" => 12,
    "greedy_stream" => 8,
    "disconnect" => 8,
    "conversation" => 2,
    "conversation_stream" => 2,
    "bad_request" => 8,
    "oversize_body" => 2,
    "over_context" => 4,
    "keepalive" => 4,
    "tool_call" => 1,
    "tool_call_stream" => 1,
]

const TOOL_KINDS = ("tool_call", "tool_call_stream")

# Weights are per pick, but a conversation is many requests. The plan wants
# about a tenth of streamed requests cancelled mid-stream; this is the share the
# weights give for a conversation of the mean length, replays included.
function cancel_share(cfg::Config)
    w = Dict(MIX)
    mean_turns = (2 + cfg.max_turns) / 2
    # a streamed tool turn is two streams when the model calls the tool
    streamed = w["short_stream"] + w["long_stream"] + w["greedy_stream"] + w["disconnect"] +
               2 * w["tool_call_stream"] +
               w["conversation_stream"] * mean_turns * (1 + cfg.replay_frac)
    return w["disconnect"] / streamed
end

const GREEDY_PROMPTS = ["Count from one to ten in words.", "Name three primary colors."]
const FINISH_REASONS = ("stop", "length", "tool_calls")

# Two unrelated conversations still share the chat template's header, so a
# replay that reports at most this many cached tokens was prefilled from scratch.
const FRESH_CACHED_MAX = 16

mutable struct Failure
    kind::String
    reason::String
    request::String
end

mutable struct Ctx
    cfg::Config
    n_ctx::Int
    log::IO
    log_lock::ReentrantLock
    t0::Float64
    latencies::Dict{String, Vector{Float64}}
    counts::Dict{String, Int}
    failed::Dict{String, Int}
    failures::Vector{Failure}
    baselines::Dict{String, NamedTuple}
    rss::Vector{Tuple{Float64, Int}}    # (seconds since start, kB)
    rss_peak_kb::Int
    health_max::Float64
    stop::Ref{Bool}
    done::Int                               # requests recorded, health probes excluded
    stream_obs::Vector{NTuple{3, Float64}}  # (seconds since start, TTFT ms, decode tok/s or NaN)
    windows::Vector{NamedTuple}
    vram::Vector{Tuple{Float64, Int}}       # (seconds since start, kB), the shape rss_verdict reads
    replay_n::Int
    replay_offsets::Vector{Int}             # first differing character of each divergence
    windows_prev::NTuple{3, Int}            # done, failures, stream_obs at the last window close
    replay_fresh::Int                       # replays whose cached prefix was at most FRESH_CACHED_MAX
    replay_eligible::Int                    # fresh, and the original was served from a longer cached prefix
    replay_eligible_diverged::Int
    tools_on::Bool                          # false once the start-up probe finds no tool support
    tool_note::String
    tool_turns::Int                         # tool requests whose reply passed the checks
    tool_calls::Int                         # of those, calls the model made
    tool_no_call::Int                       # of those, answered in text instead
    started::Float64                        # when the clients started; the single-client windows count from it
end

in_solo(ctx::Ctx) = in_solo(ctx.cfg, time() - ctx.started)

# Wall-clock marks of one streamed exchange.
mutable struct Timing
    t_write::Float64
    t_first::Float64    # first content or reasoning chunk; NaN until it arrives
    t_end::Float64
end
Timing() = Timing(NaN, NaN, NaN)

# ---------------------------------------------------------------------------
# Logging and bookkeeping

function emit(ctx::Ctx, event::AbstractString; fields...)
    d = Dict{String, Any}("t" => round(time() - ctx.t0; digits=3), "event" => event)
    for (k, v) in fields
        d[String(k)] = v
    end
    lock(ctx.log_lock) do
        println(ctx.log, JSON.json(d))
        flush(ctx.log)
    end
    return nothing
end

function record!(ctx::Ctx, kind, ms, reason, status, request)
    lock(ctx.log_lock) do
        push!(get!(ctx.latencies, kind, Float64[]), ms)
        ctx.counts[kind] = get(ctx.counts, kind, 0) + 1
        kind == "health" || (ctx.done += 1)
        if reason !== nothing
            ctx.failed[kind] = get(ctx.failed, kind, 0) + 1
            push!(ctx.failures, Failure(kind, reason, request))
        end
    end
    fields = (kind=kind, ms=round(ms; digits=1), status=status, ok=reason === nothing)
    reason === nothing ? emit(ctx, "request"; fields...) :
    emit(ctx, "request"; fields..., reason=reason, request=request)
    return nothing
end

function fail!(ctx::Ctx, kind, reason, request="")
    lock(ctx.log_lock) do
        ctx.counts[kind] = get(ctx.counts, kind, 0) + 1
        ctx.failed[kind] = get(ctx.failed, kind, 0) + 1
        push!(ctx.failures, Failure(kind, reason, request))
    end
    emit(ctx, "check"; kind=kind, ok=false, reason=reason, request=request)
    return nothing
end

# Nearest-rank percentile of a sorted vector.
pct(sorted, p) = isempty(sorted) ? NaN : sorted[clamp(ceil(Int, p * length(sorted)), 1, length(sorted))]

# ---------------------------------------------------------------------------
# A small HTTP/1.1 client. Reads exactly one response, by Content-Length or by
# chunked framing, so a connection can be reused for the next request.

struct Response
    status::Int
    headers::Dict{String, String}
    body::Vector{UInt8}
    chunks::Vector{String}      # decoded chunk payloads, when chunked
    aborted::Bool               # on_chunk asked to stop before the end
end

struct WireError <: Exception
    msg::String
end

function read_response(sock::TCPSocket; on_chunk=nothing)
    status_line = readline(sock)
    isempty(status_line) && throw(WireError("connection closed before a response"))
    m = match(r"^HTTP/1\.[01] (\d{3}) ", status_line)
    m === nothing && throw(WireError("bad status line: $(first(status_line, 80))"))
    status = parse(Int, m.captures[1])
    headers = Dict{String, String}()
    while true
        line = rstrip(readline(sock), '\r')
        isempty(line) && break
        colon = findfirst(':', line)
        colon === nothing && throw(WireError("malformed header: $(first(line, 80))"))
        headers[lowercase(strip(line[1:(colon-1)]))] = strip(line[(colon+1):end])
    end
    status == 100 && return read_response(sock; on_chunk=on_chunk)
    chunks = String[]
    body = UInt8[]
    if occursin("chunked", lowercase(get(headers, "transfer-encoding", "")))
        while true
            size_line = rstrip(readline(sock), '\r')
            n = tryparse(Int, strip(size_line); base=16)
            n === nothing && throw(WireError("bad chunk size '$(first(size_line, 40))'"))
            if n == 0
                readline(sock)
                break
            end
            data = read(sock, n)
            length(data) == n || throw(WireError("chunk cut short"))
            rstrip(readline(sock), '\r') == "" || throw(WireError("chunk not followed by CRLF"))
            text = String(data)
            push!(chunks, text)
            if on_chunk !== nothing && on_chunk(text) === false
                return Response(status, headers, body, chunks, true)
            end
        end
    elseif haskey(headers, "content-length")
        n = tryparse(Int, headers["content-length"])
        n === nothing && throw(WireError("bad Content-Length"))
        body = read(sock, n)
        length(body) == n || throw(WireError("body shorter than Content-Length"))
    else
        body = read(sock)
    end
    return Response(status, headers, body, chunks, false)
end

function build_request(method, target, body::Union{Nothing, AbstractString}; close=true, length_override=nothing)
    io = IOBuffer()
    write(io, method, " ", target, " HTTP/1.1\r\nHost: soak\r\n")
    write(io, "Connection: ", close ? "close" : "keep-alive", "\r\n")
    if body !== nothing
        n = length_override === nothing ? ncodeunits(body) : length_override
        write(io, "Content-Type: application/json\r\nContent-Length: ", string(n), "\r\n")
    end
    write(io, "\r\n")
    body === nothing || write(io, body)
    return String(take!(io))
end

# True when a stream chunk carries generated text, as opposed to the role
# chunk, the finish chunk or the usage chunk.
function has_text(chunk::AbstractString)
    for e in split(chunk, "\n\n"; keepempty=false)
        startswith(e, "data: {") || continue
        j = parse_json(Vector{UInt8}(codeunits(SubString(e, 7))))
        j isa AbstractDict || continue
        ch = get(j, "choices", nothing)
        (ch isa AbstractVector && !isempty(ch)) || continue
        delta = get(ch[1], "delta", nothing)
        delta isa AbstractDict || continue
        for k in ("content", "reasoning_content")
            v = get(delta, k, nothing)
            v isa AbstractString && !isempty(v) && return true
        end
    end
    return false
end

# One request on a fresh connection under the request timeout. A timer closes the
# socket when the deadline passes, which fails the blocked read.
function exchange(cfg::Config, raw::AbstractString; on_chunk=nothing, timing::Union{Nothing, Timing}=nothing)
    sock = Sockets.connect(Sockets.IPv4(cfg.host), cfg.port)
    timer = Timer(cfg.request_timeout) do _
        close(sock)
    end
    if timing !== nothing
        inner = on_chunk
        on_chunk = text -> begin
            isnan(timing.t_first) && has_text(text) && (timing.t_first = time())
            return inner === nothing ? true : inner(text)
        end
    end
    try
        timing === nothing || (timing.t_write = time())
        write(sock, raw)
        r = read_response(sock; on_chunk=on_chunk)
        timing === nothing || (timing.t_end = time())
        return r
    finally
        close(timer)
        close(sock)
    end
end

# ---------------------------------------------------------------------------
# Reply checks. Each returns nothing when the reply is right, else a reason.

function parse_json(bytes::AbstractVector{UInt8})
    try
        return JSON.parse(String(copy(bytes)))
    catch
        return nothing
    end
end

# The error object every non-2xx reply must carry.
function check_error(r::Response, status::Int; code=nothing)
    r.status == status || return "expected status $status, got $(r.status)"
    j = parse_json(r.body)
    j isa AbstractDict || return "error reply is not a JSON object"
    err = get(j, "error", nothing)
    err isa AbstractDict || return "error reply has no error object"
    get(err, "message", nothing) isa AbstractString || return "error.message is not a string"
    get(err, "type", nothing) isa AbstractString || return "error.type is not a string"
    haskey(err, "code") || return "error.code is missing"
    code === nothing || get(err, "code", nothing) == code ||
        return "expected error.code $code, got $(get(err, "code", nothing))"
    startswith(get(r.headers, "content-type", ""), "application/json") ||
        return "error reply is not application/json"
    return nothing
end

function check_usage(u, max_tokens)
    u isa AbstractDict || return "usage is missing"
    p, c, t = get(u, "prompt_tokens", nothing), get(u, "completion_tokens", nothing), get(u, "total_tokens", nothing)
    all(x -> x isa Integer, (p, c, t)) || return "usage fields are not integers"
    t == p + c || return "usage total_tokens != prompt_tokens + completion_tokens"
    p > 0 || return "usage prompt_tokens is not positive"
    c <= max_tokens || return "completion_tokens $c exceeds max_tokens $max_tokens"
    return nothing
end

# Returns (reason, answer); answer is (reasoning, content, finish, tool_calls).
function check_completion(r::Response, max_tokens::Int)
    r.status == 200 || return "expected status 200, got $(r.status)", nothing
    startswith(get(r.headers, "content-type", ""), "application/json") ||
        return "reply is not application/json", nothing
    j = parse_json(r.body)
    j isa AbstractDict || return "reply is not a JSON object", nothing
    get(j, "object", nothing) == "chat.completion" || return "object is not chat.completion", nothing
    get(j, "id", nothing) isa AbstractString || return "id is not a string", nothing
    ch = get(j, "choices", nothing)
    (ch isa AbstractVector && length(ch) == 1) || return "choices is not a one-element array", nothing
    msg = get(ch[1], "message", nothing)
    msg isa AbstractDict || return "choice has no message", nothing
    get(msg, "role", nothing) == "assistant" || return "message.role is not assistant", nothing
    content = get(msg, "content", nothing)
    (content isa AbstractString || content === nothing) || return "message.content is not a string", nothing
    reasoning = get(msg, "reasoning_content", "")
    reasoning isa AbstractString || return "message.reasoning_content is not a string", nothing
    finish = get(ch[1], "finish_reason", nothing)
    finish in FINISH_REASONS || return "finish_reason '$finish' is not valid", nothing
    calls = get(msg, "tool_calls", nothing)
    (calls === nothing || calls isa AbstractVector) || return "message.tool_calls is not an array", nothing
    reason = check_usage(get(j, "usage", nothing), max_tokens)
    reason === nothing || return reason, nothing
    return nothing, (reasoning=String(reasoning), content=content === nothing ? "" : String(content),
        finish=String(finish), tool_calls=calls === nothing ? Any[] : Vector{Any}(calls))
end

function check_stream(r::Response, max_tokens::Int; include_usage::Bool=false)
    r.status == 200 || return "expected status 200, got $(r.status)", nothing
    get(r.headers, "content-type", "") == "text/event-stream" ||
        return "stream content-type is not text/event-stream", nothing
    r.aborted && return "stream stopped early", nothing
    events = split(join(r.chunks), "\n\n"; keepempty=false)
    isempty(events) && return "stream has no events", nothing
    all(e -> startswith(e, "data: "), events) || return "an event does not start with 'data: '", nothing
    payloads = [String(e[7:end]) for e in events]
    payloads[end] == "[DONE]" || return "stream does not end with [DONE]", nothing
    id = nothing
    finish = nothing
    usage = nothing
    reasoning, content = String[], String[]
    calls = Dict{Int, Dict{String, Any}}()      # by delta index; fragments are concatenated
    for p in payloads[1:(end-1)]
        j = parse_json(Vector{UInt8}(codeunits(p)))
        j isa AbstractDict || return "an event is not a JSON object", nothing
        get(j, "object", nothing) == "chat.completion.chunk" || return "chunk object is wrong", nothing
        id === nothing && (id = get(j, "id", nothing))
        get(j, "id", nothing) == id || return "chunk ids differ within one stream", nothing
        finish === nothing || get(j, "usage", nothing) !== nothing ||
            return "an event follows finish_reason", nothing
        if haskey(j, "usage") && j["usage"] !== nothing
            usage = j["usage"]
            continue
        end
        ch = get(j, "choices", nothing)
        (ch isa AbstractVector && length(ch) == 1) || return "chunk choices is not a one-element array", nothing
        delta = get(ch[1], "delta", nothing)
        delta isa AbstractDict || return "chunk has no delta", nothing
        c = get(delta, "content", nothing)
        c isa AbstractString && push!(content, c)
        rc = get(delta, "reasoning_content", nothing)
        rc isa AbstractString && push!(reasoning, rc)
        td = get(delta, "tool_calls", nothing)
        if td !== nothing
            td isa AbstractVector || return "delta.tool_calls is not an array", nothing
            for d in td
                d isa AbstractDict || return "a tool_calls delta is not an object", nothing
                idx = get(d, "index", nothing)
                idx isa Integer || return "a tool_calls delta has no integer index", nothing
                call = get!(calls, Int(idx), Dict{String, Any}("function" => Dict{String, Any}("arguments" => "")))
                haskey(d, "id") && (call["id"] = d["id"])
                haskey(d, "type") && (call["type"] = d["type"])
                fn = get(d, "function", nothing)
                if fn isa AbstractDict
                    haskey(fn, "name") && (call["function"]["name"] = fn["name"])
                    a = get(fn, "arguments", nothing)
                    a isa AbstractString && (call["function"]["arguments"] *= a)
                end
            end
        end
        f = get(ch[1], "finish_reason", nothing)
        if f !== nothing
            finish === nothing || return "two finish_reason values in one stream", nothing
            finish = f
        end
    end
    finish in FINISH_REASONS || return "stream finish_reason '$finish' is not valid", nothing
    if include_usage
        reason = check_usage(usage, max_tokens)
        reason === nothing || return reason, nothing
    end
    assembled = Any[calls[i] for i in sort!(collect(keys(calls)))]
    return nothing, (reasoning=join(reasoning), content=join(content), finish=String(finish), tool_calls=assembled)
end

# ---------------------------------------------------------------------------
# Request builders

function chat_json(messages; stream=false, max_tokens=16, extra...)
    d = Dict{String, Any}(
        "model" => "soak",
        "messages" => messages,
        "max_tokens" => max_tokens,
        "stream" => stream,
    )
    for (k, v) in extra
        d[String(k)] = v
    end
    return JSON.json(d)
end

user(text) = Any[Dict("role" => "user", "content" => text)]

const WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "grove", "haze", "iris", "jade"]
words(rng, n) = join((WORDS[rand(rng, 1:length(WORDS))] for _ in 1:n), " ")

function post_chat(body; close=true)
    return build_request("POST", "/v1/chat/completions", body; close=close)
end

# Each runner returns (reason, status, request_digest); reason is nothing when
# the reply was right.
digest(raw) = replace(first(raw, 300), "\r\n" => " ")

function sampling_extra(rng)
    rand(rng) < 0.3 ? (temperature=0.8, seed=rand(rng, 1:10^6)) : (;)
end

const USAGE_OPTIONS = (stream_options=Dict("include_usage" => true),)

# completion_tokens and cached_tokens of a reply the checks already accepted.
function usage_of(r::Response, stream::Bool)
    u = nothing
    if stream
        for c in Iterators.reverse(r.chunks)
            occursin("\"usage\"", c) || continue
            for e in split(c, "\n\n"; keepempty=false)
                startswith(e, "data: {") || continue
                j = parse_json(Vector{UInt8}(codeunits(SubString(e, 7))))
                j isa AbstractDict && get(j, "usage", nothing) isa AbstractDict && (u = j["usage"])
            end
            u === nothing || break
        end
    else
        j = parse_json(r.body)
        j isa AbstractDict && (u = get(j, "usage", nothing))
    end
    u isa AbstractDict || return (completion=0, cached=0)
    d = get(u, "prompt_tokens_details", nothing)
    cached = d isa AbstractDict ? get(d, "cached_tokens", 0) : 0
    completion = get(u, "completion_tokens", 0)
    return (completion=completion isa Integer ? Int(completion) : 0, cached=cached isa Integer ? Int(cached) : 0)
end

# Time to first token, and decode speed from the first chunk to the end of the
# stream. One token has no decode interval, so it gives no speed.
function observe_stream!(ctx, r::Response, tm::Timing)
    isnan(tm.t_first) && return
    toks = usage_of(r, true).completion
    dt = tm.t_end - tm.t_first
    rate = toks > 1 && dt > 0 ? toks / dt : NaN
    lock(ctx.log_lock) do
        push!(ctx.stream_obs, (time() - ctx.t0, (tm.t_first - tm.t_write) * 1000, rate))
    end
    return nothing
end

# Sends `raw` on a fresh connection and checks the reply. Returns (response,
# reason, answer). Streamed requests are timed for the metrics windows.
function chat_exchange(ctx, raw; stream, max_tokens, include_usage=stream)
    tm = stream ? Timing() : nothing
    r = exchange(ctx.cfg, raw; timing=tm)
    reason, answer = stream ? check_stream(r, max_tokens; include_usage=include_usage) :
                     check_completion(r, max_tokens)
    reason === nothing && stream && observe_stream!(ctx, r, tm)
    return r, reason, answer
end

function run_plain(ctx, rng, kind; stream, long)
    n = long ? ctx.cfg.long_words : rand(rng, 3:12)
    extra = stream ? USAGE_OPTIONS : (;)
    body = chat_json(user(words(rng, n)); stream=stream, max_tokens=16, extra..., sampling_extra(rng)...)
    raw = post_chat(body)
    r, reason, _ = chat_exchange(ctx, raw; stream=stream, max_tokens=16)
    return reason, r.status, digest(raw)
end

function run_greedy(ctx, rng, kind; stream, index=nothing)
    i = index === nothing ? rand(rng, 1:length(GREEDY_PROMPTS)) : index
    key = "greedy$i"
    extra = stream ? (USAGE_OPTIONS..., temperature=0) : (;)
    raw = post_chat(chat_json(user(GREEDY_PROMPTS[i]); stream=stream, max_tokens=24, extra...))
    r, reason, answer = chat_exchange(ctx, raw; stream=stream, max_tokens=24)
    reason === nothing && (reason = compare_baseline(ctx, key, answer))
    return reason, r.status, digest(raw)
end

# ---------------------------------------------------------------------------
# Conversations

const CONV_MAX_TOKENS = 24
const CONV_CTX_FRACTION = 0.8       # stop well before the history could fill n_ctx

# An upper bound: no tokenizer makes more tokens than bytes, and each message
# gets a generous allowance for its role markers and the template preamble.
est_tokens(messages) = 64 + sum(m -> ncodeunits(m["content"]) + 16, messages; init=0)

reply_text(a) = a.reasoning * a.content

function first_difference(a::AbstractString, b::AbstractString)
    for (i, (x, y)) in enumerate(zip(a, b))
        x == y || return i
    end
    return min(length(a), length(b)) + 1
end

# Words the conversations never use, so an evicting prompt shares no prefix with
# one beyond the template header.
const EVICT_WORDS = ["quartz", "lantern", "meadow", "thistle", "orbit", "saffron", "tundra", "velvet",
    "walnut", "zephyr", "bramble", "cobalt", "driftwood", "falcon", "gossamer", "harbor"]

# One unrelated request between a turn and its replay. The server keeps the last
# prompt's rows, so without it the replay would be served from cache whole and
# always agree. Returns whether the eviction itself succeeded.
function evict!(ctx, rng)
    ok = Ref(false)
    timed_request!(ctx, rng, "evict"; run=() -> begin
        text = join((EVICT_WORDS[rand(rng, 1:length(EVICT_WORDS))] for _ in 1:rand(rng, 8:16)), " ")
        raw = post_chat(chat_json(user(text); max_tokens=1, temperature=0, enable_thinking=false))
        r, reason, _ = chat_exchange(ctx, raw; stream=false, max_tokens=1)
        ok[] = reason === nothing
        return reason, r.status, digest(raw)
    end)
    return ok[]
end

# Evicts, then sends the same messages again on a new connection. A divergence is
# counted and logged; it fails the run only under --replay-strict. A replay whose
# reply is malformed fails either way. Only a fresh replay can show cache-origin
# drift, and only when the original itself ran on a cached prefix, so those are
# counted apart ("drift-eligible").
function replay!(ctx, rng, raw, sent, stream, turn, answer, cached)
    evict!(ctx, rng) || return nothing
    timed_request!(ctx, rng, "replay"; run=() -> begin
        r, reason, again = chat_exchange(ctx, raw; stream=stream, max_tokens=CONV_MAX_TOKENS)
        reason === nothing || return reason, r.status, digest(raw)
        u = usage_of(r, stream)
        a, b = reply_text(answer), reply_text(again)
        offset = a == b ? 0 : first_difference(a, b)
        fresh = u.cached <= FRESH_CACHED_MAX
        eligible = fresh && cached > FRESH_CACHED_MAX
        lock(ctx.log_lock) do
            ctx.replay_n += 1
            fresh && (ctx.replay_fresh += 1)
            eligible && (ctx.replay_eligible += 1)
            eligible && offset != 0 && (ctx.replay_eligible_diverged += 1)
            offset == 0 || push!(ctx.replay_offsets, offset)
        end
        # a divergence keeps both whole replies and how each ended, for scripts/soak_margin.jl
        extra = offset == 0 ? (original=first(a, 400), replay=first(b, 400)) :
                (original=a, replay=b, original_finish=answer.finish, replay_finish=again.finish,
                    messages=sent, offset=offset)
        emit(ctx, "replay"; ok=offset == 0, turn=turn,
            original_cached=cached, replay_cached=u.cached, fresh=fresh, drift_eligible=eligible,
            thinking=false, extra...)
        if offset != 0 && ctx.cfg.replay_strict
            return "replay diverged at character $offset of turn $turn: original " *
                   "$(repr(first(a, 60))) replay $(repr(first(b, 60)))", r.status, digest(raw)
        end
        return nothing, r.status, digest(raw)
    end)
    return nothing
end

# One conversation: each turn sends the whole history, greedy, and appends the
# reply. Every turn is a request of its own kind. It ends quietly when the next
# request could not be shown to fit, or at the deadline; a failed turn ends it
# after the failure is recorded. Any client but the first ends it when a
# single-client window opens.
function run_conversation!(ctx, rng, kind; stream, deadline, client=1)
    cfg = ctx.cfg
    planned = rand(rng, 2:cfg.max_turns)
    # a thinking model would spend the whole budget on reasoning and leave no content
    extra = stream ? (USAGE_OPTIONS..., temperature=0, enable_thinking=false) :
            (temperature=0, enable_thinking=false)
    messages = Any[]
    ended = "complete"
    turns = 0
    for turn in 1:planned
        if time() >= deadline
            ended = "deadline"
            break
        end
        if client > 1 && in_solo(ctx)
            ended = "solo window"
            break
        end
        text = turn == 1 ? words(rng, rand(rng, 5:12)) : "And then? " * words(rng, rand(rng, 3:8))
        push!(messages, Dict("role" => "user", "content" => text))
        if est_tokens(messages) + CONV_MAX_TOKENS > CONV_CTX_FRACTION * ctx.n_ctx
            pop!(messages)
            ended = "context"
            break
        end
        raw = post_chat(chat_json(messages; stream=stream, max_tokens=CONV_MAX_TOKENS, extra...))
        got = Ref{Any}(nothing)
        timed_request!(ctx, rng, kind; run=() -> begin
            r, reason, answer = chat_exchange(ctx, raw; stream=stream, max_tokens=CONV_MAX_TOKENS)
            reason === nothing && (got[] = (answer, usage_of(r, stream).cached))
            return reason, r.status, "turn $turn: " * digest(raw)
        end)
        if got[] === nothing
            ended = "failed"
            break
        end
        turns += 1
        answer, cached = got[]
        if isempty(answer.content)      # all reasoning: there is nothing to append
            ended = "empty reply"
            break
        end
        sent = copy(messages)
        push!(messages, Dict("role" => "assistant", "content" => answer.content))
        rand(rng) < cfg.replay_frac && replay!(ctx, rng, raw, sent, stream, turn, answer, cached)
    end
    emit(ctx, "conversation"; kind=kind, planned=planned, turns=turns, ended=ended)
    return nothing
end

function compare_baseline(ctx, key, answer)
    base = lock(ctx.log_lock) do
        get!(ctx.baselines, key, answer)
    end
    answer == base && return nothing
    return "greedy reply differs from the first one for this prompt: expected " *
           "$(repr(base.content)) got $(repr(answer.content))"
end

# Reads the first content event and hangs up. Passing means the part that was
# read is well formed; the server must also keep serving afterwards, which the
# other kinds and the health probe check.
function run_disconnect(ctx, rng, kind)
    raw = post_chat(chat_json(user(words(rng, 5)); stream=true, max_tokens=256))
    seen = Ref(0)
    r = exchange(ctx.cfg, raw; on_chunk=c -> begin
        seen[] += 1
        return seen[] < 3     # the role chunk, then one more, then hang up
    end)
    r.status == 200 || return "expected status 200, got $(r.status)", r.status, digest(raw)
    isempty(r.chunks) && return "no chunk before the disconnect", r.status, digest(raw)
    startswith(r.chunks[1], "data: ") || return "first event is not 'data: '", r.status, digest(raw)
    return nothing, r.status, digest(raw)
end

# Requests the server must refuse with a 4xx and the error object.
function bad_requests(ctx)
    return [
        (post_chat("{\"messages\": ["), 400, nothing),
        (post_chat("[1, 2]"), 400, nothing),
        (post_chat("{}"), 400, nothing),
        (post_chat("{\"messages\": []}"), 400, nothing),
        (post_chat("{\"messages\": \"hi\"}"), 400, nothing),
        (post_chat(chat_json(user("x"); stream="yes")), 400, nothing),
        (post_chat(chat_json(user("x"); temperature=9)), 400, nothing),
        (post_chat(chat_json(user("x"); max_tokens=-1)), 400, nothing),
        (post_chat("{\"messages\": [3]}"), 400, nothing),
        (build_request("GET", "/no/such/route", nothing), 404, nothing),
        (build_request("GET", "/v1/chat/completions", nothing), 405, nothing),
        (build_request("POST", "/health", "{}"), 405, nothing),
        ("NOT A REQUEST\r\n\r\n", 400, nothing),
        ("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: abc\r\n\r\n", 400, nothing),
    ]
end

function run_bad_request(ctx, rng, kind)
    raw, status, code = rand(rng, bad_requests(ctx))
    r = exchange(ctx.cfg, raw)
    return check_error(r, status; code=code), r.status, digest(raw)
end

# A declared length past the server's body limit is refused before the body is
# read, so nothing large is sent.
function run_oversize_body(ctx, rng, kind)
    raw = build_request("POST", "/v1/chat/completions", "{}"; length_override=64 * 1024 * 1024)
    r = exchange(ctx.cfg, raw)
    return check_error(r, 413), r.status, digest(raw)
end

# One word is at least one token, so n_ctx + 50 words cannot fit.
function run_over_context(ctx, rng, kind)
    raw = post_chat(chat_json(user(words(rng, ctx.n_ctx + 50)); max_tokens=4))
    r = exchange(ctx.cfg, raw)
    return check_error(r, 400; code="context_length_exceeded"), r.status, digest(raw[1:min(end, 300)])
end

# Two requests on one connection: the second must see a clean response boundary.
function run_keepalive(ctx, rng, kind)
    cfg = ctx.cfg
    sock = Sockets.connect(Sockets.IPv4(cfg.host), cfg.port)
    timer = Timer(_ -> close(sock), cfg.request_timeout)
    try
        write(sock, build_request("GET", "/health", nothing; close=false))
        r1 = read_response(sock)
        r1.status == 200 || return "first request: status $(r1.status)", r1.status, "GET /health"
        raw = post_chat(chat_json(user(words(rng, 4))); close=true)
        write(sock, raw)
        r2 = read_response(sock)
        reason, _ = check_completion(r2, 16)
        return reason, r2.status, digest(raw)
    finally
        close(timer)
        close(sock)
    end
end

# ---------------------------------------------------------------------------
# Tool calls

const TOOL_CITIES = ["Paris", "Oslo", "Lima", "Kyoto", "Cairo", "Quito", "Hanoi", "Perth"]
const TOOL_MAX_TOKENS = 128
const WEATHER_TOOL = Dict(
    "type" => "function",
    "function" => Dict(
        "name" => "get_weather",
        "description" => "Current temperature in a city.",
        "parameters" => Dict(
            "type" => "object",
            "properties" => Dict("city" => Dict("type" => "string")),
            "required" => ["city"],
        ),
    ),
)

# Greedy and not thinking, for the same reasons as the conversations.
tool_extra(stream) = stream ? (USAGE_OPTIONS..., temperature=0, enable_thinking=false, tools=[WEATHER_TOOL]) :
                     (temperature=0, enable_thinking=false, tools=[WEATHER_TOOL])

# A call the client could act on: id, type, the declared name, and arguments that
# are a JSON object with a string city.
function check_tool_call(c)
    c isa AbstractDict || return "a tool call is not an object"
    id = get(c, "id", nothing)
    (id isa AbstractString && !isempty(id)) || return "tool call has no id"
    get(c, "type", nothing) == "function" || return "tool call type is not function"
    f = get(c, "function", nothing)
    f isa AbstractDict || return "tool call has no function object"
    get(f, "name", nothing) == "get_weather" || return "tool call names $(repr(get(f, "name", nothing))), not get_weather"
    args = get(f, "arguments", nothing)
    args isa AbstractString || return "tool call arguments are not a string"
    j = parse_json(Vector{UInt8}(codeunits(args)))
    j isa AbstractDict || return "tool call arguments are not a JSON object: $(repr(first(args, 80)))"
    city = get(j, "city", nothing)
    (city isa AbstractString && !isempty(city)) || return "tool call arguments have no string city"
    return nothing
end

# finish_reason and the calls must agree, and every call must be well formed.
function check_tool_answer(answer)
    calls = answer.tool_calls
    if answer.finish == "tool_calls" && isempty(calls)
        return "finish_reason is tool_calls but there are no calls"
    end
    if !isempty(calls) && answer.finish != "tool_calls"
        return "tool calls present with finish_reason '$(answer.finish)'"
    end
    for c in calls
        reason = check_tool_call(c)
        reason === nothing || return reason
    end
    return nothing
end

# The model is asked for the weather with one tool on offer. A call is answered
# with a tool message and the follow-up reply is checked as well; a text answer
# is counted, not failed, since the soak does not grade what the model decides.
function run_tool_call(ctx, rng, kind; stream)
    city = TOOL_CITIES[rand(rng, 1:length(TOOL_CITIES))]
    messages = Any[Dict("role" => "user", "content" => "What is the weather in $city? Use the tool.")]
    raw = post_chat(chat_json(messages; stream=stream, max_tokens=TOOL_MAX_TOKENS, tool_extra(stream)...))
    r, reason, answer = chat_exchange(ctx, raw; stream=stream, max_tokens=TOOL_MAX_TOKENS)
    reason === nothing || return reason, r.status, digest(raw)
    reason = check_tool_answer(answer)
    reason === nothing || return reason, r.status, digest(raw)
    if isempty(answer.tool_calls)
        lock(ctx.log_lock) do
            ctx.tool_turns += 1
            ctx.tool_no_call += 1
        end
        return nothing, r.status, digest(raw)
    end
    push!(messages, Dict("role" => "assistant", "content" => answer.content, "tool_calls" => answer.tool_calls))
    for c in answer.tool_calls
        push!(messages, Dict("role" => "tool", "tool_call_id" => c["id"], "content" => "{\"temp_c\": 12}"))
    end
    raw2 = post_chat(chat_json(messages; stream=stream, max_tokens=TOOL_MAX_TOKENS, tool_extra(stream)...))
    r2, reason, again = chat_exchange(ctx, raw2; stream=stream, max_tokens=TOOL_MAX_TOKENS)
    reason === nothing || return "after the tool result: " * reason, r2.status, digest(raw2)
    reason = check_tool_answer(again)
    reason === nothing || return "after the tool result: " * reason, r2.status, digest(raw2)
    lock(ctx.log_lock) do
        ctx.tool_turns += 1
        ctx.tool_calls += 1
    end
    return nothing, r2.status, digest(raw2)
end

# A server whose chat format has no tool-call parser refuses any request that
# carries tools with a 400; that is a fact about the model, so the tool kinds are
# dropped from the mix. Anything else other than a well-formed 200 is a failure.
function probe_tools!(ctx)
    raw = post_chat(chat_json(Any[Dict("role" => "user", "content" => "Hi")]; max_tokens=1,
        temperature=0, enable_thinking=false, tools=[WEATHER_TOOL]))
    try
        r = exchange(ctx.cfg, raw)
        if r.status == 400
            j = parse_json(r.body)
            err = j isa AbstractDict ? get(j, "error", nothing) : nothing
            msg = err isa AbstractDict ? string(get(err, "message", "")) : ""
            if (err isa AbstractDict && get(err, "code", nothing) == "tools_unsupported") ||
               occursin("not supported", lowercase(msg))
                ctx.tools_on = false
                ctx.tool_note = "the server does not support tools ($(first(msg, 120))); tool kinds are off"
                emit(ctx, "note"; message=ctx.tool_note)
                return nothing
            end
        end
        reason = r.status == 200 ? first(check_completion(r, 1)) : "expected status 200, got $(r.status)"
        if reason !== nothing
            ctx.tools_on = false
            fail!(ctx, "tool_probe", reason, digest(raw))
        end
    catch e
        ctx.tools_on = false
        fail!(ctx, "tool_probe", "io error: $(first(e isa WireError ? e.msg : sprint(showerror, e), 160))", digest(raw))
    end
    return nothing
end


function dispatch(ctx, rng, kind)
    kind == "short" && return run_plain(ctx, rng, kind; stream=false, long=false)
    kind == "short_stream" && return run_plain(ctx, rng, kind; stream=true, long=false)
    kind == "long" && return run_plain(ctx, rng, kind; stream=false, long=true)
    kind == "long_stream" && return run_plain(ctx, rng, kind; stream=true, long=true)
    kind == "greedy" && return run_greedy(ctx, rng, kind; stream=false)
    kind == "greedy_stream" && return run_greedy(ctx, rng, kind; stream=true)
    kind == "disconnect" && return run_disconnect(ctx, rng, kind)
    kind == "bad_request" && return run_bad_request(ctx, rng, kind)
    kind == "oversize_body" && return run_oversize_body(ctx, rng, kind)
    kind == "over_context" && return run_over_context(ctx, rng, kind)
    kind == "keepalive" && return run_keepalive(ctx, rng, kind)
    kind == "tool_call" && return run_tool_call(ctx, rng, kind; stream=false)
    kind == "tool_call_stream" && return run_tool_call(ctx, rng, kind; stream=true)
    error("unknown kind $kind")
end

mix_for(tools::Bool) = tools ? MIX : [kw for kw in MIX if first(kw) ∉ TOOL_KINDS]

function pick_kind(rng; tools::Bool=true)
    mix = mix_for(tools)
    total = sum(last, mix)
    x = rand(rng, 1:total)
    for (k, w) in mix
        x <= w && return k
        x -= w
    end
    return first(first(mix))
end

# One pick of the mix. A conversation records its own turns.
function run_kind!(ctx, rng, kind, deadline; client=1)
    kind == "conversation" && return run_conversation!(ctx, rng, kind; stream=false, deadline=deadline, client=client)
    kind == "conversation_stream" && return run_conversation!(ctx, rng, kind; stream=true, deadline=deadline, client=client)
    return timed_request!(ctx, rng, kind)
end

# The kinds client `i` of `n` sends once each before it draws from the mix. The
# clients split the mix between them, so a short or slow run still sends every
# kind; a weight-1 kind can draw nothing in a few hundred picks.
cover_kinds(i, n; tools::Bool=true) = [first(kw) for (j, kw) in enumerate(mix_for(tools)) if mod1(j, n) == i]

# One step of client `i`: in a single-client window the first client runs a
# conversation and the others wait it out. Outside one, the client works through
# `cover` first.
function client_step!(ctx, rng, i, deadline, cover=String[])
    if in_solo(ctx)
        i == 1 || return sleep(min(1.0, max(0.0, deadline - time())))
        return run_kind!(ctx, rng, rand(rng, ("conversation", "conversation_stream")), deadline)
    end
    kind = isempty(cover) ? pick_kind(rng; tools=ctx.tools_on) : popfirst!(cover)
    return run_kind!(ctx, rng, kind, deadline; client=i)
end

function timed_request!(ctx, rng, kind; run=() -> dispatch(ctx, rng, kind))
    t = time()
    reason, status, request = try
        run()
    catch e
        msg = e isa WireError ? e.msg : sprint(showerror, e)
        "io error: $(first(msg, 200))", 0, ""
    end
    record!(ctx, kind, (time() - t) * 1000, reason, status, request)
    return nothing
end

# ---------------------------------------------------------------------------
# Background probes

function health_probe(ctx)
    cfg = ctx.cfg
    raw = build_request("GET", "/health", nothing)
    while !ctx.stop[]
        t = time()
        reason = try
            r = exchange(cfg, raw)
            j = parse_json(r.body)
            ms = (time() - t) * 1000
            ctx.health_max = max(ctx.health_max, ms)
            if r.status != 200 || !(j isa AbstractDict) || get(j, "status", nothing) != "ok"
                "health answered status $(r.status)"
            elseif ms > cfg.health_timeout * 1000
                "health took $(round(ms; digits=0)) ms"
            end
        catch e
            "health probe failed: $(first(e isa WireError ? e.msg : sprint(showerror, e), 200))"
        end
        ms = (time() - t) * 1000
        record!(ctx, "health", ms, reason, 200, "GET /health")
        sleep(cfg.health_interval)
    end
end

function read_rss_kb(pid::Int)
    for line in eachline("/proc/$pid/status")
        m = match(r"^VmRSS:\s+(\d+) kB", line)
        m === nothing || return parse(Int, m.captures[1])
    end
    error("no VmRSS line")
end

function rss_probe(ctx)
    pid = ctx.cfg.pid
    pid === nothing && return
    function sample()
        try
            kb = read_rss_kb(pid)
            lock(ctx.log_lock) do
                push!(ctx.rss, (time() - ctx.t0, kb))
                ctx.rss_peak_kb = max(ctx.rss_peak_kb, kb)
            end
            emit(ctx, "rss"; kb=kb)
        catch e
            fail!(ctx, "rss", "cannot read RSS of pid $pid: $(first(sprint(showerror, e), 120))")
        end
    end
    while !ctx.stop[]
        sample()
        t_end = time() + ctx.cfg.rss_interval
        while time() < t_end && !ctx.stop[]
            sleep(min(0.25, ctx.cfg.rss_interval))
        end
    end
    sample()
end

# Growth is judged between a baseline taken after warm-up (the highest sample in
# the 20-40% span of the post-warm-up samples) and the median of the last fifth
# of them. Samples before cfg.warmup are reported but never judged. The same
# verdict serves VRAM, with its own absolute threshold.
function rss_verdict(samples::Vector{Tuple{Float64, Int}}, cfg::Config; growth_mb=cfg.rss_growth_mb)
    isempty(samples) && return (judged=false, grew=false, start_kb=0, peak_kb=0, end_kb=0, baseline_kb=0, slope_mb_min=NaN)
    all_kbs = last.(samples)
    start_kb, peak_kb, end_kb = first(all_kbs), maximum(all_kbs), last(all_kbs)
    samples = [s for s in samples if s[1] >= cfg.warmup]
    kbs = last.(samples)
    n = length(samples)
    n < 5 && return (judged=false, grew=false, start_kb, peak_kb, end_kb, baseline_kb=start_kb, slope_mb_min=NaN)
    span = samples[end][1] - samples[1][1]
    t1 = samples[1][1]
    window = [kb for (t, kb) in samples if t1 + 0.2span <= t <= t1 + 0.4span]
    baseline = isempty(window) ? kbs[max(1, n ÷ 4)] : maximum(window)
    tail = sort(kbs[(end - max(1, n ÷ 5) + 1):end])
    final = tail[(length(tail) + 1) ÷ 2]
    # least-squares slope over the judged samples
    ts = first.(samples) ./ 60
    mt, mk = sum(ts) / n, sum(kbs) / n
    denom = sum((t - mt)^2 for t in ts)
    slope = denom == 0 ? 0.0 : sum((t - mt) * (k - mk) for (t, k) in zip(ts, kbs)) / denom / 1024
    grew = (final - baseline) > max(cfg.rss_growth_frac * baseline, growth_mb * 1024)
    return (judged=true, grew, start_kb, peak_kb, end_kb, baseline_kb=baseline, slope_mb_min=slope)
end

# Memory in use on the first GPU, in MB, or nothing when nvidia-smi cannot say.
function read_vram_mb()
    try
        out = read(pipeline(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`; stderr=devnull), String)
        return parse(Int, strip(first(split(out, '\n'; keepempty=false))))
    catch
        return nothing
    end
end

# One window every cfg.metrics_interval seconds. TTFT and decode speed come from
# the streamed requests that finished in the window; the last window, cut short
# by the end of the run, is logged but marked partial and never judged.
function metrics_probe(ctx)
    cfg = ctx.cfg
    vram_noted = false
    lock(ctx.log_lock) do
        ctx.windows_prev = (ctx.done, length(ctx.failures), length(ctx.stream_obs))
    end
    t_start = time() - ctx.t0
    index = 0
    function close_window(full)
        index += 1
        mb = nothing
        if cfg.vram
            mb = read_vram_mb()
            if mb === nothing
                vram_noted || emit(ctx, "note"; message="nvidia-smi gave no reading; VRAM is not sampled")
                vram_noted = true
            else
                lock(ctx.log_lock) do
                    push!(ctx.vram, (time() - ctx.t0, mb * 1024))
                end
            end
        end
        w = lock(ctx.log_lock) do
            done0, err0, cursor = ctx.windows_prev
            obs = ctx.stream_obs[(cursor + 1):end]
            ttft = sort!([o[2] for o in obs])
            rates = [o[3] for o in obs if !isnan(o[3])]
            ctx.windows_prev = (ctx.done, length(ctx.failures), length(ctx.stream_obs))
            (index=index, t_start=t_start, full=full, requests=ctx.done - done0,
                errors=length(ctx.failures) - err0, streamed=length(obs),
                ttft_p50=pct(ttft, 0.50), ttft_p99=pct(ttft, 0.99),
                tok_s=isempty(rates) ? NaN : sum(rates) / length(rates),
                rss_kb=isempty(ctx.rss) ? 0 : ctx.rss[end][2], vram_mb=mb)
        end
        lock(ctx.log_lock) do
            push!(ctx.windows, w)
        end
        nn(x) = isnan(x) ? nothing : round(x; digits=1)
        emit(ctx, "metrics"; window=w.index, partial=!full, requests=w.requests, errors=w.errors,
            streamed=w.streamed, ttft_p50_ms=nn(w.ttft_p50), ttft_p99_ms=nn(w.ttft_p99),
            tok_s=nn(w.tok_s), rss_kb=w.rss_kb == 0 ? nothing : w.rss_kb, vram_mb=w.vram_mb)
        t_start = time() - ctx.t0
    end
    while !ctx.stop[]
        t_end = time() + cfg.metrics_interval
        while time() < t_end && !ctx.stop[]
            sleep(min(0.25, cfg.metrics_interval))
        end
        close_window(time() >= t_end)
    end
end

# Least-squares line through (window start, value) points after the warm-up. The
# run is flagged when the fitted rise from the first to the last point exceeds
# both the absolute threshold and the fraction of the first value. Fewer than
# min_points usable windows is too little to judge, which passes.
function trend_verdict(points::Vector{Tuple{Float64, Float64}}, warmup; growth_abs, growth_frac, min_points=5)
    pts = [p for p in points if p[1] >= warmup && isfinite(p[2])]
    n = length(pts)
    n < min_points && return (judged=false, grew=false, n, first_value=NaN, last_value=NaN, rise=NaN)
    ts, ys = first.(pts), last.(pts)
    mt, my = sum(ts) / n, sum(ys) / n
    denom = sum((t - mt)^2 for t in ts)
    slope = denom == 0 ? 0.0 : sum((t - mt) * (y - my) for (t, y) in zip(ts, ys)) / denom
    rise = slope * (ts[end] - ts[1])
    grew = rise > growth_abs && rise > growth_frac * ys[1]
    return (judged=true, grew, n, first_value=ys[1], last_value=ys[end], rise=rise)
end

p99_verdict(windows, cfg::Config) = trend_verdict(
    Tuple{Float64, Float64}[(w.t_start, w.ttft_p99) for w in windows if w.full], cfg.warmup;
    growth_abs=cfg.p99_growth_ms, growth_frac=cfg.p99_growth_frac)

# ---------------------------------------------------------------------------
# Run

mutable struct Result
    ok::Bool
    counts::Dict{String, Int}
    failed::Dict{String, Int}
    failures::Vector{Failure}
    latency::Dict{String, NamedTuple}
    rss::NamedTuple
    health_max_ms::Float64
    seconds::Float64
    replay::NamedTuple      # (n, fresh, eligible, diverged, eligible_diverged, first_offset)
    tools::NamedTuple       # (enabled, note, turns, calls, no_call)
    metrics::NamedTuple     # (windows, mean_tok_s)
    p99::NamedTuple
    vram::NamedTuple
end

function summarize(ctx::Ctx, seconds)
    lat = Dict{String, NamedTuple}()
    for (k, v) in ctx.latencies
        s = sort(v)
        lat[k] = (n=length(s), p50=pct(s, 0.50), p95=pct(s, 0.95), p99=pct(s, 0.99), max=s[end])
    end
    rss = rss_verdict(ctx.rss, ctx.cfg)
    rss.grew && push!(
        ctx.failures,
        Failure("rss", "RSS grew from $(rss.baseline_kb ÷ 1024) MB after warm-up to $(rss.end_kb ÷ 1024) MB", ""),
    )
    vram = rss_verdict(ctx.vram, ctx.cfg; growth_mb=ctx.cfg.vram_growth_mb)
    vram.grew && push!(
        ctx.failures,
        Failure("vram", "VRAM grew from $(vram.baseline_kb ÷ 1024) MB after warm-up to $(vram.end_kb ÷ 1024) MB", ""),
    )
    p99 = p99_verdict(ctx.windows, ctx.cfg)
    p99.grew && push!(
        ctx.failures,
        Failure("p99", "p99 latency trending: time to first token rose $(round(p99.rise; digits=0)) ms over " *
                       "$(p99.n) windows after warm-up, from $(round(p99.first_value; digits=0)) ms", ""),
    )
    rates = [o[3] for o in ctx.stream_obs if !isnan(o[3])]
    metrics = (windows=count(w -> w.full, ctx.windows), mean_tok_s=isempty(rates) ? NaN : sum(rates) / length(rates))
    replay = (n=ctx.replay_n, fresh=ctx.replay_fresh, eligible=ctx.replay_eligible,
        diverged=length(ctx.replay_offsets), eligible_diverged=ctx.replay_eligible_diverged,
        first_offset=isempty(ctx.replay_offsets) ? 0 : ctx.replay_offsets[1])
    tools = (enabled=ctx.tools_on, note=ctx.tool_note, turns=ctx.tool_turns, calls=ctx.tool_calls,
        no_call=ctx.tool_no_call)
    ok = isempty(ctx.failures)
    return Result(ok, ctx.counts, ctx.failed, ctx.failures, lat, rss, ctx.health_max, seconds, replay, tools, metrics, p99, vram)
end

"""
    soak(cfg::Config; log=devnull) -> Result

Run the soak against the server at `cfg.host:cfg.port` for `cfg.duration`
seconds. `result.ok` is false when any check failed.
"""
function soak(cfg::Config; log::IO=devnull)
    ctx = Ctx(
        cfg, 0, log, ReentrantLock(), time(),
        Dict{String, Vector{Float64}}(), Dict{String, Int}(), Dict{String, Int}(),
        Failure[], Dict{String, NamedTuple}(), Tuple{Float64, Int}[], 0, 0.0, Ref(false),
        0, NTuple{3, Float64}[], NamedTuple[], Tuple{Float64, Int}[], 0, Int[], (0, 0, 0),
        0, 0, 0, true, "", 0, 0, 0, time(),
    )
    emit(ctx, "start"; port=cfg.port, clients=cfg.clients, duration=cfg.duration, seed=cfg.seed,
        solo_every=cfg.solo_every, solo_len=cfg.solo_len)

    # The context limit decides what counts as an oversize prompt.
    try
        r = exchange(cfg, build_request("GET", "/props", nothing))
        j = parse_json(r.body)
        ctx.n_ctx = Int(j["default_generation_settings"]["n_ctx"])
    catch e
        fail!(ctx, "setup", "cannot read n_ctx from /props: $(first(sprint(showerror, e), 160))")
        return summarize(ctx, 0.0)
    end

    probe_tools!(ctx)

# Each greedy prompt twice, in sequence, before the mix starts: a server that
    # is not repeatable alone fails here rather than under load.
    rng0 = MersenneTwister(cfg.seed)
    for i in 1:length(GREEDY_PROMPTS), _ in 1:2
        timed_request!(ctx, rng0, "greedy"; run=() -> run_greedy(ctx, rng0, "greedy"; stream=false, index=i))
    end

    started = time()
    ctx.started = started
    deadline = started + cfg.duration
    probes = Task[]
    push!(probes, @async health_probe(ctx))
    cfg.pid === nothing || push!(probes, @async rss_probe(ctx))
    push!(probes, @async metrics_probe(ctx))
    clients = map(1:cfg.clients) do i
        @async begin
            rng = MersenneTwister(cfg.seed + i)
            cover = cover_kinds(i, cfg.clients; tools=ctx.tools_on)
            while time() < deadline
                client_step!(ctx, rng, i, deadline, cover)
            end
        end
    end
    foreach(wait, clients)
    ctx.stop[] = true
    foreach(wait, probes)
    seconds = time() - started
    res = summarize(ctx, seconds)
    emit(ctx, "summary"; ok=res.ok, counts=res.counts, failed=res.failed, seconds=round(seconds; digits=1))
    return res
end

function print_summary(io::IO, res::Result)
    println(io, "soak ran ", round(res.seconds; digits=1), " s: ", res.ok ? "PASS" : "FAIL")
    println(io)
    println(io, rpad("kind", 21), lpad("n", 7), lpad("failed", 8), lpad("p50 ms", 10), lpad("p95 ms", 10), lpad("p99 ms", 10))
    for k in sort!(collect(keys(res.counts)))
        l = get(res.latency, k, nothing)
        f(x) = lpad(l === nothing ? "-" : string(round(x; digits=1)), 10)
        println(io, rpad(k, 21), lpad(res.counts[k], 7), lpad(get(res.failed, k, 0), 8),
            l === nothing ? "" : f(l.p50) * f(l.p95) * f(l.p99))
    end
    println(io)
    println(io, "health: max ", round(res.health_max_ms; digits=1), " ms")
    mb(kb) = round(kb / 1024; digits=1)
    r = res.rss
    if r.peak_kb == 0
        println(io, "rss: not sampled")
    else
        println(io, "rss MB: start ", mb(r.start_kb), "  peak ", mb(r.peak_kb), "  end ", mb(r.end_kb),
            "  slope ", isnan(r.slope_mb_min) ? "n/a" : "$(round(r.slope_mb_min; digits=2)) MB/min",
            r.judged ? "" : "  (too few samples to judge growth)")
    end
    v = res.vram
    if v.peak_kb != 0
        println(io, "vram MB: start ", mb(v.start_kb), "  peak ", mb(v.peak_kb), "  end ", mb(v.end_kb),
            v.judged ? "" : "  (too few samples after warm-up to judge growth)")
    end
    rp = res.replay
    println(io, "replay: ", rp.n, " compared, ", rp.fresh, " fresh, ", rp.eligible, " drift-eligible; ",
        rp.diverged, " diverged (", rp.eligible_diverged, " of the drift-eligible)",
        rp.diverged == 0 ? "" : ", first at character $(rp.first_offset)")
    tl = res.tools
    if !tl.enabled
        println(io, "tools: off", isempty(tl.note) ? "" : " - " * tl.note)
    else
        rate = tl.turns == 0 ? "n/a" : "$(round(100 * tl.no_call / tl.turns; digits=1))%"
        println(io, "tools: ", tl.turns, " turns, ", tl.calls, " called and answered, ", tl.no_call,
            " answered in text (no_call ", rate, ")")
    end
    p = res.p99
    println(io, "metrics: ", res.metrics.windows, " windows, mean decode ",
        isnan(res.metrics.mean_tok_s) ? "n/a" : "$(round(res.metrics.mean_tok_s; digits=1)) tok/s",
        p.judged ? ", p99 TTFT first $(round(p.first_value; digits=1)) ms, last $(round(p.last_value; digits=1)) ms " *
                   "after warm-up, fitted rise $(round(p.rise; digits=1)) ms ($(p.grew ? "TRENDING" : "flat"))" :
        ", p99 trend not judged (needs 5 windows after warm-up)")
    if !isempty(res.failures)
        println(io)
        println(io, length(res.failures), " failures; first ", min(20, length(res.failures)), ":")
        for f in first(res.failures, 20)
            println(io, "  [", f.kind, "] ", f.reason)
            isempty(f.request) || println(io, "      request: ", f.request)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Command line

const OPTIONS = Dict(
    "--host" => (:host, identity), "--port" => (:port, s -> parse(Int, s)),
    "--duration" => (:duration, s -> parse(Float64, s)), "--clients" => (:clients, s -> parse(Int, s)),
    "--pid" => (:pid, s -> parse(Int, s)), "--seed" => (:seed, s -> parse(Int, s)),
    "--request-timeout" => (:request_timeout, s -> parse(Float64, s)),
    "--health-timeout" => (:health_timeout, s -> parse(Float64, s)),
    "--health-interval" => (:health_interval, s -> parse(Float64, s)),
    "--rss-interval" => (:rss_interval, s -> parse(Float64, s)),
    "--rss-growth-mb" => (:rss_growth_mb, s -> parse(Float64, s)),
    "--rss-growth-frac" => (:rss_growth_frac, s -> parse(Float64, s)),
    "--long-words" => (:long_words, s -> parse(Int, s)),
    "--max-turns" => (:max_turns, s -> parse(Int, s)),
    "--replay-frac" => (:replay_frac, s -> parse(Float64, s)),
    "--metrics-interval" => (:metrics_interval, s -> parse(Float64, s)),
    "--warmup" => (:warmup, s -> parse(Float64, s)),
    "--p99-growth-ms" => (:p99_growth_ms, s -> parse(Float64, s)),
    "--p99-growth-frac" => (:p99_growth_frac, s -> parse(Float64, s)),
    "--vram-growth-mb" => (:vram_growth_mb, s -> parse(Float64, s)),
    "--solo-every" => (:solo_every, s -> parse(Float64, s)),
    "--solo-len" => (:solo_len, s -> parse(Float64, s)),
)

# Options that stand alone.
const FLAGS = Dict("--vram" => :vram, "--replay-strict" => :replay_strict)

function main(args)
    kw = Dict{Symbol, Any}()
    log_path = "soak-events.jsonl"
    i = 1
    try
        while i <= length(args)
            a = args[i]
            if haskey(FLAGS, a)
                kw[FLAGS[a]] = true
                i += 1
                continue
            end
            i < length(args) || throw(ArgumentError("$a needs a value"))
            if a == "--log"
                log_path = args[i+1]
            elseif haskey(OPTIONS, a)
                name, conv = OPTIONS[a]
                kw[name] = conv(args[i+1])
            else
                throw(ArgumentError("unknown argument $a"))
            end
            i += 2
        end
        cfg = Config(; kw...)
        cfg.clients > 0 && cfg.duration > 0 || throw(ArgumentError("--clients and --duration must be positive"))
        cfg.max_turns >= 2 || throw(ArgumentError("--max-turns must be at least 2"))
        cfg.metrics_interval > 0 || throw(ArgumentError("--metrics-interval must be positive"))
        cfg.solo_every == 0 || 0 < cfg.solo_len < cfg.solo_every ||
            throw(ArgumentError("--solo-len must be positive and shorter than --solo-every"))
        res = open(log_path, "w") do log
            soak(cfg; log=log)
        end
        print_summary(stdout, res)
        return res.ok ? 0 : 1
    catch e
        e isa ArgumentError || rethrow()
        println(stderr, "soak: ", e.msg)
        return 2
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(GessoSoak.main(ARGS))
end
