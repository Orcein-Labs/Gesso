# Soak client for the OpenAI-compatible chat server.
#
#   julia --project scripts/soak.jl --port 8080 --duration 600 --clients 4 --pid <server pid>
#
# Drives a running server with a fixed mix of requests (short and long prompts,
# streaming and not, repeated greedy prompts, disconnects mid-stream, malformed
# requests, oversize bodies and prompts over the context limit) and checks every
# reply. One JSON line per event goes to --log (default soak-events.jsonl); a
# summary is printed at the end. The exit code is 1 when any check failed.
#
# Checks: every reply is well formed for its type; errors carry the documented
# error object; a greedy prompt returns the same text every time, streamed or
# not; /health answers throughout; the server's resident memory (read from
# /proc/<pid>/status when --pid is given) does not keep growing.
#
# Options (all take a value): --host --port --duration --clients --pid --seed
#   --log --request-timeout --health-timeout --health-interval --rss-interval
#   --rss-growth-mb --rss-growth-frac --long-words
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
end

# Request kinds and their weights in the mix.
const MIX = [
    "short" => 20,
    "short_stream" => 20,
    "long" => 4,
    "long_stream" => 4,
    "greedy" => 12,
    "greedy_stream" => 8,
    "disconnect" => 8,
    "bad_request" => 8,
    "oversize_body" => 2,
    "over_context" => 4,
    "keepalive" => 4,
]

const GREEDY_PROMPTS = ["Count from one to ten in words.", "Name three primary colors."]
const FINISH_REASONS = ("stop", "length", "tool_calls")

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
end

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

# One request on a fresh connection under the request timeout. A timer closes the
# socket when the deadline passes, which fails the blocked read.
function exchange(cfg::Config, raw::AbstractString; on_chunk=nothing)
    sock = Sockets.connect(Sockets.IPv4(cfg.host), cfg.port)
    timer = Timer(cfg.request_timeout) do _
        close(sock)
    end
    try
        write(sock, raw)
        return read_response(sock; on_chunk=on_chunk)
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

# Returns (reason, answer); answer is (reasoning, content, finish).
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
    reason = check_usage(get(j, "usage", nothing), max_tokens)
    reason === nothing || return reason, nothing
    return nothing, (reasoning=String(reasoning), content=content === nothing ? "" : String(content), finish=String(finish))
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
    return nothing, (reasoning=join(reasoning), content=join(content), finish=String(finish))
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

function run_plain(ctx, rng, kind; stream, long)
    n = long ? ctx.cfg.long_words : rand(rng, 3:12)
    body = chat_json(user(words(rng, n)); stream=stream, max_tokens=16, sampling_extra(rng)...)
    raw = post_chat(body)
    r = exchange(ctx.cfg, raw)
    reason, _ = stream ? check_stream(r, 16) : check_completion(r, 16)
    return reason, r.status, digest(raw)
end

function run_greedy(ctx, rng, kind; stream, index=nothing)
    i = index === nothing ? rand(rng, 1:length(GREEDY_PROMPTS)) : index
    key = "greedy$i"
    extra = stream ? (stream_options=Dict("include_usage" => true), temperature=0) : (;)
    raw = post_chat(chat_json(user(GREEDY_PROMPTS[i]); stream=stream, max_tokens=24, extra...))
    r = exchange(ctx.cfg, raw)
    reason, answer = stream ? check_stream(r, 24; include_usage=true) : check_completion(r, 24)
    reason === nothing && (reason = compare_baseline(ctx, key, answer))
    return reason, r.status, digest(raw)
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
    error("unknown kind $kind")
end

function pick_kind(rng)
    total = sum(last, MIX)
    x = rand(rng, 1:total)
    for (k, w) in MIX
        x <= w && return k
        x -= w
    end
    return first(first(MIX))
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
# the 20-40% span of the run) and the median of the last fifth of the samples.
function rss_verdict(samples::Vector{Tuple{Float64, Int}}, cfg::Config)
    isempty(samples) && return (judged=false, grew=false, start_kb=0, peak_kb=0, end_kb=0, baseline_kb=0, slope_mb_min=NaN)
    kbs = last.(samples)
    start_kb, peak_kb, end_kb = first(kbs), maximum(kbs), last(kbs)
    n = length(samples)
    n < 5 && return (judged=false, grew=false, start_kb, peak_kb, end_kb, baseline_kb=start_kb, slope_mb_min=NaN)
    span = samples[end][1] - samples[1][1]
    t1 = samples[1][1]
    window = [kb for (t, kb) in samples if t1 + 0.2span <= t <= t1 + 0.4span]
    baseline = isempty(window) ? kbs[max(1, n ÷ 4)] : maximum(window)
    tail = sort(kbs[(end - max(1, n ÷ 5) + 1):end])
    final = tail[(length(tail) + 1) ÷ 2]
    # least-squares slope over the whole run
    ts = first.(samples) ./ 60
    mt, mk = sum(ts) / n, sum(kbs) / n
    denom = sum((t - mt)^2 for t in ts)
    slope = denom == 0 ? 0.0 : sum((t - mt) * (k - mk) for (t, k) in zip(ts, kbs)) / denom / 1024
    grew = (final - baseline) > max(cfg.rss_growth_frac * baseline, cfg.rss_growth_mb * 1024)
    return (judged=true, grew, start_kb, peak_kb, end_kb, baseline_kb=baseline, slope_mb_min=slope)
end

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
    ok = isempty(ctx.failures)
    return Result(ok, ctx.counts, ctx.failed, ctx.failures, lat, rss, ctx.health_max, seconds)
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
    )
    emit(ctx, "start"; port=cfg.port, clients=cfg.clients, duration=cfg.duration, seed=cfg.seed)

    # The context limit decides what counts as an oversize prompt.
    try
        r = exchange(cfg, build_request("GET", "/props", nothing))
        j = parse_json(r.body)
        ctx.n_ctx = Int(j["default_generation_settings"]["n_ctx"])
    catch e
        fail!(ctx, "setup", "cannot read n_ctx from /props: $(first(sprint(showerror, e), 160))")
        return summarize(ctx, 0.0)
    end

    # Each greedy prompt twice, in sequence, before the mix starts: a server that
    # is not repeatable alone fails here rather than under load.
    rng0 = MersenneTwister(cfg.seed)
    for i in 1:length(GREEDY_PROMPTS), _ in 1:2
        timed_request!(ctx, rng0, "greedy"; run=() -> run_greedy(ctx, rng0, "greedy"; stream=false, index=i))
    end

    started = time()
    deadline = started + cfg.duration
    probes = Task[]
    push!(probes, @async health_probe(ctx))
    cfg.pid === nothing || push!(probes, @async rss_probe(ctx))
    clients = map(1:cfg.clients) do i
        @async begin
            rng = MersenneTwister(cfg.seed + i)
            while time() < deadline
                timed_request!(ctx, rng, pick_kind(rng))
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
    println(io, rpad("kind", 15), lpad("n", 7), lpad("failed", 8), lpad("p50 ms", 10), lpad("p95 ms", 10), lpad("p99 ms", 10))
    for k in sort!(collect(keys(res.counts)))
        l = get(res.latency, k, nothing)
        f(x) = lpad(l === nothing ? "-" : string(round(x; digits=1)), 10)
        println(io, rpad(k, 15), lpad(res.counts[k], 7), lpad(get(res.failed, k, 0), 8),
            l === nothing ? "" : f(l.p50) * f(l.p95) * f(l.p99))
    end
    println(io)
    println(io, "health: max ", round(res.health_max_ms; digits=1), " ms")
    r = res.rss
    if r.peak_kb == 0
        println(io, "rss: not sampled")
    else
        mb(kb) = round(kb / 1024; digits=1)
        println(io, "rss MB: start ", mb(r.start_kb), "  peak ", mb(r.peak_kb), "  end ", mb(r.end_kb),
            "  slope ", isnan(r.slope_mb_min) ? "n/a" : "$(round(r.slope_mb_min; digits=2)) MB/min",
            r.judged ? "" : "  (too few samples to judge growth)")
    end
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
)

function main(args)
    kw = Dict{Symbol, Any}()
    log_path = "soak-events.jsonl"
    i = 1
    try
        while i <= length(args)
            a = args[i]
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
