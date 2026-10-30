# Listener, connection loop and dispatch. One listener task accepts; each
# connection gets its own task; generation is single-flight behind `gate`, which
# holds a bounded queue of requests waiting for their turn.

"""
    GenerationGate

Single-flight access to the engine. `busy` is set while one generation runs.
Requests that arrive meanwhile wait in a queue of at most `max_queue`, each for
at most `queue_timeout` seconds. `changed` wakes the waiters whenever the slot is
released or a waiter's client goes away.
"""
mutable struct GenerationGate
    lock::ReentrantLock
    changed::Threads.Condition
    busy::Bool
    waiting::Int
    max_queue::Int
    queue_timeout::Float64
end

function GenerationGate(max_queue::Int, queue_timeout::Float64)
    lock = ReentrantLock()
    return GenerationGate(lock, Threads.Condition(lock), false, 0, max_queue, queue_timeout)
end

"""
    Server

A running listener. `srv.port` is the bound port, which is the real one when the
server was started with port 0.
"""
mutable struct Server
    model::ChatModel
    listener::Sockets.TCPServer
    host::String
    port::Int
    gate::GenerationGate
    task::Union{Nothing, Task}
    read_timeout::Float64
    max_connections::Int
    open_connections::Threads.Atomic{Int}
end

"""
    serve(model; host="127.0.0.1", port=8080, block=true,
          read_timeout=30, max_connections=64,
          max_queue=16, queue_timeout=300) -> Server

Listen on `host:port` and serve `model`. With `block=true` this call returns when
the server is closed. With `block=false` it returns at once with the listener
running in a task. `port=0` picks a free port, reported in `srv.port`. `host`
must be an IPv4 address.

`read_timeout` is how many seconds a connection gets to deliver each whole
request, from the moment the server starts waiting for it. That covers an idle
keep-alive connection and a client that sends its request a byte at a time.
When it runs out the connection is closed. `max_connections` caps how many
connections are open at once; past it a new connection gets a 503 and is
closed. Neither limit applies while a response is being written.

Generation runs one request at a time. `max_queue` caps how many chat requests
may wait behind the one running; a request arriving to a full queue gets a 503
with `Retry-After`. `queue_timeout` is how many seconds a queued request waits
for its turn before it gets a 503 and its connection is closed. A client that
disconnects while queued leaves the queue at once. `max_queue=0` refuses every
chat request that arrives while another is running.
"""
function serve(
    model::ChatModel;
    host::AbstractString="127.0.0.1",
    port::Integer=8080,
    block::Bool=true,
    read_timeout::Real=30,
    max_connections::Integer=64,
    max_queue::Integer=16,
    queue_timeout::Real=300,
)
    read_timeout > 0 || throw(ArgumentError("read_timeout must be positive"))
    max_connections > 0 || throw(ArgumentError("max_connections must be positive"))
    max_queue >= 0 || throw(ArgumentError("max_queue must not be negative"))
    queue_timeout > 0 || throw(ArgumentError("queue_timeout must be positive"))
    ip = parse(Sockets.IPv4, host)
    listener = Sockets.listen(ip, port)
    _, bound = Sockets.getsockname(listener)
    srv = Server(
        model,
        listener,
        String(host),
        Int(bound),
        GenerationGate(Int(max_queue), Float64(queue_timeout)),
        nothing,
        Float64(read_timeout),
        Int(max_connections),
        Threads.Atomic{Int}(0),
    )
    if block
        _accept_loop(srv)
    else
        srv.task = @async _accept_loop(srv)
    end
    return srv
end

"""
    close(srv::Server)

Stop accepting connections. Connections already open finish their current request.
"""
Base.close(srv::Server) = close(srv.listener)

function _accept_loop(srv::Server)
    while isopen(srv.listener)
        sock = try
            accept(srv.listener)
        catch
            break
        end
        if Threads.atomic_add!(srv.open_connections, 1) >= srv.max_connections
            Threads.atomic_sub!(srv.open_connections, 1)
            @async _refuse_connection(sock)
            continue
        end
        @async _handle_connection(srv, sock)
    end
    return nothing
end

# The refusal is written from its own task so a client that never reads cannot
# stall the accept loop.
function _refuse_connection(sock::TCPSocket)
    _try_write(() -> write_error(sock, 503, "too many open connections"; keepalive=false))
    _try_write(() -> close(sock))
    return nothing
end

function _handle_connection(srv::Server, sock::TCPSocket)
    try
        while _serve_one(srv, sock)
        end
    catch e
        e isa Base.IOError ||
            e isa EOFError ||
            @error "connection failed" exception=(e, catch_backtrace())
    finally
        Threads.atomic_sub!(srv.open_connections, 1)
        close(sock)
    end
    return nothing
end

# Reads one request under the read deadline. A timer closes the socket if the
# request is not complete in time; the blocked read then fails, and the
# connection ends without a reply because there is no one left to send it to.
function _read_request_timed(srv::Server, sock::TCPSocket)
    timed_out = Threads.Atomic{Bool}(false)
    timer = Timer(srv.read_timeout) do _
        timed_out[] = true
        close(sock)
    end
    try
        return read_request(sock), false
    catch e
        timed_out[] && return nothing, true
        rethrow()
    finally
        close(timer)
    end
end

# Returns true when the connection stays open for another request.
function _serve_one(srv::Server, sock::TCPSocket)::Bool
    req, timed_out = try
        _read_request_timed(srv, sock)
    catch e
        e isa HTTPError || rethrow()
        _try_write(
            () -> write_error(
                sock,
                e.status,
                e.message;
                keepalive=false,
                code=e.code,
                param=e.param,
            ),
        )
        return false
    end
    (timed_out || req === nothing) && return false

    keepalive = _wants_keepalive(req)
    try
        return route(srv, req, sock, keepalive)
    catch e
        if e isa HTTPError
            _try_write(
                () -> write_error(
                    sock,
                    e.status,
                    e.message;
                    keepalive=keepalive,
                    code=e.code,
                    param=e.param,
                ),
            )
            return keepalive
        elseif e isa Base.IOError || e isa EOFError
            return false
        end
        @error "request failed" exception=(e, catch_backtrace())
        _try_write(() -> write_error(sock, 500, "internal error"; keepalive=false))
        return false
    end
end

_wants_keepalive(req::HTTPRequest) = begin
    conn = lowercase(get(req.headers, "connection", ""))
    req.version == "HTTP/1.1" ? conn != "close" : conn == "keep-alive"
end

# Waits for the generation slot. Returns :acquired (the caller must then call
# _release_slot!), :full when the queue is at its limit, :timeout when the wait
# outlasted the queue timeout, or :cancelled when the client left while queued.
# A request that arrives while others wait joins the back of the queue even if
# the slot is free, so a queued request is not overtaken.
function _acquire_slot!(gate::GenerationGate, cancelled::Ref{Bool})::Symbol
    lock(gate.lock) do
        if !gate.busy && gate.waiting == 0
            gate.busy = true
            return :acquired
        end
        gate.waiting >= gate.max_queue && return :full
        gate.waiting += 1
        expired = Ref(false)
        timer = Timer(gate.queue_timeout) do _
            expired[] = true
            _wake_waiters(gate)
        end
        try
            while true
                if !gate.busy
                    gate.busy = true
                    return :acquired
                end
                cancelled[] && return :cancelled
                expired[] && return :timeout
                wait(gate.changed)
            end
        finally
            gate.waiting -= 1
            close(timer)
        end
    end
end

function _release_slot!(gate::GenerationGate)
    lock(gate.lock) do
        gate.busy = false
        notify(gate.changed)
    end
    return nothing
end

# Called when a queued client goes away, so its waiter leaves the queue now
# instead of when its turn would have come.
function _wake_waiters(gate::GenerationGate)
    lock(gate.lock) do
        notify(gate.changed)
    end
    return nothing
end

# Used only to send an error response. The socket may already be gone, and an
# error response to a dead client has no one to receive it.
function _try_write(f)
    try
        f()
    catch
    end
    return nothing
end
