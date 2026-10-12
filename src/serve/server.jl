# Listener, connection loop and dispatch. One listener task accepts; each
# connection gets its own task; generation is single-flight behind `gate`.

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
    gate::ReentrantLock
    task::Union{Nothing,Task}
end

"""
    serve(model; host="127.0.0.1", port=8080, block=true) -> Server

Listen on `host:port` and serve `model`. With `block=true` this call returns when
the server is closed. With `block=false` it returns at once with the listener
running in a task. `port=0` picks a free port, reported in `srv.port`. `host`
must be an IPv4 address.
"""
function serve(model::ChatModel; host::AbstractString="127.0.0.1", port::Integer=8080,
        block::Bool=true)
    ip = parse(Sockets.IPv4, host)
    listener = Sockets.listen(ip, port)
    _, bound = Sockets.getsockname(listener)
    srv = Server(model, listener, String(host), Int(bound), ReentrantLock(), nothing)
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
        @async _handle_connection(srv, sock)
    end
    return nothing
end

function _handle_connection(srv::Server, sock::TCPSocket)
    try
        while _serve_one(srv, sock)
        end
    catch e
        e isa Base.IOError || e isa EOFError || @error "connection failed" exception=(e, catch_backtrace())
    finally
        close(sock)
    end
    return nothing
end

# Returns true when the connection stays open for another request.
function _serve_one(srv::Server, sock::TCPSocket)::Bool
    req = try
        read_request(sock)
    catch e
        e isa HTTPError || rethrow()
        _try_write(() -> write_error(sock, e.status, e.message; keepalive=false,
            code=e.code, param=e.param))
        return false
    end
    req === nothing && return false

    keepalive = _wants_keepalive(req)
    try
        return route(srv, req, sock, keepalive)
    catch e
        if e isa HTTPError
            _try_write(() -> write_error(sock, e.status, e.message; keepalive=keepalive,
                code=e.code, param=e.param))
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

# Used only to send an error response. The socket may already be gone, and an
# error response to a dead client has no one to receive it.
function _try_write(f)
    try
        f()
    catch
    end
    return nothing
end
