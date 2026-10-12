# Hand-written HTTP/1.1 framing. The package takes no HTTP dependency: the
# server needs exact control of keep-alive, chunked request bodies and chunked
# SSE responses, and each of those is a few dozen lines here.

const MAX_HEADER_BYTES = 64 * 1024
const MAX_BODY_BYTES = 32 * 1024 * 1024

struct HTTPError <: Exception
    status::Int
    message::String
    code::Union{Nothing,String}
    param::Union{Nothing,String}
end

HTTPError(status::Int, message::AbstractString; code=nothing, param=nothing) =
    HTTPError(status, String(message), code === nothing ? nothing : String(code),
        param === nothing ? nothing : String(param))

struct HTTPRequest
    method::String
    target::String
    version::String
    headers::Dict{String,String}
    body::Vector{UInt8}
end

const _REASONS = Dict{Int,String}(
    200 => "OK",
    400 => "Bad Request",
    404 => "Not Found",
    405 => "Method Not Allowed",
    413 => "Payload Too Large",
    431 => "Request Header Fields Too Large",
    500 => "Internal Server Error",
)

# Returns `nothing` on a clean close before any byte of a line arrived. The
# header budget is shared across the whole request head so a client cannot
# stream headers forever.
function _read_line(io::IO, budget::Ref{Int})
    buf = UInt8[]
    while true
        if eof(io)
            isempty(buf) && return nothing
            throw(HTTPError(400, "connection closed mid-line"))
        end
        b = read(io, UInt8)
        budget[] -= 1
        budget[] < 0 && throw(HTTPError(431, "header section too large"))
        b == UInt8('\n') && break
        push!(buf, b)
    end
    if !isempty(buf) && buf[end] == UInt8('\r')
        pop!(buf)
    end
    return String(buf)
end

# Some clients keep the CRLF that ended the previous body; RFC 9112 2.2 says to
# skip empty lines before a request line.
function read_request(io::IO)
    budget = Ref(MAX_HEADER_BYTES)
    line = _read_line(io, budget)
    while line !== nothing && isempty(line)
        line = _read_line(io, budget)
    end
    line === nothing && return nothing

    parts = split(line, ' ')
    length(parts) == 3 || throw(HTTPError(400, "malformed request line"))
    method, target, version = String.(parts)
    startswith(version, "HTTP/1.") || throw(HTTPError(400, "unsupported HTTP version"))

    headers = Dict{String,String}()
    while true
        h = _read_line(io, budget)
        h === nothing && throw(HTTPError(400, "connection closed inside headers"))
        isempty(h) && break
        colon = findfirst(':', h)
        colon === nothing && throw(HTTPError(400, "malformed header line"))
        headers[lowercase(strip(h[1:(colon - 1)]))] = strip(h[(colon + 1):end])
    end

    # Clients such as curl hold a large body until the server agrees.
    if lowercase(get(headers, "expect", "")) == "100-continue"
        write(io, "HTTP/1.1 100 Continue\r\n\r\n")
        flush(io)
    end
    body = _read_body(io, headers, budget)
    return HTTPRequest(method, target, version, headers, body)
end

function _read_body(io::IO, headers::Dict{String,String}, budget::Ref{Int})
    if occursin("chunked", lowercase(get(headers, "transfer-encoding", "")))
        return _read_chunked(io, budget)
    end
    haskey(headers, "content-length") || return UInt8[]
    n = tryparse(Int, headers["content-length"])
    (n === nothing || n < 0) && throw(HTTPError(400, "invalid Content-Length"))
    n > MAX_BODY_BYTES && throw(HTTPError(413, "request body too large"))
    try
        return read(io, n)
    catch e
        e isa EOFError && throw(HTTPError(400, "body shorter than Content-Length"))
        rethrow()
    end
end

function _read_chunked(io::IO, budget::Ref{Int})
    body = UInt8[]
    while true
        # Each chunk-size line has its own small budget, so a body of many chunks
        # is not charged against the header limit.
        line = _read_line(io, Ref(MAX_HEADER_BYTES))
        line === nothing && throw(HTTPError(400, "connection closed inside chunked body"))
        size_text = strip(first(split(line, ';'; limit=2)))
        n = tryparse(Int, size_text; base=16)
        (n === nothing || n < 0) && throw(HTTPError(400, "invalid chunk size"))
        n == 0 && break
        length(body) + n > MAX_BODY_BYTES && throw(HTTPError(413, "request body too large"))
        try
            append!(body, read(io, n))
        catch e
            e isa EOFError && throw(HTTPError(400, "chunk shorter than its size"))
            rethrow()
        end
        trailer = _read_line(io, Ref(MAX_HEADER_BYTES))
        (trailer === nothing || !isempty(trailer)) && throw(HTTPError(400, "chunk not followed by CRLF"))
    end
    # Trailers are ignored; the request ends at the first empty line.
    while true
        t = _read_line(io, budget)
        t === nothing && throw(HTTPError(400, "connection closed inside trailers"))
        isempty(t) && break
    end
    return body
end

function _write_head(io::IO, status::Int, headers::Vector{Pair{String,String}})
    write(io, "HTTP/1.1 ", string(status), " ", get(_REASONS, status, "Unknown"), "\r\n")
    for (k, v) in headers
        write(io, k, ": ", v, "\r\n")
    end
    write(io, "\r\n")
    flush(io)
    return nothing
end

_connection_header(keepalive::Bool) = keepalive ? "keep-alive" : "close"

function write_json(io::IO, status::Int, obj; keepalive::Bool=true)
    body = Vector{UInt8}(codeunits(JSON.json(obj)))
    _write_head(io, status, [
        "Content-Type" => "application/json",
        "Content-Length" => string(length(body)),
        "Connection" => _connection_header(keepalive),
    ])
    write(io, body)
    flush(io)
    return nothing
end

function write_error(io::IO, status::Int, message::AbstractString; keepalive::Bool=true,
        code::Union{Nothing,AbstractString}=nothing, param::Union{Nothing,AbstractString}=nothing)
    kind = status == 404 ? "not_found_error" : status >= 500 ? "server_error" : "invalid_request_error"
    err = Dict{String,Any}("message" => message, "type" => kind,
        "code" => code === nothing ? status : code)
    param === nothing || (err["param"] = param)
    write_json(io, status, Dict("error" => err); keepalive=keepalive)
end

# Chunked framing for SSE: each event is one chunk, so the connection can be
# reused after the stream ends. A bare SSE body with no length would force the
# client to read until close.
function start_event_stream(io::IO; keepalive::Bool)
    _write_head(io, 200, [
        "Content-Type" => "text/event-stream",
        "Cache-Control" => "no-cache",
        "Transfer-Encoding" => "chunked",
        "Connection" => _connection_header(keepalive),
    ])
    return nothing
end

function write_chunk(io::IO, text::AbstractString)
    write(io, string(ncodeunits(text), base=16), "\r\n", text, "\r\n")
    flush(io)
    return nothing
end

write_sse(io::IO, data::AbstractString) = write_chunk(io, "data: " * data * "\n\n")

function end_event_stream(io::IO)
    write(io, "0\r\n\r\n")
    flush(io)
    return nothing
end
