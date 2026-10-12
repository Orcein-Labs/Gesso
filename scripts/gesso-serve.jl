# Launcher for the OpenAI-compatible chat server.
#
#   julia --project scripts/gesso-serve.jl --check
#       CPU load check. Starts the server on an ephemeral port with a fake
#       engine, asks it for /health, shuts down, and exits 0 when the answer is
#       right. It does not touch weights, a GPU or a tokenizer.
#
#   julia --project scripts/gesso-serve.jl --gguf <path> [--host 127.0.0.1]
#         [--port 8080] [--n-ctx 65536] [--model-id ornith-35b] [--cpu]
#       Loads a GGUF and serves it. The path comes from --gguf or from
#       GESSO_ORNITH35_GGUF. GESSO_BACKEND=lava|cuda|cpu picks the backend and
#       defaults to lava; --cpu is the same as GESSO_BACKEND=cpu. Only the GPU
#       package that was asked for is loaded, so the package never depends on it.

using Gesso
using Sockets

const CHECK_REPLY = "load check\n</think>\n\nok"
const DEFAULTS = (host="127.0.0.1", port=8080, n_ctx=65536, model_id="ornith-35b")

function check_model()
    return ChatModel(
        tokenize=s -> Int.(codeunits(s)),
        detokenize=ids -> String(UInt8.(ids)),
        render=(msgs, tools, thinking) -> join((string(m["content"]) for m in msgs), "\n"),
        generate=(prompt_ids; max_new_tokens, on_token) -> begin
            out = Int[]
            for b in codeunits(CHECK_REPLY)
                length(out) >= max_new_tokens && break
                push!(out, Int(b))
                on_token(Int(b)) === false && break
            end
            return (generated=out, cached=0)
        end,
        n_ctx=4096,
        model_id="gesso-check",
    )
end

function run_check()
    srv = serve(check_model(); host="127.0.0.1", port=0, block=false)
    try
        sock = Sockets.connect(Sockets.localhost, srv.port)
        write(sock, "GET /health HTTP/1.1\r\nHost: check\r\nConnection: close\r\n\r\n")
        resp = read(sock, String)
        close(sock)
        if !(occursin("200 OK", resp) && occursin("\"ok\"", resp))
            println(stderr, "gesso-serve check failed: /health did not answer 200 ok")
            return 1
        end
        println("gesso-serve check ok (ephemeral port $(srv.port), fake engine)")
        return 0
    finally
        close(srv)
    end
end

const USAGE = """
usage: julia --project scripts/gesso-serve.jl --check
       julia --project scripts/gesso-serve.jl --gguf <path> [--host H] [--port P] [--n-ctx N] [--model-id ID] [--cpu]"""

# Flags take a value, except --cpu and --check. Anything unknown is an error,
# because a mistyped flag silently ignored would serve the wrong model.
function parse_launch(args)
    opts = Dict{String,Any}("cpu" => false)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--cpu"
            opts["cpu"] = true
        elseif a in ("--gguf", "--host", "--port", "--n-ctx", "--model-id")
            i < length(args) || throw(ArgumentError("$a needs a value"))
            opts[a] = args[i + 1]
            i += 1
        else
            throw(ArgumentError("unknown argument $a"))
        end
        i += 1
    end
    return opts
end

function resolve_gguf(opts)
    path = get(opts, "--gguf", get(ENV, "GESSO_ORNITH35_GGUF", ""))
    isempty(path) && throw(ArgumentError(
        "no GGUF given: pass --gguf <path> or set GESSO_ORNITH35_GGUF"))
    isfile(path) || throw(ArgumentError("GGUF not found: $path"))
    return path
end

# The GPU package is loaded at run time, so the script needs no top-level `using`
# and a cpu run never touches a GPU stack. The backend binding exists only once
# the extension is loaded, hence the invokelatest calls.
function gpu_backend(which::AbstractString)
    pkg, ctor = which == "lava" ? (:Lava, :LavaBackend) : (:CUDA, :CUDABackend)
    Core.eval(Main, :(using $pkg))
    return Base.invokelatest(() -> getfield(Gesso, ctor)())
end

function launch(opts)
    path = resolve_gguf(opts)
    host = get(opts, "--host", DEFAULTS.host)
    port = parse(Int, get(opts, "--port", string(DEFAULTS.port)))
    n_ctx = parse(Int, get(opts, "--n-ctx", string(DEFAULTS.n_ctx)))
    model_id = get(opts, "--model-id", DEFAULTS.model_id)

    which = opts["cpu"] ? "cpu" : get(ENV, "GESSO_BACKEND", "lava")
    # Fail on a bad backend name before spending minutes on the weights.
    which in ("lava", "cuda", "cpu") ||
        throw(ArgumentError("GESSO_BACKEND must be lava, cuda or cpu, got '$which'"))
    model, tensors, spec = Gesso.load_gguf(path)
    tk = Gesso.Inference.load_gguf_tokenizer(path)
    which == "cpu" && return serve_loaded(model, tensors, spec, tk, Gesso.CPUBackend(), host, port, n_ctx, model_id)
    backend = gpu_backend(which)
    # The GPU methods are newer than this frame. Every later call dispatches on
    # the backend, the warm-up and the server tasks included, so the rest of the
    # launch runs in the latest world instead of wrapping each call.
    tensors = Base.invokelatest(() -> Gesso.to_device(backend, tensors))
    return Base.invokelatest(serve_loaded, model, tensors, spec, tk, backend, host, port, n_ctx, model_id)
end

function serve_loaded(model, tensors, spec, tk, backend, host, port, n_ctx, model_id)
    chat = gguf_chat_model(model, tensors, spec, tk; backend=backend, n_ctx=n_ctx, model_id=model_id)

    # One short generation before the port opens, so the first client does not
    # pay for kernel setup and the GPU allocations that come with it.
    warm = chat.render([Dict{String,Any}("role" => "user", "content" => "hi")], Any[], false)
    chat.generate(chat.tokenize(warm); max_new_tokens=4, on_token=_ -> true)

    srv = serve(chat; host=host, port=port, block=false)
    println("gesso-serve ready on $(srv.host):$(srv.port) n_ctx=$(n_ctx) model=$(model_id)")
    flush(stdout)
    wait(srv.task)
    return 0
end

function main(args)
    if args == ["--check"]
        return run_check()
    end
    try
        return launch(parse_launch(args))
    catch e
        e isa ArgumentError || rethrow()
        println(stderr, "gesso-serve: ", e.msg)
        println(stderr, USAGE)
        return 2
    end
end

exit(main(ARGS))
