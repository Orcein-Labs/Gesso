# Launcher for the OpenAI-compatible chat server.
#
#   julia --project scripts/gesso-serve.jl --check
#       CPU load check. Starts the server on an ephemeral port with a fake
#       engine, asks it for /health, shuts down, and exits 0 when the answer is
#       right. It does not touch weights, a GPU or a tokenizer.
#
# The real-model path is not on this branch. The GGUF tokenizer and chat
# renderer are not merged here, so there is nothing for a real launch to load.

using Gesso
using Sockets

const CHECK_REPLY = "load check\n</think>\n\nok"

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

function main(args)
    if args == ["--check"]
        return run_check()
    end
    println(stderr, "usage: julia --project scripts/gesso-serve.jl --check")
    println(stderr, "The real-model launch is not on this branch (no GGUF tokenizer or chat renderer).")
    return 2
end

exit(main(ARGS))
