# OpenAI-compatible chat server over a Gesso ChatModel.
#
# The HTTP framing, request mapping and streaming parser live here and run
# against fakes in the tests. The engine enters as a ChatModel, a bundle of
# closures. engine.jl builds one from a GGUF; nothing else in this directory
# touches weights, a GPU or a tokenizer.

module Serve

using JSON
using Sockets
using ..Inference: _chat_json_parse

include("http.jl")
include("qwen35_xml.jl")
include("chatmodel.jl")
include("engine.jl")
include("server.jl")
include("handlers.jl")

export ChatModel, serve, gguf_chat_model

end
