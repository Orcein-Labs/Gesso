# OpenAI-compatible chat server over a Gesso ChatModel.
#
# The server does not know the engine. A ChatModel carries the tokenizer, the
# chat renderer and a generate closure; everything here is HTTP framing, request
# mapping and the streaming parser. Keeping the engine behind closures lets the
# parser and the socket layer run against fakes, and keeps tokenizer and
# template code out of this directory.

module Serve

using JSON
using Sockets

include("http.jl")
include("qwen35_xml.jl")
include("chatmodel.jl")
include("server.jl")
include("handlers.jl")

export ChatModel, serve

end
