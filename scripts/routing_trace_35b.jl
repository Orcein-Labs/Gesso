# Record Ornith-1.5-35B-A3B expert routing for a prompt (GESSO_ROUTING_TRACE).
#
#   GESSO_ORNITH35_GGUF=/path/to/Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   GESSO_ROUTING_TRACE=/path/to/trace.txt \
#   GESSO_TRACE_PROMPT=9419,11,271 GESSO_TRACE_TOKENS=16 \
#   julia --project=benchmark -t 16 scripts/routing_trace_35b.jl
#
# Writes one line per (layer, token row) to GESSO_ROUTING_TRACE; turn it into a
# profile with scripts/routing_profile.jl. GESSO_TRACE_PROMPT is comma-separated
# token ids (required). Runs on the CUDA backend, where the trace is recorded.

using Gesso
using CUDA

const GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")
isfile(GGUF) || error("routing_trace_35b: set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF file")
const GGML_DIR = get(ENV, "GESSO_GGML_DIR", "")
isdir(GGML_DIR) || error("routing_trace_35b: set GESSO_GGML_DIR to the llama.cpp build/bin directory")
isempty(get(ENV, "GESSO_ROUTING_TRACE", "")) && error("routing_trace_35b: set GESSO_ROUTING_TRACE")
const PROMPT = [parse(Int, s) for s in split(get(ENV, "GESSO_TRACE_PROMPT", ""), ',')]
isempty(PROMPT) && error("routing_trace_35b: set GESSO_TRACE_PROMPT to comma-separated token ids")
const NTOK = parse(Int, get(ENV, "GESSO_TRACE_TOKENS", "16"))

backend = Gesso.CUDABackend()
model, tensors, spec = Gesso.load_gguf(GGUF)
dev = Gesso.to_device(backend, tensors)
ids = Gesso.reference_generate(
    model, dev, PROMPT; max_new_tokens=NTOK, eps=spec.rms_norm_eps, backend=backend,
)
println("RECEIPT routing_trace_35b prompt=", join(PROMPT, ","), " generated=", join(ids, ","))
