# Ornith-1.5-35B-A3B Hello ID-gate + warmed decode on one backend.
#
#   GESSO_ORNITH35_GGUF=/path/to/Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   GESSO_BACKEND=lava|cuda|cpu \
#   julia --project=benchmark -t 16 scripts/gate_35b.jl
#
# GESSO_BACKEND defaults to lava. Only the requested GPU package is loaded:
# CUDA is never pulled in unless it is asked for. The benchmark workspace env
# carries both GPU packages; the root project lists them as weak dependencies,
# so `--project=.` cannot load either.
#
# Prints one RECEIPT line. Exit 1 if the greedy ids do not match.
# Decode rate excludes prefill: (t(8 tokens) - t(1 token)) / 7, on a warmed
# second pass. Reference: llama.cpp CUDA tg64 ngl99 ncmoe24 fa = 32.80 tok/s.

using Gesso

const GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")
isfile(GGUF) || error("gate_35b: set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF file")
const EXPECTED = [9419, 11, 271, 40, 2688, 4313, 310, 958, 279]
const NEW = length(EXPECTED) - 1

which = get(ENV, "GESSO_BACKEND", "lava")
if which == "lava"
    using Lava
    backend = Gesso.LavaBackend()
elseif which == "cuda"
    using CUDA
    backend = Gesso.CUDABackend()
elseif which == "cpu"
    backend = Gesso.CPUBackend()
else
    error("gate_35b: GESSO_BACKEND must be lava, cuda or cpu, got '$which'")
end

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
    dev = which == "cpu" ? tensors : Gesso.to_device(backend, tensors)
end

gen(n) = Gesso.reference_generate(
    model, dev, [9419]; max_new_tokens=n, eps=spec.rms_norm_eps, backend=backend,
)

ids = gen(NEW)          # compile + correctness
gen(1)
t1 = @elapsed gen(1)
t8 = @elapsed gen(NEW)
tok_s = (NEW - 1) / (t8 - t1)

ok = ids == EXPECTED
println(
    "RECEIPT gate_35b backend=$which threads=$(Threads.nthreads()) ",
    "ids_ok=$ok load_s=$(round(t_load; digits=1)) ",
    "prefill_1tok_s=$(round(t1; digits=3)) decode_tok_s=$(round(tok_s; digits=3))",
)
ok || (println("ids got  $ids\nids want $EXPECTED"); exit(1))
