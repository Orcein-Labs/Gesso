# Ornith-1.5-35B-A3B Hello ID-gate + warmed decode on one backend.
#
#   flock /tmp/gesso-gpu.lock julia --project=. -t 16 scripts/gate_35b.jl [cuda|cpu]
#
# Prints one RECEIPT line. Exit 1 if the greedy ids do not match.
# Decode rate excludes prefill: (t(8 tokens) - t(1 token)) / 7, on a warmed
# second pass. Reference: llama.cpp CUDA tg64 ngl99 ncmoe24 fa = 32.80 tok/s.

using Gesso

const GGUF = get(
    ENV,
    "GESSO_ORNITH35_GGUF",
    joinpath(homedir(), ".cache", "gesso", "Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf"),
)
const EXPECTED = [9419, 11, 271, 40, 2688, 4313, 310, 958, 279]
const NEW = length(EXPECTED) - 1

which = isempty(ARGS) ? "cuda" : ARGS[1]
if which == "cuda"
    using CUDA
    backend = Gesso.CUDABackend()
else
    backend = Gesso.CPUBackend()
end

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
    dev = which == "cuda" ? Gesso.to_device(backend, tensors) : tensors
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
