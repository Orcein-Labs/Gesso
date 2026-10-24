# Saluki 27B hidden-state capture: greedy decode of every prompt in
# CAPTURE_PROMPTS (saluki_capture_prompts.jl), recording the final-normed state
# behind each argmax step through reference_generate's on_hidden.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_CAPTURE_DIR=/path/to/capture \
#   julia --project=benchmark -t 16 scripts/saluki_capture_hidden.jl
#
# Optional env: GESSO_CAPTURE_NEW (new tokens per prompt, default 256),
# GESSO_CAPTURE_ONLY (comma list of prompt tags to run, default all).
#
# Per prompt i with tag t the directory gets:
#   NN_t.ids  the full id sequence (prompt + generated), raw little-endian Int32
#   NN_t.hid  Float32 matrix (dim, n_hidden), column-major, one column per
#             on_hidden call in call order
# and one line in index.tsv. A prompt whose .hid exists is skipped, so an
# interrupted run resumes. Both files are written under a temp name and moved
# into place, .hid last.

using Gesso
using Printf

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("saluki_capture_hidden: set GESSO_QWEN38_27B_GGUF")
const DIR = get(ENV, "GESSO_CAPTURE_DIR", "")
isempty(DIR) && error("saluki_capture_hidden: set GESSO_CAPTURE_DIR")
mkpath(DIR)
const NEW = parse(Int, get(ENV, "GESSO_CAPTURE_NEW", "256"))
const ONLY =
    Set(String.(filter(!isempty, strip.(split(get(ENV, "GESSO_CAPTURE_ONLY", ""), ',')))))
const IM_END = 248046
Base.ENDIAN_BOM == 0x04030201 || error("saluki_capture_hidden: little-endian host required")

include(joinpath(@__DIR__, "saluki_prompts.jl"))
include(joinpath(@__DIR__, "saluki_capture_prompts.jl"))

backend = Gesso.CPUBackend()
model, tensors, spec = Gesso.load_gguf(GGUF)
tk = Gesso.Inference.load_gguf_tokenizer(GGUF)
dim = model.embedding.dim
gen(prompt, n; kw...) = Gesso.reference_generate(
    model,
    tensors,
    prompt;
    max_new_tokens=n,
    eps=spec.rms_norm_eps,
    backend,
    eos_ids=(IM_END,),
    kw...,
)

index_path = joinpath(DIR, "index.tsv")
new_index = !isfile(index_path)
open(index_path, "a") do io
    new_index && println(
        io,
        join(
            [
                "idx",
                "tag",
                "prompt_len",
                "n_ids",
                "n_hidden",
                "dim",
                "first_pos",
                "seconds",
            ],
            '\t',
        ),
    )
    for (i, (tag, msg)) in enumerate(CAPTURE_PROMPTS)
        (isempty(ONLY) || tag in ONLY) || continue
        stem = joinpath(DIR, "$(lpad(string(i), 2, '0'))_$(tag)")
        if isfile(stem * ".hid")
            println("SKIP $tag (exists)")
            continue
        end
        prompt = Gesso.Inference.encode(tk, chat(msg))
        P = length(prompt)
        cols = Vector{Float32}[]
        firsts = Int[]
        hook = (pos, v) -> (push!(firsts, pos); push!(cols, v); nothing)
        t0 = time_ns()
        ids = gen(prompt, NEW; on_hidden=hook)
        secs = (time_ns() - t0) / 1e9

        n_ids = length(ids)
        n_hid = length(cols)
        n_hid == n_ids - P || error(
            "saluki_capture_hidden: $tag made $n_hid hidden calls for $(n_ids - P) new ids",
        )
        firsts == collect(P:(P+n_hid-1)) || error(
            "saluki_capture_hidden: $tag hidden positions are not P:(P+n-1) with P=$P (first=$(isempty(firsts) ? 0 : firsts[1]))",
        )

        H = Matrix{Float32}(undef, dim, n_hid)
        for (j, v) in enumerate(cols)
            H[:, j] = v
        end
        ids_tmp = stem * ".ids.tmp"
        open(f -> write(f, htol.(Int32.(ids))), ids_tmp, "w")
        mv(ids_tmp, stem * ".ids"; force=true)
        hid_tmp = stem * ".hid.tmp"
        open(f -> write(f, H), hid_tmp, "w")
        mv(hid_tmp, stem * ".hid"; force=true)

        println(
            io,
            join(
                [
                    i,
                    tag,
                    P,
                    n_ids,
                    n_hid,
                    dim,
                    isempty(firsts) ? 0 : firsts[1],
                    @sprintf("%.2f", secs)
                ],
                '\t',
            ),
        )
        flush(io)
        @printf(
            "CAPTURE %s prompt=%d new=%d hidden=%d seconds=%.1f\n",
            tag,
            P,
            n_ids - P,
            n_hid,
            secs
        )
        flush(stdout)
    end
end
