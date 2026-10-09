# Qwen3.8-27B (dense qwen35, UD-IQ2_M) greedy decode on the CPU backend.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Qwen3.8-27B-UD-IQ2_M.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/gen_qwen38_27b.jl
#
# Prints one `IDS` line (prompt followed by the greedy continuation) and one
# `MEM` line with the process peak RSS after the load and after the decode.
# The prompt defaults to token 9419 ("Hello" in the Qwen3.5 vocabulary, the
# same id the Ornith gate uses). Override with GESSO_27B_PROMPT=id,id,...
# The MTP block is not executed.
#
# Fallback note: IQ2_S, IQ3_S, IQ1_M, Q2_K and Q3_K matmuls dequantize per call
# through ggml `to_float` (logged once per type). This is a correctness bridge,
# so decode is slow.

using Gesso

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("gen_qwen38_27b: set GESSO_QWEN38_27B_GGUF to the Qwen3.8-27B GGUF")
const PROMPT = [parse(Int, s) for s in split(get(ENV, "GESSO_27B_PROMPT", "9419"), ',')]
const NEW = parse(Int, get(ENV, "GESSO_27B_NEW", "4"))

function _peak_rss_mib()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") || continue
        kib = parse(Int, split(line)[2])
        return round(kib / 1024; digits=1)
    end
    return missing
end

function _rss_mib()
    for line in eachline("/proc/self/status")
        startswith(line, "VmRSS:") || continue
        kib = parse(Int, split(line)[2])
        return round(kib / 1024; digits=1)
    end
    return missing
end

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
end
println("MEM after_load rss_mib=$(_rss_mib()) peak_rss_mib=$(_peak_rss_mib()) load_s=$(round(t_load; digits=1))")

t_gen = @elapsed ids = Gesso.reference_generate(
    model, tensors, PROMPT; max_new_tokens=NEW, eps=spec.rms_norm_eps,
    backend=Gesso.CPUBackend(),
)
println("IDS ", ids)
println("MEM after_decode rss_mib=$(_rss_mib()) peak_rss_mib=$(_peak_rss_mib()) decode_s=$(round(t_gen; digits=1))")
println("MTP ", tensors.mtp === nothing ? "none" : (layer=tensors.mtp.layer, n_predict=tensors.mtp.n_predict, n_tensors=length(tensors.mtp.tensor_names)))
