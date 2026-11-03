// Reference dumper for the deepseek41 toy: runs token ids through a llama.cpp
// build and writes per-position logits, greedy ids and the per-layer taps the
// Julia forward also exposes (attn_out-L, ffn_out-L, l_out-L, result_norm,
// result_output) at full float32 precision.
//
// Build against a llama.cpp build tree (GESSO_GGML_DIR), for example:
//
//     g++ -std=c++17 -O1 -I$LLAMA/include -I$LLAMA/ggml/include \
//         tools/dump-deepseek41-ref.cpp -o dump-deepseek41-ref \
//         -L$LLAMA/build-ref/bin -lllama -lggml -lggml-base \
//         -Wl,-rpath,$LLAMA/build-ref/bin
//
// Run:  ./dump-deepseek41-ref model.gguf 1,2,3 out.txt
//
// Output format (one record per line):
//   tokens <T> <id> ...
//   argmax <t> <id>                       (per position, greedy over logits)
//   tap <name> <count> <v> ...            (%.9g, enough to round-trip float32)

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static const int kLayers = 6;
// attention-stage taps are captured for the first layers only (keeps the dump small)
static const int kStageLayers = 3;

static bool wanted(const char * name) {
    std::string s = name;
    if (s == "result_norm" || s == "result_output") {
        return true;
    }
    for (int il = 0; il < kLayers; ++il) {
        const std::string sfx = "-" + std::to_string(il);
        for (const char * base : {"attn_out", "ffn_out", "l_out"}) {
            if (s == base + sfx) {
                return true;
            }
        }
    }
    for (int il = 0; il < kStageLayers; ++il) {
        const std::string sfx = "-" + std::to_string(il);
        for (const char * base : {"attn_norm", "qr_norm", "q", "kv", "comp_latent", "comp_k",
                                  "attn_comp", "attn_derope", "attn_wo_a"}) {
            if (s == base + sfx) {
                return true;
            }
        }
    }
    return false;
}

// Reads a tensor as float32 whatever its storage type (F32, F16 or I32).
static bool read_as_float(const struct ggml_tensor * t, std::vector<float> & out) {
    const size_t n = (size_t) ggml_nelements(t);
    out.assign(n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        return true;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    if (t->type == GGML_TYPE_F16) {
        for (size_t i = 0; i < n; ++i) {
            uint16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            out[i] = ggml_fp16_to_fp32(h);
        }
        return true;
    }
    if (t->type == GGML_TYPE_I32) {
        for (size_t i = 0; i < n; ++i) {
            int32_t v;
            std::memcpy(&v, raw.data() + 4 * i, 4);
            out[i] = (float) v;
        }
        return true;
    }
    return false;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    if (!wanted(t->name)) {
        return false;
    }
    if (ask) {
        return true;
    }
    auto * taps = static_cast<std::map<std::string, std::vector<float>> *>(ud);
    std::vector<float> v;
    if (!read_as_float(t, v)) {
        std::fprintf(stderr, "skip tap %s: unsupported type %s\n", t->name, ggml_type_name(t->type));
        return false;
    }
    (*taps)[t->name] = std::move(v);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s model.gguf tok,tok,... out.txt\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[3];

    std::vector<llama_token> toks;
    {
        std::stringstream ss(argv[2]);
        std::string item;
        while (std::getline(ss, item, ',')) {
            toks.push_back((llama_token) std::atoi(item.c_str()));
        }
    }
    if (toks.empty()) {
        std::fprintf(stderr, "no tokens\n");
        return 2;
    }
    const int32_t T = (int32_t) toks.size();

    llama_backend_init();

    std::map<std::string, std::vector<float>> taps;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = (uint32_t) std::max(T, 16);
    cparams.n_batch      = (uint32_t) T;
    cparams.n_ubatch     = (uint32_t) T;
    cparams.no_perf      = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // full-precision KV cache: the default F16 cache rounds K/V and would hide a real diff
    cparams.type_k      = GGML_TYPE_F32;
    cparams.type_v      = GGML_TYPE_F32;
    cparams.cb_eval      = cb_eval;
    cparams.cb_eval_user_data = &taps;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(T, 0, 1);
    batch.n_tokens = T;
    for (int32_t i = 0; i < T; ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t V = llama_vocab_n_tokens(vocab);

    FILE * f = std::fopen(out_path, "w");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    std::fprintf(f, "tokens %d", T);
    for (llama_token t : toks) {
        std::fprintf(f, " %d", t);
    }
    std::fprintf(f, "\n");

    for (int32_t t = 0; t < T; ++t) {
        const float * row = llama_get_logits_ith(ctx, t);
        int32_t best = 0;
        for (int32_t v = 1; v < V; ++v) {
            if (row[v] > row[best]) {
                best = v;
            }
        }
        std::fprintf(f, "argmax %d %d\n", t, best);
    }

    // result_output is the same logits tensor; keep it as the logits tap
    for (const auto & [name, data] : taps) {
        std::fprintf(f, "tap %s %zu", name.c_str(), data.size());
        for (float x : data) {
            std::fprintf(f, " %.9g", (double) x);
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::printf("wrote %s: %d positions, %d vocab, %zu taps\n", out_path, T, V, taps.size());
    return 0;
}
