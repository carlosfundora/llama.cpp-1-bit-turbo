// spec-harness: Reusable speculative decoding validation tool.
//
// Capture mode: Run target model with EAGLE3 extraction enabled,
// dump features + tokens to binary for offline validation.
//
// Usage:
//   llama-spec-harness -m <target.gguf> -md <eagle3.gguf> -ngl 99 -ngld 99
//     -p "prompt" -n 50 --harness-output /tmp/spec_harness/run.bin

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

// Binary capture format
// Header:  magic[4]="SPEC", version(u32), n_embd(u32), n_layers(u32), layer_ids[3](i32)
// Per-token record:
//   token_id(i32), next_token_id(i32), features(float × n_layers × n_embd)

static const char SPEC_MAGIC[4] = {'S','P','E','C'};
static const uint32_t SPEC_VERSION = 1;

struct capture_header {
    char     magic[4];
    uint32_t version;
    uint32_t n_embd;
    uint32_t n_layers;
    int32_t  layer_ids[3];
    uint32_t n_records;  // filled after capture
};

struct capture_record {
    int32_t  token_id;
    int32_t  next_token_id;
    // followed by: float features[n_layers * n_embd]
};

// Default prompts for batch mode
static const std::vector<std::string> DEFAULT_PROMPTS = {
    "The capital of France is",
    "In quantum mechanics, the uncertainty principle states that",
    "The largest planet in our solar system is",
    "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    if n <=",
    "The process of photosynthesis converts sunlight into",
    "According to Einstein's theory of general relativity,",
    "The Fibonacci sequence begins with 0, 1, and each subsequent number is",
    "In machine learning, gradient descent is an optimization algorithm that",
    "The human genome contains approximately",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON",
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < 1) {
        params.n_predict = 50;
    }

    // Output path from SPEC_HARNESS_OUTPUT env or default
    std::string output_path = "/tmp/spec_harness/capture.bin";
    const char * env_output = std::getenv("SPEC_HARNESS_OUTPUT");
    if (env_output) {
        output_path = env_output;
    }

    LOG_INF("spec_harness: output = %s\n", output_path.c_str());
    LOG_INF("spec_harness: n_predict = %d\n", params.n_predict);

    // Init backend
    llama_backend_init();
    llama_numa_init(params.numa);

    // Load target model
    auto llama_init_tgt = common_init_from_params(params);
    llama_model  * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt  = llama_init_tgt->context();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // Load draft model (EAGLE3) — just for metadata (extract layer IDs, n_embd)
    llama_model_ptr model_dft;
    {
        auto params_dft = params;
        params_dft.model = params.speculative.mparams_dft;
        params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
        auto mparams_dft = common_model_params_to_llama(params_dft);
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (!model_dft) {
            LOG_ERR("Failed to load draft model\n");
            return 1;
        }
    }

    // Enable EAGLE3 extraction on target context
    llama_set_eagle3(ctx_tgt, model_dft.get());

    const int n_aux = llama_model_eagle3_n_aux_layers(model_dft.get());
    const int n_embd = llama_model_n_embd(model_dft.get());
    const int feat_size = n_aux * n_embd;  // floats per token

    LOG_INF("spec_harness: n_aux_layers=%d, n_embd=%d, feat_size=%d\n", n_aux, n_embd, feat_size);

    // Get extract layer IDs from the model metadata (Bonsai-4B: [1, 18, 35])

    // Prepare prompts
    std::vector<std::string> prompts;
    if (!params.prompt.empty()) {
        prompts.push_back(params.prompt);
    } else {
        prompts = DEFAULT_PROMPTS;
    }

    // Sampler for greedy decoding
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // Capture data: all records across all prompts
    struct record_t {
        int32_t token_id;
        int32_t next_token_id;
        std::vector<float> features;
    };
    std::vector<record_t> all_records;

    int total_tokens = 0;

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        LOG_INF("\n=== Prompt %zu/%zu: \"%s\" ===\n", pi + 1, prompts.size(), prompts[pi].c_str());

        // Tokenize
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompts[pi], true, true);
        if (inp.empty()) continue;

        LOG_INF("  tokens: %zu\n", inp.size());

        // Reset context for each prompt
        llama_memory_clear(llama_get_memory(ctx_tgt), false);
        common_sampler_reset(smpl);

        // Process prompt (all except last token)
        if (inp.size() > 1) {
            if (llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1)) != 0) {
                LOG_ERR("  prompt eval failed\n");
                continue;
            }
        }

        // Get features from prompt eval (if extraction worked)
        int32_t n_feat_prompt = 0;
        const float * feat_prompt = llama_get_eagle3_target_features(ctx_tgt, &n_feat_prompt);
        if (feat_prompt && n_feat_prompt > 0) {
            int n_prompt_tokens = n_feat_prompt / feat_size;
            LOG_INF("  prompt features: %d tokens × %d floats\n", n_prompt_tokens, feat_size);
        }

        // Generate tokens one at a time, capturing features at each step
        llama_token id_last = inp.back();
        (void)inp;
        // int n_past = inp.size() - 1;

        for (int i = 0; i < params.n_predict; i++) {
            // Decode the current token
            llama_batch batch = llama_batch_get_one(&id_last, 1);
            if (llama_decode(ctx_tgt, batch) != 0) {
                LOG_ERR("  decode failed at step %d\n", i);
                break;
            }
            // n_past++;

            // Extract features for this token
            int32_t n_feat = 0;
            const float * features = llama_get_eagle3_target_features(ctx_tgt, &n_feat);

            // Sample next token (greedy)
            common_sampler_accept(smpl, id_last, false);
            llama_token id_next = common_sampler_sample(smpl, ctx_tgt, 0);

            // Print generated text
            LOG("%s", common_token_to_piece(ctx_tgt, id_next).c_str());

            // Save record: current token, its features, and the next token (ground truth)
            if (features && n_feat >= feat_size) {
                record_t rec;
                rec.token_id = id_last;
                rec.next_token_id = id_next;
                rec.features.assign(features, features + feat_size);
                all_records.push_back(std::move(rec));
            } else {
                LOG_WRN("  no features at step %d (n_feat=%d, need %d)\n", i, n_feat, feat_size);
            }

            // Check for EOS
            if (llama_vocab_is_eog(vocab, id_next)) {
                LOG_INF("\n  [EOS at step %d]\n", i);
                break;
            }

            id_last = id_next;
            total_tokens++;
        }

        LOG_INF("\n  captured %zu records for this prompt\n", all_records.size() - (total_tokens - params.n_predict));
    }

    LOG_INF("\n=== Capture complete: %zu total records ===\n", all_records.size());

    // Write binary capture file
    {
        // Ensure output directory exists
        std::string dir = output_path.substr(0, output_path.find_last_of('/'));
        if (!dir.empty()) {
            std::string cmd = "mkdir -p '" + dir + "'";
            if (system(cmd.c_str()) != 0) {
                LOG_WRN("Warning: Command failed: %s\n", cmd.c_str());
            }
        }

        std::ofstream out(output_path, std::ios::binary);
        if (!out.is_open()) {
            LOG_ERR("Failed to open output file: %s\n", output_path.c_str());
            return 1;
        }

        // Write header
        capture_header hdr = {};
        memcpy(hdr.magic, SPEC_MAGIC, 4);
        hdr.version = SPEC_VERSION;
        hdr.n_embd = n_embd;
        hdr.n_layers = n_aux;
        hdr.layer_ids[0] = 1;   // TODO: read from model metadata
        hdr.layer_ids[1] = 18;
        hdr.layer_ids[2] = 35;
        hdr.n_records = all_records.size();
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // Write records
        for (const auto & rec : all_records) {
            out.write(reinterpret_cast<const char*>(&rec.token_id), sizeof(int32_t));
            out.write(reinterpret_cast<const char*>(&rec.next_token_id), sizeof(int32_t));
            out.write(reinterpret_cast<const char*>(rec.features.data()), feat_size * sizeof(float));
        }

        out.close();

        size_t file_size = sizeof(capture_header) + all_records.size() * (8 + feat_size * 4);
        LOG_INF("Wrote %s (%.2f MB, %zu records)\n",
                output_path.c_str(), file_size / (1024.0 * 1024.0), all_records.size());
    }

    // Cleanup
    common_sampler_free(smpl);
    llama_backend_free();

    return 0;
}
