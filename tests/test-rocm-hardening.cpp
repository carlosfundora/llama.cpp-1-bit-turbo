#include "ggml.h"
#include "gguf.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-model-saver.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <vector>

static std::string create_tensor_type_fixture(void) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1u << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * tensor_ctx = ggml_init(params);
    assert(tensor_ctx != nullptr);

    struct gguf_context * gguf_ctx = gguf_init_empty();
    assert(gguf_ctx != nullptr);

    llama_model_saver model_saver(LLM_ARCH_LLAMA, gguf_ctx);
    model_saver.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_LLAMA));

    auto add_tensor = [&](const char * name, ggml_type type) {
        struct ggml_tensor * tensor = ggml_new_tensor_1d(tensor_ctx, type, 32);
        assert(tensor != nullptr);
        ggml_set_name(tensor, name);
        std::memset(tensor->data, 0, ggml_nbytes(tensor));
        gguf_add_tensor(gguf_ctx, tensor);
    };

    add_tensor("blk.0.attn_norm.weight", GGML_TYPE_F32);
    add_tensor("blk.0.ffn_norm.weight", GGML_TYPE_F16);
    add_tensor("output_norm.weight", GGML_TYPE_BF16);

    char fixture_path[] = "/tmp/llama-rocm-hardening-XXXXXX";
    const int fd = mkstemp(fixture_path);
    assert(fd >= 0);
    close(fd);

    model_saver.save(fixture_path);

    gguf_free(gguf_ctx);
    ggml_free(tensor_ctx);

    return fixture_path;
}

int main(void) {
    const std::string fixture_path = create_tensor_type_fixture();
    std::vector<std::string> splits;

    llama_model_loader loader(
        /*metadata                     =*/ nullptr,
        /*set_tensor_data              =*/ nullptr,
        /*set_tensor_data_ud           =*/ nullptr,
        /*fname                        =*/ fixture_path,
        /*splits                       =*/ splits,
        /*file                         =*/ nullptr,
        /*use_mmap                     =*/ false,
        /*use_direct_io                =*/ false,
        /*check_tensors                =*/ false,
        /*no_alloc                     =*/ true,
        /*param_overrides_p            =*/ nullptr,
        /*param_tensor_buft_overrides_p=*/ nullptr);

    assert(loader.require_tensor_meta("blk.0.attn_norm.weight")->type == GGML_TYPE_F32);
    assert(loader.require_tensor_meta("blk.0.ffn_norm.weight")->type == GGML_TYPE_F16);
    assert(loader.require_tensor_meta("output_norm.weight")->type == GGML_TYPE_BF16);

    std::remove(fixture_path.c_str());
    return 0;
}
