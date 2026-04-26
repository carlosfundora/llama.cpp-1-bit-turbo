import sys

file_path = "ggml/src/ggml-cuda/ggml-cuda.cu"
with open(file_path, "r") as f:
    content = f.read()

new_code = """static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        bool disable_graph = false;
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_AMPERE) {
            disable_graph = true;
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            disable_graph = true;
        }

        if (disable_graph) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}"""

old_code = """static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}"""

if new_code in content:
    content = content.replace(new_code, old_code)
    with open(file_path, "w") as f:
        f.write(content)
    print("Success")
else:
    print("Failed to find block")
