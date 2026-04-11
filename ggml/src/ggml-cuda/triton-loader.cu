// triton-loader.cu — runtime loader for AOT-compiled Triton kernels (.hsaco/.cubin)
//
// Kernels are compiled at build time by cmake/triton_aot_compile.py when
// GGML_TRITON=ON. At runtime this loader scans the kernel directory, loads each
// module via the driver API, and caches function handles by name.

#ifdef GGML_TRITON

#include "triton-loader.cuh"
#include "common.cuh"

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef GGML_USE_HIP
#  include <hip/hip_runtime.h>
#  define CHECK_DRIVER(call) do { \
       hipError_t _e = (call); \
       if (_e != hipSuccess) { \
           GGML_LOG_ERROR("HIP driver error %s at %s:%d\n", \
               hipGetErrorString(_e), __FILE__, __LINE__); \
       } \
   } while(0)
#else
#  include <cuda.h>
#  define CHECK_DRIVER(call) do { \
       CUresult _e = (call); \
       if (_e != CUDA_SUCCESS) { \
           const char * _s = nullptr; \
           cuGetErrorString(_e, &_s); \
           GGML_LOG_ERROR("CUDA driver error %s at %s:%d\n", \
               _s ? _s : "unknown", __FILE__, __LINE__); \
       } \
   } while(0)
#endif

namespace ggml_triton {

namespace {

struct KernelRegistry {
    std::vector<triton_module_t>                   modules;
    std::unordered_map<std::string, triton_function_t> functions;
    std::mutex                                     mtx;
    bool                                           initialized = false;
};

static KernelRegistry & registry() {
    static KernelRegistry r;
    return r;
}

// Determine the kernel directory to use.
static std::string resolve_kernel_dir(const char * hint) {
    // 1. Explicit caller-supplied hint
    if (hint && hint[0] != '\0') {
        return hint;
    }
    // 2. Environment variable
    const char * env = std::getenv("GGML_TRITON_KERNEL_DIR");
    if (env && env[0] != '\0') {
        return env;
    }
    // 3. Compile-time macro set by CMake
#ifdef GGML_TRITON_KERNEL_DIR
    {
        std::string macro_dir = GGML_TRITON_KERNEL_DIR;
        if (!macro_dir.empty()) {
            return macro_dir;
        }
    }
#endif
    // 4. Fallback: ./triton-kernels/ relative to cwd
    return "./triton-kernels";
}

// Load all .hsaco (HIP) or .cubin (CUDA) files from dir into the registry.
static void load_directory(const std::string & dir) {
    namespace fs = std::filesystem;

#ifdef GGML_USE_HIP
    const std::string ext = ".hsaco";
#else
    const std::string ext = ".cubin";
#endif

    std::error_code ec;
    if (!fs::exists(dir, ec) || ec) {
        GGML_LOG_WARN("ggml_triton::init: kernel directory not found: %s\n", dir.c_str());
        return;
    }

    int loaded = 0;
    for (const auto & entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string path = entry.path().string();
        if (path.size() < ext.size() ||
            path.compare(path.size() - ext.size(), ext.size(), ext) != 0) {
            continue;
        }

        triton_module_t mod;
        auto result = triton_module_load(&mod, path.c_str());
#ifdef GGML_USE_HIP
        if (result != hipSuccess) {
            GGML_LOG_WARN("ggml_triton: failed to load %s: %s\n",
                path.c_str(), hipGetErrorString(result));
            continue;
        }
#else
        if (result != CUDA_SUCCESS) {
            const char * s = nullptr;
            cuGetErrorString(result, &s);
            GGML_LOG_WARN("ggml_triton: failed to load %s: %s\n",
                path.c_str(), s ? s : "unknown");
            continue;
        }
#endif
        registry().modules.push_back(mod);

        // Extract kernel name from filename: strip directory and extension.
        // e.g. "_fused_planar4_quant_pack_kernel_ng64_nl16.hsaco"
        //   → "_fused_planar4_quant_pack_kernel_ng64_nl16"
        const std::string stem = entry.path().stem().string();

        // Register the primary kernel function (same name as the stem).
        triton_function_t fn;
        auto fn_result = triton_module_get_function(&fn, mod, stem.c_str());
#ifdef GGML_USE_HIP
        if (fn_result == hipSuccess) {
#else
        if (fn_result == CUDA_SUCCESS) {
#endif
            registry().functions[stem] = fn;
            GGML_LOG_INFO("ggml_triton: loaded kernel '%s'\n", stem.c_str());
            ++loaded;
        } else {
            // Triton 3.x names the function after the Python function name,
            // not the filename. Try to strip the _<suffix> part if present.
            // e.g. "_fused_planar4_quant_pack_kernel_ng64_nl16" → "_fused_planar4_quant_pack_kernel"
            const auto last_ng = stem.rfind("_ng");
            if (last_ng != std::string::npos) {
                const std::string base_name = stem.substr(0, last_ng);
                triton_function_t fn2;
                auto fn2_result = triton_module_get_function(&fn2, mod, base_name.c_str());
#ifdef GGML_USE_HIP
                if (fn2_result == hipSuccess) {
#else
                if (fn2_result == CUDA_SUCCESS) {
#endif
                    // Register under both the full stem and the bare kernel name.
                    registry().functions[stem]      = fn2;
                    registry().functions[base_name] = fn2;
                    GGML_LOG_INFO("ggml_triton: loaded kernel '%s' (as '%s')\n",
                        stem.c_str(), base_name.c_str());
                    ++loaded;
                } else {
                    GGML_LOG_WARN("ggml_triton: could not resolve function in %s\n",
                        path.c_str());
                }
            } else {
                GGML_LOG_WARN("ggml_triton: could not resolve function in %s\n",
                    path.c_str());
            }
        }
    }

    GGML_LOG_INFO("ggml_triton: loaded %d kernel(s) from %s\n", loaded, dir.c_str());
}

}  // anonymous namespace

void init(const char * kernel_dir) {
    std::lock_guard<std::mutex> lock(registry().mtx);
    if (registry().initialized) return;
    registry().initialized = true;

    const std::string dir = resolve_kernel_dir(kernel_dir);
    GGML_LOG_INFO("ggml_triton::init kernel_dir=%s\n", dir.c_str());
    load_directory(dir);
}

triton_function_t get_kernel(const char * kernel_name) {
    auto & reg = registry();
    // Fast path — no lock after init
    const auto it = reg.functions.find(kernel_name);
    if (it != reg.functions.end()) return it->second;
    return nullptr;
}

void launch(triton_function_t fn,
            dim3 grid, dim3 block,
            void ** args,
            size_t shared,
            cudaStream_t stream) {
    CHECK_DRIVER(
        triton_launch_kernel(
            fn,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            static_cast<unsigned int>(shared),
            stream,
            args,
            nullptr
        )
    );
}

}  // namespace ggml_triton

#endif // GGML_TRITON
