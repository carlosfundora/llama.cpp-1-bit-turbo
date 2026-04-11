#pragma once
#ifdef GGML_TRITON

#include <string>
#include <unordered_map>

#ifdef GGML_USE_HIP
#  include <hip/hip_runtime.h>
#  define triton_module_t         hipModule_t
#  define triton_function_t       hipFunction_t
#  define triton_module_load(m,p) hipModuleLoad(m, p)
#  define triton_module_get_function(f,m,n) hipModuleGetFunction(f, m, n)
#  define triton_launch_kernel    hipModuleLaunchKernel
#else
#  include <cuda.h>
#  define triton_module_t         CUmodule
#  define triton_function_t       CUfunction
#  define triton_module_load(m,p) cuModuleLoad(m, p)
#  define triton_module_get_function(f,m,n) cuModuleGetFunction(f, m, n)
#  define triton_launch_kernel    cuLaunchKernel
#endif

namespace ggml_triton {

// Initialize: scan GGML_TRITON_KERNEL_DIR for .hsaco/.cubin files and load them.
// kernel_dir == nullptr → use GGML_TRITON_KERNEL_DIR env var, then compile-time
// GGML_TRITON_KERNEL_DIR macro, then "./triton-kernels/" relative to executable.
void init(const char * kernel_dir = nullptr);

// Return a cached function handle by kernel name (e.g. "_fused_planar4_quant_pack_kernel_ng64_nl16").
// Returns nullptr if the kernel was not loaded.
triton_function_t get_kernel(const char * kernel_name);

// Thin launch wrapper over triton_launch_kernel.
void launch(triton_function_t fn,
            dim3 grid, dim3 block,
            void ** args,
            size_t shared   = 0,
            cudaStream_t stream = 0);

}  // namespace ggml_triton

#endif // GGML_TRITON
