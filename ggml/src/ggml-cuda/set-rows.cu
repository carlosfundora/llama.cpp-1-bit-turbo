#include "set-rows.cuh"
#include "cpy-utils.cuh"

#ifdef GGML_TRITON
#include "triton-loader.cuh"
#endif

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


#ifdef GGML_TRITON
// ---------------------------------------------------------------------------
// Triton dispatch helpers for fused RotorQuant set_rows kernels
//
// These functions are called by Track-A planar/iso set_rows implementations.
// They attempt to dispatch to the pre-compiled Triton kernel first; callers
// must fall back to the C++ path when they return false.
//
// Kernel file names follow the convention produced by triton_aot_compile.py:
//   _fused_planar4_quant_pack_kernel_ng64_nl16.(hsaco|cubin)
//   _fused_planar4_unpack_dequant_kernel_ng64.(hsaco|cubin)
//   _fused_iso4_quant_pack_kernel_ng32_nl16.(hsaco|cubin)
//   _fused_iso4_unpack_dequant_kernel_ng32.(hsaco|cubin)
//
// Kernel arguments mirror the Triton @triton.jit signatures exactly.
// Compile-time constexprs (n_groups, n_levels, BLOCK_G) are baked into the
// compiled binary; only runtime scalar args are passed via the void** array.
// ---------------------------------------------------------------------------

static bool try_triton_planar4_quant(
    const half   * __restrict__ input,   // [batch, emb_dim] fp16
    uint8_t      * __restrict__ packed,  // [batch, n_groups] uint8 output
    const float  * __restrict__ rot2,    // [n_groups, 2]  cos/sin
    const float  * __restrict__ centroids, // [16]
    int32_t batch_size, int32_t emb_dim,
    int32_t n_groups,                    // must match compiled n_groups=64
    int32_t stride_in_b, int32_t stride_in_d,
    int32_t stride_pack_b, int32_t stride_pack_g,
    cudaStream_t stream)
{
    auto fn = ggml_triton::get_kernel("_fused_planar4_quant_pack_kernel_ng64_nl16");
    if (!fn) return false;

    void * args[] = {
        (void*)&input, (void*)&packed,
        (void*)&rot2, (void*)&centroids,
        (void*)&batch_size, (void*)&emb_dim,
        (void*)&stride_in_b, (void*)&stride_in_d,
        (void*)&stride_pack_b, (void*)&stride_pack_g,
    };

    constexpr int BLOCK_G = 8;
    dim3 grid(batch_size, (n_groups + BLOCK_G - 1) / BLOCK_G);
    dim3 block(1);
    ggml_triton::launch(fn, grid, block, args, 0, stream);
    return true;
}

static bool try_triton_planar4_unpack_dequant(
    const uint8_t * __restrict__ packed,    // [batch, n_groups] uint8
    half          * __restrict__ output,    // [batch, emb_dim]  fp16
    const float   * __restrict__ rot2,      // [n_groups, 2]
    const float   * __restrict__ centroids, // [16]
    const float   * __restrict__ norms,     // [batch]
    int32_t batch_size, int32_t emb_dim,
    int32_t n_groups,
    int32_t stride_pack_b, int32_t stride_pack_g,
    int32_t stride_out_b,  int32_t stride_out_d,
    cudaStream_t stream)
{
    auto fn = ggml_triton::get_kernel("_fused_planar4_unpack_dequant_kernel_ng64");
    if (!fn) return false;

    void * args[] = {
        (void*)&packed, (void*)&output,
        (void*)&rot2, (void*)&centroids, (void*)&norms,
        (void*)&batch_size, (void*)&emb_dim,
        (void*)&stride_pack_b, (void*)&stride_pack_g,
        (void*)&stride_out_b,  (void*)&stride_out_d,
    };

    constexpr int BLOCK_G = 8;
    dim3 grid(batch_size, (n_groups + BLOCK_G - 1) / BLOCK_G);
    dim3 block(1);
    ggml_triton::launch(fn, grid, block, args, 0, stream);
    return true;
}

static bool try_triton_iso4_quant(
    const half   * __restrict__ input,     // [batch, d_padded] fp16
    uint8_t      * __restrict__ packed,    // [batch, n_groups*2] uint8
    const float  * __restrict__ qL,        // [n_groups, 4]
    const float  * __restrict__ qR,        // [n_groups, 4]
    const float  * __restrict__ centroids, // [16]
    int32_t batch_size, int32_t d_padded,
    int32_t n_groups,
    int32_t stride_in_b, int32_t stride_in_d,
    int32_t stride_pack_b, int32_t stride_pack_e,
    cudaStream_t stream)
{
    auto fn = ggml_triton::get_kernel("_fused_iso4_quant_pack_kernel_ng32_nl16");
    if (!fn) return false;

    void * args[] = {
        (void*)&input, (void*)&packed,
        (void*)&qL, (void*)&qR, (void*)&centroids,
        (void*)&batch_size, (void*)&d_padded,
        (void*)&stride_in_b, (void*)&stride_in_d,
        (void*)&stride_pack_b, (void*)&stride_pack_e,
    };

    constexpr int BLOCK_G = 8;
    dim3 grid(batch_size, (n_groups + BLOCK_G - 1) / BLOCK_G);
    dim3 block(1);
    ggml_triton::launch(fn, grid, block, args, 0, stream);
    return true;
}

static bool try_triton_iso4_unpack_dequant(
    const uint8_t * __restrict__ packed,    // [batch, n_groups*2] uint8
    half          * __restrict__ output,    // [batch, head_dim]   fp16
    const float   * __restrict__ qL,        // [n_groups, 4]
    const float   * __restrict__ qR,        // [n_groups, 4]
    const float   * __restrict__ centroids, // [16]
    const float   * __restrict__ norms,     // [batch]
    int32_t batch_size, int32_t d_padded, int32_t head_dim,
    int32_t n_groups,
    int32_t stride_pack_b, int32_t stride_pack_e,
    int32_t stride_out_b,  int32_t stride_out_d,
    cudaStream_t stream)
{
    auto fn = ggml_triton::get_kernel("_fused_iso4_unpack_dequant_kernel_ng32");
    if (!fn) return false;

    void * args[] = {
        (void*)&packed, (void*)&output,
        (void*)&qL, (void*)&qR, (void*)&centroids, (void*)&norms,
        (void*)&batch_size, (void*)&d_padded, (void*)&head_dim,
        (void*)&stride_pack_b, (void*)&stride_pack_e,
        (void*)&stride_out_b,  (void*)&stride_out_d,
    };

    constexpr int BLOCK_G = 8;
    dim3 grid(batch_size, (n_groups + BLOCK_G - 1) / BLOCK_G);
    dim3 block(1);
    ggml_triton::launch(fn, grid, block, args, 0, stream);
    return true;
}

// Suppress unused-function warnings until Track A wires these in.
static void ggml_triton_suppress_unused_warnings() __attribute__((unused));
static void ggml_triton_suppress_unused_warnings() {
    (void)try_triton_planar4_quant;
    (void)try_triton_planar4_unpack_dequant;
    (void)try_triton_iso4_quant;
    (void)try_triton_iso4_unpack_dequant;
}
#endif // GGML_TRITON


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
