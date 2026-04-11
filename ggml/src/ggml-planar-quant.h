#pragma once

// RotorQuant PlanarQuant — 3-bit and 4-bit KV cache compression via Givens rotation.
// Ported into llama.cpp-1-bit-turbo from johndpope/llama.cpp (turbo branch).

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// PlanarQuant 3-bit (block_planar3_0)
// ---------------------------------------------------------------------------
void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x, block_planar3_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_planar3_0    (const float * GGML_RESTRICT x, void            * GGML_RESTRICT y, int64_t k);
void dequantize_row_planar3_0  (const void  * GGML_RESTRICT x, float           * GGML_RESTRICT y, int64_t k);
size_t quantize_planar3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// ---------------------------------------------------------------------------
// PlanarQuant 4-bit (block_planar4_0)
// ---------------------------------------------------------------------------
void quantize_row_planar4_0_ref(const float * GGML_RESTRICT x, block_planar4_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_planar4_0    (const float * GGML_RESTRICT x, void            * GGML_RESTRICT y, int64_t k);
void dequantize_row_planar4_0  (const void  * GGML_RESTRICT x, float           * GGML_RESTRICT y, int64_t k);
size_t quantize_planar4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

#ifdef __cplusplus
}
#endif
