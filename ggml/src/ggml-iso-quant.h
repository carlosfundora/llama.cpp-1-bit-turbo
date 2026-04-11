#pragma once

// RotorQuant IsoQuant — 3-bit and 4-bit KV cache compression via Hadamard/isometric rotation.
// Ported into llama.cpp-1-bit-turbo from johndpope/llama.cpp (turbo branch).

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// IsoQuant 3-bit (block_iso3_0)
// ---------------------------------------------------------------------------
void quantize_row_iso3_0_ref(const float * GGML_RESTRICT x, block_iso3_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_iso3_0    (const float * GGML_RESTRICT x, void         * GGML_RESTRICT y, int64_t k);
void dequantize_row_iso3_0  (const void  * GGML_RESTRICT x, float        * GGML_RESTRICT y, int64_t k);
size_t quantize_iso3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// ---------------------------------------------------------------------------
// IsoQuant 4-bit (block_iso4_0)
// ---------------------------------------------------------------------------
void quantize_row_iso4_0_ref(const float * GGML_RESTRICT x, block_iso4_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_iso4_0    (const float * GGML_RESTRICT x, void         * GGML_RESTRICT y, int64_t k);
void dequantize_row_iso4_0  (const void  * GGML_RESTRICT x, float        * GGML_RESTRICT y, int64_t k);
size_t quantize_iso4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

#ifdef __cplusplus
}
#endif
