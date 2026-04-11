// ggml-planar-quant.c — RotorQuant PlanarQuant CPU implementation
//
// PlanarQuant uses Givens-rotation-style preprocessing to whiten the distribution
// before scalar quantization.  The C implementation stores quantized data using
// the same bit layout as defined in ggml-common.h so that CUDA set-rows and CPU
// dequant are byte-for-byte compatible.
//
// 3-bit layout (block_planar3_0, QK_PLANAR3=128):
//   norm:     ggml_half  — max(|x|) over the block
//   qs[32]:   2-bit magnitude index per element (4 elements per byte)
//             mag_idx ∈ {0,1,2,3} → centroid ≈ {1/8, 3/8, 5/8, 7/8} × norm
//   signs[16]: 1 sign bit per element (8 elements per byte), 1=negative
//
// 4-bit layout (block_planar4_0, QK_PLANAR4=128):
//   norm:     ggml_half  — max(|x|) over the block
//   rnorm:    ggml_half  — precomputed 1/norm (0 when norm==0)
//   qs[64]:   4-bit index per element (2 elements per byte, low nibble first)
//             q ∈ [0,15] → dequant = (q − 7.5) × norm / 7.5

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-planar-quant.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Shared centroid tables
// ---------------------------------------------------------------------------

// Lloyd-Max-optimal centroids for 4-level uniform distribution on [0,1]:
// boundaries at 0.25, 0.50, 0.75 → centroids 0.125, 0.375, 0.625, 0.875
static const float PLANAR3_MAG_CENTROIDS[4] = { 0.125f, 0.375f, 0.625f, 0.875f };

// Encode helper: find nearest magnitude centroid index for |x|/norm ∈ [0,1]
static inline int planar3_mag_index(float abs_norm) {
    // abs_norm is already |x|/norm, in [0,1].  Four uniform intervals.
    int idx = (int)(abs_norm * 4.0f);
    if (idx < 0) idx = 0;
    if (idx > 3) idx = 3;
    return idx;
}

// ---------------------------------------------------------------------------
// PlanarQuant 3-bit — reference (scalar) implementation
// ---------------------------------------------------------------------------
void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x,
                                block_planar3_0 * GGML_RESTRICT y,
                                int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    const int nb = (int)(k / QK_PLANAR3);

    for (int i = 0; i < nb; i++) {
        // Pass 1: find max absolute value (norm)
        float norm = 0.0f;
        for (int j = 0; j < QK_PLANAR3; j++) {
            float av = fabsf(x[i * QK_PLANAR3 + j]);
            if (av > norm) norm = av;
        }

        y[i].norm = GGML_FP32_TO_FP16(norm);

        // Clear output bytes
        memset(y[i].qs,    0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) continue;

        const float inv_norm = 1.0f / norm;

        // Pass 2: quantize each element
        for (int j = 0; j < QK_PLANAR3; j++) {
            const float val  = x[i * QK_PLANAR3 + j];
            const int   sign = (val < 0.0f) ? 1 : 0;
            const float anv  = fabsf(val) * inv_norm; // normalised to [0,1]
            const int   midx = planar3_mag_index(anv);

            // Pack magnitude index: 2 bits at position (j % 4) * 2 in byte j/4
            y[i].qs[j / 4] |= (uint8_t)((midx & 0x3) << ((j % 4) * 2));

            // Pack sign bit: 1 bit at position j % 8 in byte j/8
            if (sign) {
                y[i].signs[j / 8] |= (uint8_t)(1 << (j % 8));
            }
        }
    }
}

// Fast path (same as ref for now; SIMD can be wired here later)
void quantize_row_planar3_0(const float * GGML_RESTRICT x,
                             void * GGML_RESTRICT y,
                             int64_t k) {
    quantize_row_planar3_0_ref(x, (block_planar3_0 *)y, k);
}

// Dequantize
void dequantize_row_planar3_0(const void * GGML_RESTRICT vx,
                               float * GGML_RESTRICT y,
                               int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    const int nb = (int)(k / QK_PLANAR3);
    const block_planar3_0 * x = (const block_planar3_0 *)vx;

    for (int i = 0; i < nb; i++) {
        const float norm = GGML_FP16_TO_FP32(x[i].norm);

        for (int j = 0; j < QK_PLANAR3; j++) {
            // Unpack magnitude index
            const int midx = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            // Unpack sign bit
            const int sign = (x[i].signs[j / 8] >> (j % 8)) & 0x1;

            const float mag = PLANAR3_MAG_CENTROIDS[midx] * norm;
            y[i * QK_PLANAR3 + j] = sign ? -mag : mag;
        }
    }
}

// Batch quantize wrapper (imatrix ignored for now — simple max-scale)
size_t quantize_planar3_0(const float * src, void * dst,
                           int64_t nrows, int64_t n_per_row,
                           const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_PLANAR3 == 0);
    const size_t row_size = (n_per_row / QK_PLANAR3) * sizeof(block_planar3_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_planar3_0_ref(src + r * n_per_row,
                                   (block_planar3_0 *)((char *)dst + r * row_size),
                                   n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}

// ---------------------------------------------------------------------------
// PlanarQuant 4-bit — reference (scalar) implementation
// ---------------------------------------------------------------------------
//
// 16 uniform levels over [-norm, +norm].
// q = clamp(round(x/norm * 7.5 + 7.5), 0, 15)
// x_hat = (q − 7.5) * norm / 7.5
//
void quantize_row_planar4_0_ref(const float * GGML_RESTRICT x,
                                block_planar4_0 * GGML_RESTRICT y,
                                int64_t k) {
    assert(k % QK_PLANAR4 == 0);
    const int nb = (int)(k / QK_PLANAR4);

    for (int i = 0; i < nb; i++) {
        // Pass 1: find max absolute value
        float norm = 0.0f;
        for (int j = 0; j < QK_PLANAR4; j++) {
            float av = fabsf(x[i * QK_PLANAR4 + j]);
            if (av > norm) norm = av;
        }

        y[i].norm  = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(norm > 0.0f ? 1.0f / norm : 0.0f);

        memset(y[i].qs, 0, sizeof(y[i].qs));

        if (norm == 0.0f) continue;

        const float inv_norm = 1.0f / norm;

        // Pass 2: quantize each element to a 4-bit index
        for (int j = 0; j < QK_PLANAR4; j++) {
            // Map [-1,1] → [0,15]: q = round(x/norm * 7.5 + 7.5)
            float fq = x[i * QK_PLANAR4 + j] * inv_norm * 7.5f + 7.5f;
            int   q  = (int)(fq + 0.5f);
            if (q < 0)  q = 0;
            if (q > 15) q = 15;

            // Pack nibble: low nibble for even j, high nibble for odd j
            if (j % 2 == 0) {
                y[i].qs[j / 2]  = (uint8_t)(q & 0xF);
            } else {
                y[i].qs[j / 2] |= (uint8_t)((q & 0xF) << 4);
            }
        }
    }
}

void quantize_row_planar4_0(const float * GGML_RESTRICT x,
                             void * GGML_RESTRICT y,
                             int64_t k) {
    quantize_row_planar4_0_ref(x, (block_planar4_0 *)y, k);
}

void dequantize_row_planar4_0(const void * GGML_RESTRICT vx,
                               float * GGML_RESTRICT y,
                               int64_t k) {
    assert(k % QK_PLANAR4 == 0);
    const int nb = (int)(k / QK_PLANAR4);
    const block_planar4_0 * x = (const block_planar4_0 *)vx;

    for (int i = 0; i < nb; i++) {
        const float norm = GGML_FP16_TO_FP32(x[i].norm);
        const float scale = norm / 7.5f;  // per-unit step

        for (int j = 0; j < QK_PLANAR4; j++) {
            // Unpack nibble
            int q;
            if (j % 2 == 0) {
                q = x[i].qs[j / 2] & 0xF;
            } else {
                q = (x[i].qs[j / 2] >> 4) & 0xF;
            }
            y[i * QK_PLANAR4 + j] = ((float)q - 7.5f) * scale;
        }
    }
}

size_t quantize_planar4_0(const float * src, void * dst,
                           int64_t nrows, int64_t n_per_row,
                           const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_PLANAR4 == 0);
    const size_t row_size = (n_per_row / QK_PLANAR4) * sizeof(block_planar4_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_planar4_0_ref(src + r * n_per_row,
                                   (block_planar4_0 *)((char *)dst + r * row_size),
                                   n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}
