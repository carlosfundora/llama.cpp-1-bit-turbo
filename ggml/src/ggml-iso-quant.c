// ggml-iso-quant.c — RotorQuant IsoQuant CPU implementation
//
// IsoQuant uses a Hadamard/isometric rotation (instead of Givens) to whiten
// the distribution before scalar quantization.  The block layout is identical
// to PlanarQuant (same struct sizes), and the CPU implementation is likewise
// identical — the rotation is typically pre-applied to model weights offline;
// the C code here handles the inference-path quantize/dequantize of KV tensors.
//
// See ggml-planar-quant.c for the full algorithm description.

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-iso-quant.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>

// Centroid table: same as PlanarQuant 3-bit (uniform 4-level on [0,1])
static const float ISO3_MAG_CENTROIDS[4] = { 0.125f, 0.375f, 0.625f, 0.875f };

static inline int iso3_mag_index(float abs_norm) {
    int idx = (int)(abs_norm * 4.0f);
    if (idx < 0) idx = 0;
    if (idx > 3) idx = 3;
    return idx;
}

// ---------------------------------------------------------------------------
// IsoQuant 3-bit — reference (scalar) implementation
// ---------------------------------------------------------------------------
void quantize_row_iso3_0_ref(const float * GGML_RESTRICT x,
                              block_iso3_0 * GGML_RESTRICT y,
                              int64_t k) {
    assert(k % QK_ISO3 == 0);
    const int nb = (int)(k / QK_ISO3);

    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_ISO3; j++) {
            float av = fabsf(x[i * QK_ISO3 + j]);
            if (av > norm) norm = av;
        }

        y[i].norm = GGML_FP32_TO_FP16(norm);
        memset(y[i].qs,    0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) continue;

        const float inv_norm = 1.0f / norm;

        for (int j = 0; j < QK_ISO3; j++) {
            const float val  = x[i * QK_ISO3 + j];
            const int   sign = (val < 0.0f) ? 1 : 0;
            const float anv  = fabsf(val) * inv_norm;
            const int   midx = iso3_mag_index(anv);

            y[i].qs[j / 4] |= (uint8_t)((midx & 0x3) << ((j % 4) * 2));
            if (sign) {
                y[i].signs[j / 8] |= (uint8_t)(1 << (j % 8));
            }
        }
    }
}

void quantize_row_iso3_0(const float * GGML_RESTRICT x,
                          void * GGML_RESTRICT y,
                          int64_t k) {
    quantize_row_iso3_0_ref(x, (block_iso3_0 *)y, k);
}

void dequantize_row_iso3_0(const void * GGML_RESTRICT vx,
                            float * GGML_RESTRICT y,
                            int64_t k) {
    assert(k % QK_ISO3 == 0);
    const int nb = (int)(k / QK_ISO3);
    const block_iso3_0 * x = (const block_iso3_0 *)vx;

    for (int i = 0; i < nb; i++) {
        const float norm = GGML_FP16_TO_FP32(x[i].norm);

        for (int j = 0; j < QK_ISO3; j++) {
            const int midx = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            const int sign = (x[i].signs[j / 8] >> (j % 8)) & 0x1;
            const float mag = ISO3_MAG_CENTROIDS[midx] * norm;
            y[i * QK_ISO3 + j] = sign ? -mag : mag;
        }
    }
}

size_t quantize_iso3_0(const float * src, void * dst,
                        int64_t nrows, int64_t n_per_row,
                        const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_ISO3 == 0);
    const size_t row_size = (n_per_row / QK_ISO3) * sizeof(block_iso3_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_iso3_0_ref(src + r * n_per_row,
                                (block_iso3_0 *)((char *)dst + r * row_size),
                                n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}

// ---------------------------------------------------------------------------
// IsoQuant 4-bit — reference (scalar) implementation
// ---------------------------------------------------------------------------
void quantize_row_iso4_0_ref(const float * GGML_RESTRICT x,
                              block_iso4_0 * GGML_RESTRICT y,
                              int64_t k) {
    assert(k % QK_ISO4 == 0);
    const int nb = (int)(k / QK_ISO4);

    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_ISO4; j++) {
            float av = fabsf(x[i * QK_ISO4 + j]);
            if (av > norm) norm = av;
        }

        y[i].norm  = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(norm > 0.0f ? 1.0f / norm : 0.0f);
        memset(y[i].qs, 0, sizeof(y[i].qs));

        if (norm == 0.0f) continue;

        const float inv_norm = 1.0f / norm;

        for (int j = 0; j < QK_ISO4; j++) {
            float fq = x[i * QK_ISO4 + j] * inv_norm * 7.5f + 7.5f;
            int   q  = (int)(fq + 0.5f);
            if (q < 0)  q = 0;
            if (q > 15) q = 15;

            if (j % 2 == 0) {
                y[i].qs[j / 2]  = (uint8_t)(q & 0xF);
            } else {
                y[i].qs[j / 2] |= (uint8_t)((q & 0xF) << 4);
            }
        }
    }
}

void quantize_row_iso4_0(const float * GGML_RESTRICT x,
                          void * GGML_RESTRICT y,
                          int64_t k) {
    quantize_row_iso4_0_ref(x, (block_iso4_0 *)y, k);
}

void dequantize_row_iso4_0(const void * GGML_RESTRICT vx,
                            float * GGML_RESTRICT y,
                            int64_t k) {
    assert(k % QK_ISO4 == 0);
    const int nb = (int)(k / QK_ISO4);
    const block_iso4_0 * x = (const block_iso4_0 *)vx;

    for (int i = 0; i < nb; i++) {
        const float norm  = GGML_FP16_TO_FP32(x[i].norm);
        const float scale = norm / 7.5f;

        for (int j = 0; j < QK_ISO4; j++) {
            int q;
            if (j % 2 == 0) {
                q = x[i].qs[j / 2] & 0xF;
            } else {
                q = (x[i].qs[j / 2] >> 4) & 0xF;
            }
            y[i * QK_ISO4 + j] = ((float)q - 7.5f) * scale;
        }
    }
}

size_t quantize_iso4_0(const float * src, void * dst,
                        int64_t nrows, int64_t n_per_row,
                        const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_ISO4 == 0);
    const size_t row_size = (n_per_row / QK_ISO4) * sizeof(block_iso4_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_iso4_0_ref(src + r * n_per_row,
                                (block_iso4_0 *)((char *)dst + r * row_size),
                                n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}
