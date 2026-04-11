#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;
    const float neg_d = -d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d
    const uint8_t bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const uint8_t bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = bit_0 ? d : neg_d;
    v.y = bit_1 ? d : neg_d;
}

static __device__ __forceinline__ void dequantize_q1_0_g128(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0_g128 * x = (const block_q1_0_g128 *) vx;

    const float d = x[ib].d;
    const float neg_d = -d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d
    const uint8_t bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const uint8_t bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = bit_0 ? d : neg_d;
    v.y = bit_1 ? d : neg_d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// RotorQuant dequantize functions
// planar3_0: norm(f16) + qs[32](2-bit mag) + signs[16](1-bit sign), 128 elements
static __device__ __forceinline__ void dequantize_planar3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar3_0 * x = (const block_planar3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    // iqs is element index (0, 2, 4, ...) — returns pair at iqs and iqs+1
    const int j0 = iqs;
    const int j1 = iqs + 1;

    // Centroids: 0.125, 0.375, 0.625, 0.875 (Lloyd-Max 4-level uniform [0,1])
    const float centroids[4] = { 0.125f, 0.375f, 0.625f, 0.875f };

    const int midx0 = (x[ib].qs[j0 / 4] >> ((j0 % 4) * 2)) & 0x3;
    const int sign0 = (x[ib].signs[j0 / 8] >> (j0 % 8)) & 0x1;
    v.x = sign0 ? -(centroids[midx0] * norm) : (centroids[midx0] * norm);

    const int midx1 = (x[ib].qs[j1 / 4] >> ((j1 % 4) * 2)) & 0x3;
    const int sign1 = (x[ib].signs[j1 / 8] >> (j1 % 8)) & 0x1;
    v.y = sign1 ? -(centroids[midx1] * norm) : (centroids[midx1] * norm);
}

// planar4_0: norm(f16) + rnorm(f16) + qs[64](4-bit nibble-packed), 128 elements
static __device__ __forceinline__ void dequantize_planar4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar4_0 * x = (const block_planar4_0 *) vx;
    const float norm  = __half2float(x[ib].norm);
    const float scale = norm / 7.5f;
    const int j0 = iqs;
    const int j1 = iqs + 1;

    const int q0 = (j0 % 2 == 0) ? (x[ib].qs[j0 / 2] & 0xF) : ((x[ib].qs[j0 / 2] >> 4) & 0xF);
    const int q1 = (j1 % 2 == 0) ? (x[ib].qs[j1 / 2] & 0xF) : ((x[ib].qs[j1 / 2] >> 4) & 0xF);

    v.x = ((float)q0 - 7.5f) * scale;
    v.y = ((float)q1 - 7.5f) * scale;
}

// iso3_0: identical layout to planar3_0
static __device__ __forceinline__ void dequantize_iso3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso3_0 * x = (const block_iso3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    const int j0 = iqs;
    const int j1 = iqs + 1;

    const float centroids[4] = { 0.125f, 0.375f, 0.625f, 0.875f };

    const int midx0 = (x[ib].qs[j0 / 4] >> ((j0 % 4) * 2)) & 0x3;
    const int sign0 = (x[ib].signs[j0 / 8] >> (j0 % 8)) & 0x1;
    v.x = sign0 ? -(centroids[midx0] * norm) : (centroids[midx0] * norm);

    const int midx1 = (x[ib].qs[j1 / 4] >> ((j1 % 4) * 2)) & 0x3;
    const int sign1 = (x[ib].signs[j1 / 8] >> (j1 % 8)) & 0x1;
    v.y = sign1 ? -(centroids[midx1] * norm) : (centroids[midx1] * norm);
}

// iso4_0: identical layout to planar4_0
static __device__ __forceinline__ void dequantize_iso4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso4_0 * x = (const block_iso4_0 *) vx;
    const float norm  = __half2float(x[ib].norm);
    const float scale = norm / 7.5f;
    const int j0 = iqs;
    const int j1 = iqs + 1;

    const int q0 = (j0 % 2 == 0) ? (x[ib].qs[j0 / 2] & 0xF) : ((x[ib].qs[j0 / 2] >> 4) & 0xF);
    const int q1 = (j1 % 2 == 0) ? (x[ib].qs[j1 / 2] & 0xF) : ((x[ib].qs[j1 / 2] >> 4) & 0xF);

    v.x = ((float)q0 - 7.5f) * scale;
    v.y = ((float)q1 - 7.5f) * scale;
}
