#!/bin/bash

# Fix implicit conversion warnings to int in ggml-blas
sed -i 's/int n = src0->ne\[0\];/int n = (int)src0->ne[0];/g' ggml/src/ggml-blas/ggml-blas.cpp
sed -i 's/int k = src0->ne\[1\];/int k = (int)src0->ne[1];/g' ggml/src/ggml-blas/ggml-blas.cpp
sed -i 's/int m = src1->ne\[0\];/int m = (int)src1->ne[0];/g' ggml/src/ggml-blas/ggml-blas.cpp

# Fix missing __builtin_available for cblas_sgemm on tvOS
sed -i 's/cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);/\n#if defined(__APPLE__)\n    if (__builtin_available(macOS 13.3, iOS 16.4, tvOS 16.4, watchOS 9.4, *)) {\n#endif\n        cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);\n#if defined(__APPLE__)\n    } else {\n        ggml_abort("cblas_sgemm not available on this platform");\n    }\n#endif\n/g' ggml/src/ggml-blas/ggml-blas.cpp

# Fix arm64 build error for undefined symbols in ggml-cpu.c
cat << 'EOC' >> ggml/src/ggml-cpu/arch/arm/quants.c
#if !defined(__ARM_NEON)
void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
}
void ggml_vec_dot_q1_0_g128_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_q1_0_g128_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
}
#endif
EOC
