//******************************************************************************
// ET-SoC-1 BLAS Kernel Parameter Definitions
//******************************************************************************

#ifndef ET_BLAS_H
#define ET_BLAS_H

#include <stdint.h>
#include <stddef.h>

// C[M,N] = A[M,K] * B[N,K]^T   (K is the contiguous dimension in both A and B)
//
// Strided batched: batch_count independent GEMMs.
// Set stride_A=0 to broadcast a single A across all batches.
struct et_sgemm_params {
    int64_t M;
    int64_t N;
    int64_t K;

    const void *A;           // [K × M] per batch, K contiguous
    int64_t     lda;         // row stride of A in bytes (stride along M)

    const void *B;           // [K × N] per batch, K contiguous
    int64_t     ldb;         // row stride of B in bytes (stride along N)

    void       *C;           // [M × N] per batch
    int64_t     ldc;         // row stride of C in bytes (stride along N)

    int64_t batch_count;     // number of independent GEMMs (≥1)
    int64_t stride_A;        // bytes between consecutive A matrices (0 = broadcast)
    int64_t stride_B;        // bytes between consecutive B matrices
    int64_t stride_C;        // bytes between consecutive C matrices
};

// DRAM bandwidth test: read src0_bytes + src1_bytes from two buffers.
struct et_bw_params {
    const void *src0;
    int64_t     src0_bytes;
    const void *src1;
    int64_t     src1_bytes;
};

#endif // ET_BLAS_H
