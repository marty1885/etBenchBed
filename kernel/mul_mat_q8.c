//******************************************************************************
// Q8_0 Matrix Multiply Kernel for ET-SoC-1
//
// C[M,N] = A[M,K] * B[N,K]^T   where A is Q8_0 quantized, B and C are F32.
//
// All 2048 harts participate via scalar SIMD (no tensor engine).
// Each hart processes a stripe of M rows: m = hart_id, hart_id+2048, ...
//
// Uses et_sgemm_params with:
//   A -> Q8_0 data, lda = (K/32) * sizeof(block_q8_0) = (K/32) * 34
//   B -> F32 data,  ldb = K * sizeof(float)
//   C -> F32 output, ldc = M * sizeof(float)
//******************************************************************************

#include <etsoc/common/utils.h>
#include <stdint.h>
#include "include/et_blas.h"
#include "include/platform.h"
#include "include/math_fp.h"
#include "include/quants.h"
#include "include/block_ops.h"

#define STRIDE_M 2048   /* 32 shires x 32 minions x 2 harts */

int entry_point(struct et_sgemm_params* params, void* env) {
    uint64_t hart_id = get_hart_id();

    const int64_t M = params->M;
    const int64_t N = params->N;
    const int64_t K = params->K;
    const int64_t K_blocks = K / QK8_0;

    const char* A = (const char*)params->A;   /* Q8_0 weight matrix */
    const char* B = (const char*)params->B;   /* F32 activation matrix */
    char*       C = (char*)params->C;          /* F32 output */

    const int64_t lda = params->lda;   /* bytes per row of A (Q8_0) */
    const int64_t ldb = params->ldb;   /* bytes per row of B (F32) */
    const int64_t ldc = params->ldc;   /* bytes per row of C (F32) */

    for (int64_t n = 0; n < N; n++) {
        const float* b_col = (const float*)(B + n * ldb);

        for (int64_t m = hart_id; m < M; m += STRIDE_M) {
            const block_q8_0* q_row = (const block_q8_0*)(A + m * lda);
            float sum = compute_row_dot_q8_0(q_row, b_col, K_blocks);

            float* dst = (float*)(C + n * ldc + m * sizeof(float));
            atomic_store_f32((volatile float*)dst, sum);
        }
    }

    return 0;
}
