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

#define STRIDE_M        2048  /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT 1024  /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS 256   /* K >= 8192 elements */
#define KSPLIT_MAX_ROWS     8     /* max rows per minion for K-split */
#define TILE_KB           256     /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */

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

    /* K-split decision: both harts in a minion collaborate on same rows */
    const int64_t minion_id = hart_id >> 1;          /* 0..1023 global */
    const int64_t local_minion = (hart_id >> 1) & 0x1F;  /* 0..31 within shire */
    const int is_hart1 = hart_id & 1;
    const int64_t rows_per_minion = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    const int64_t k_half = K_blocks / 2;
    /*
     * K-split when K is large enough to benefit, and either:
     *   - few rows (≤4): always safe, proven working
     *   - more rows (5-8): only if each hart's half fits in one tile,
     *     otherwise L1 thrashing from 2 harts × 8 rows kills performance
     */
    const int use_ksplit = (K_blocks >= KSPLIT_MIN_K_BLOCKS)
                        && (rows_per_minion <= KSPLIT_MAX_ROWS)
                        && (rows_per_minion <= 4 || k_half <= TILE_KB);

    if (use_ksplit) {
        /* Each hart processes half the K dimension */
        const int64_t k_start = is_hart1 ? k_half : 0;
        const int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        volatile float* l2scp_slot =
            (volatile float*)et_shire_l2scp_local(local_minion * 64);

        for (int64_t n = 0; n < N; n++) {
            const float* b_col = (const float*)(B + n * ldb);

            for (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                const block_q8_0* q_row = (const block_q8_0*)(A + m * lda);
                float partial = compute_row_dot_q8_0(
                    q_row + k_start, b_col + k_start * 32, k_len);

                if (is_hart1) {
                    /* Write partial sum to L2SCP, make visible */
                    *l2scp_slot = partial;
                    FENCE;
                    flush_to_l2((const void*)l2scp_slot, 1, 64);
                    WAIT_CACHEOPS;
                    et_sem_post(ET_BARRIER_MINION);   /* signal: data ready */
                    et_sem_wait(ET_BARRIER_MINION);   /* wait: hart 0 done reading */
                } else {
                    et_sem_wait(ET_BARRIER_MINION);   /* wait: data ready */
                    float other = *l2scp_slot;        /* L2SCP read bypasses L1 */
                    et_sem_post(ET_BARRIER_MINION);   /* ack: done reading */

                    float* dst = (float*)(C + n * ldc + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst, partial + other);
                }
            }
        }
    } else if (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        for (int64_t n = 0; n < N; n++) {
            const float* b_col = (const float*)(B + n * ldb);

            for (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                const int64_t m1 = m0 + STRIDE_M;
                const int64_t m2 = m0 + STRIDE_M * 2;
                const int64_t m3 = m0 + STRIDE_M * 3;

                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                for (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                    int64_t tile_len = K_blocks - kb;
                    if (tile_len > TILE_KB) tile_len = TILE_KB;
                    const float* b_tile = b_col + kb * 32;

                    /* All rows in the group read the same B tile (L1 reuse) */
                                   s0 += compute_row_dot_q8_0(
                                       (const block_q8_0*)(A + m0 * lda) + kb,
                                       b_tile, tile_len);
                    if (m1 < M) s1 += compute_row_dot_q8_0(
                                       (const block_q8_0*)(A + m1 * lda) + kb,
                                       b_tile, tile_len);
                    if (m2 < M) s2 += compute_row_dot_q8_0(
                                       (const block_q8_0*)(A + m2 * lda) + kb,
                                       b_tile, tile_len);
                    if (m3 < M) s3 += compute_row_dot_q8_0(
                                       (const block_q8_0*)(A + m3 * lda) + kb,
                                       b_tile, tile_len);
                }

                float* c_base = (float*)(C + n * ldc);
                               atomic_store_f32((volatile float*)(c_base + m0), s0);
                if (m1 < M) atomic_store_f32((volatile float*)(c_base + m1), s1);
                if (m2 < M) atomic_store_f32((volatile float*)(c_base + m2), s2);
                if (m3 < M) atomic_store_f32((volatile float*)(c_base + m3), s3);
            }
        }
    } else {
        /* Simple path for small K (single tile, no B reuse benefit) */
        for (int64_t n = 0; n < N; n++) {
            const float* b_col = (const float*)(B + n * ldb);

            for (int64_t m = hart_id; m < M; m += STRIDE_M) {
                const block_q8_0* q_row = (const block_q8_0*)(A + m * lda);
                float sum = compute_row_dot_q8_0(q_row, b_col, K_blocks);

                float* dst = (float*)(C + n * ldc + m * sizeof(float));
                atomic_store_f32((volatile float*)dst, sum);
            }
        }
    }

    return 0;
}
