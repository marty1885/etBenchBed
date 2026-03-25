#include <etsoc/common/utils.h>
#include <stdint.h>
#include "include/et_blas.h"
#include "include/platform.h"
#include "include/tensor.h"

/*
 * F32 Matrix Multiply for ET-SoC-1 - TensorFMA32.
 *
 * C[M,N] = A[M,K] * B[N,K]^T   (strided batched SGEMM)
 *
 * K-parallel + interleaved tiles + ring reduce.
 * No batched-K yet (needs investigation on hang).
 * This is the last known working version.
 */

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32
#define TILE_K             16
#define TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#define TILE_N             16
#define CACHEOP_MAX        0
#define REP_RATE           0
/* ─────────────────────────────────────────────────────────────────── */

int entry_point(struct et_sgemm_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;
    if (hart_id & 1) return 0;

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_minion_id = get_minion_id();

    const int64_t K = params->K;
    const int64_t M = params->M;
    const int64_t N = params->N;

    const int64_t lda = params->lda;
    const int64_t ldb = params->ldb;
    const int64_t ldc = params->ldc;

    const char* A_base = (const char*)params->A;
    const char* B_base = (const char*)params->B;
    char*       C_base = (char*)params->C;

    setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;

    const int64_t m_tiles = M / TILE_M;
    const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
    const int64_t batch_count = params->batch_count;
    const int64_t base_tiles = m_tiles * n_tiles * batch_count;

    const int64_t stride_A = params->stride_A;
    const int64_t stride_B = params->stride_B;
    const int64_t stride_C = params->stride_C;

    const int64_t total_harts = NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
    const int64_t k_steps = K / TILE_K;
    int64_t k_splits = 1;
    if (base_tiles < total_harts) {
        k_splits = (total_harts + base_tiles - 1) / base_tiles;
        int64_t ks = 1;
        while (ks * 2 <= k_splits && ks * 2 <= 32 && k_steps % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    const int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    const int64_t k_split = local_minion % k_splits;
    const int64_t local_tile_idx = local_minion / k_splits;
    const int64_t tiles_stride = (int64_t)NUM_COMPUTE_SHIRES * tiles_per_shire;

    const int64_t k_steps_per_split = k_steps / k_splits;
    const int64_t k_start = k_split * k_steps_per_split * TILE_K;
    const int64_t k_end   = k_start + k_steps_per_split * TILE_K;

    const uint64_t group_base_global = my_minion_id - k_split;

    for (int64_t tile = (int64_t)shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
         tile < base_tiles;
         tile += tiles_stride) {

        const int64_t tiles_per_batch = m_tiles * n_tiles;
        const int64_t batch_idx     = tile / tiles_per_batch;
        const int64_t tile_in_batch = tile % tiles_per_batch;
        const int64_t nb_idx = tile_in_batch / m_tiles;
        const int64_t mb_idx = tile_in_batch % m_tiles;

        const char* A_batch = A_base + batch_idx * stride_A;
        const char* B_batch = B_base + batch_idx * stride_B;
        char*       C_batch = C_base + batch_idx * stride_C;

        const int64_t mb = mb_idx * TILE_M;
        const int64_t nb = nb_idx * TILE_N;
        const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        for (int64_t kb = k_start; kb < k_end; kb += TILE_K) {

            tensor_load(
                false, false, 0, 0, 0,
                (uint64_t)(B_batch + nb * ldb + kb * sizeof(float)),
                0, n_cur - 1, (uint64_t)ldb, 0
            );

            tensor_load(
                false, false, TILE_K, 7, 0,
                (uint64_t)(A_batch + mb * lda + kb * sizeof(float)),
                0, TILE_K - 1, (uint64_t)lda, 1
            );

            tensor_wait(TENSOR_LOAD_WAIT_0);
            tensor_wait(TENSOR_LOAD_WAIT_1);

            tensor_fma(
                false, 3, n_cur - 1, TILE_K - 1, 0,
                false, false, false, false,
                TILE_K, 0, 0,
                (kb == k_start)
            );

            tensor_wait(TENSOR_FMA_WAIT);
        }

        if (k_splits > 1) {
            const uint64_t num_regs = (uint64_t)n_cur * 2;

            if (k_split > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD,
                                   num_regs,
                                   group_base_global + k_split - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            if (k_split < k_splits - 1) {
                tensor_reduce_send(0, num_regs,
                                   group_base_global + k_split + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        if (k_split == k_splits - 1) {
            tensor_store(
                0, 0, 3, n_cur - 1,
                (uint64_t)(C_batch + nb * ldc + mb * sizeof(float)),
                0, (uint64_t)ldc
            );
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    return 0;
}
