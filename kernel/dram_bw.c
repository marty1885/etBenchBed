#include <etsoc/common/utils.h>
#include <stdint.h>
#include "include/et_blas.h"
#include "include/platform.h"
#include "include/tensor.h"

/*
 * DRAM Read Bandwidth Benchmark for ET-SoC-1.
 *
 * Pure read benchmark using et_tensor_load_l2scp.
 * Optimal config: 15-line loads (960B, coprime to 16 channels and 4 L2 banks)
 *                 CacheOpMax=4 (prevents L2 bank flooding)
 *
 * Bandwidth = (src0_bytes + src1_bytes) / time
 */

#define NUM_ACTIVE_HARTS   1024
#define NUM_COMPUTE_SHIRES 32

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#define STAGGER_LINES      15    /* 0 = 16-line baseline, 15 = best     */
#define CACHEOP_MAX        4     /* 0 = unlimited, 4 = best             */
#define REP_RATE           0     /* 0 = no delay                        */
/* ─────────────────────────────────────────────────────────────────── */

#if STAGGER_LINES == 0
  #define LINES_PER_LOAD  16
#else
  #define LINES_PER_LOAD  STAGGER_LINES
#endif
#define CHUNK_BYTES  (LINES_PER_LOAD * 64)

int entry_point(struct et_bw_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;
    if (hart_id & 1) return 0;

    uint64_t global_id = (shire_id << 5) + ((hart_id >> 1) & 0x1F);

    const char* src0_base = (const char*)params->src0;
    const char* src1_base = (const char*)params->src1;

    const int64_t src0_bytes = params->src0_bytes;
    const int64_t src1_bytes = params->src1_bytes;
    const int64_t total_chunks = (src0_bytes + src1_bytes + CHUNK_BYTES - 1)
                                  / CHUNK_BYTES;

    setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;

    et_tensor_load_l2scp_conf_t conf;
    conf.use_tmask = false;
    conf.num_lines = LINES_PER_LOAD - 1;
    conf.stride    = 64;

    for (int64_t i = global_id; i < total_chunks; i += NUM_ACTIVE_HARTS) {
        const int64_t byte_off = i * (int64_t)CHUNK_BYTES;

        if (byte_off < src0_bytes) {
            conf.addr = (uint64_t)(src0_base + byte_off);
        } else {
            conf.addr = (uint64_t)(src1_base + (byte_off - src0_bytes));
        }

        const int channel = (i / NUM_ACTIVE_HARTS) & 1;
        conf.id = channel;
        conf.dst_start = channel * CHUNK_BYTES;
        tensor_wait(2 + channel);
        et_tensor_load_l2scp(&conf);
    }

    tensor_wait(2);
    tensor_wait(3);

    FENCE;
    return 0;
}
