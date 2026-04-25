/*
 * Empty kernel for measuring kernel launch overhead and op-op gap.
 *
 * Layout (one 64-byte cacheline per slot, no false sharing):
 *   slot[idx][ 0..7]  = t_start (hpmcounter3)
 *   slot[idx][ 8..15] = t_end
 *   slot[idx][16..63] = unused
 *
 * Only hart 0 writes; one slot per kernel invocation. Padding to a full
 * cacheline avoids any same-line cross-core write hazard.
 * Visibility to host comes from the kernel-boundary memory horizon, not
 * from manual L3 flush.
 */

#include <etsoc/common/utils.h>
#include <stdint.h>
#include "include/platform.h"

struct overhead_params {
    volatile uint8_t* out;    /* device pointer, length B*64 */
    uint64_t idx;
};

int entry_point(struct overhead_params* params, void* env) {
    uint64_t t0 = et_get_timestamp();
    uint64_t hart_id = get_hart_id();

    if (hart_id == 0) {
        volatile uint64_t* w = (volatile uint64_t*)(params->out + params->idx * 64);
        uint64_t t1 = et_get_timestamp();
        w[0] = t0;
        w[1] = t1;
    }

    FENCE;
    return 0;
}
