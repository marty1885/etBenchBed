#include <etsoc/common/utils.h>
#include <stdint.h>
#include "include/platform.h"

#ifndef REP_RATE
#define REP_RATE 0
#endif

#ifndef CACHEOP_MAX
#define CACHEOP_MAX 0
#endif

struct hang_params {
    uint64_t rep_rate;
    uint64_t cacheop_max;
};

int entry_point(struct hang_params* params, void* env) {
    uint64_t hart_id = get_hart_id();

    if ((hart_id % 2) == 0) {
        uint64_t rate = params ? params->rep_rate : REP_RATE;
        uint64_t max = params ? params->cacheop_max : CACHEOP_MAX;

        setup_cache_scp();
        ucache_control(1, rate, max);
        CLEAR_TENSOR_ERROR;
        ucache_control(0, rate, max);
    }

    FENCE;
    return 0;
}
