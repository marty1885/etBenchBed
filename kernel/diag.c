#include <etsoc/common/utils.h>
#include <stdint.h>
#include "include/platform.h"

struct diag_params {
    volatile uint64_t *buf;
    int64_t count;
};

int entry_point(struct diag_params* params, void* env) {
    uint64_t hart_id = get_hart_id();

    /* Hart 0 writes a simple pattern to verify kernel execution */
    if (hart_id == 0) {
        volatile uint64_t *buf = params->buf;
        int64_t count = params->count;
        for (int64_t i = 0; i < count; i++) {
            buf[i] = (uint64_t)(i + 1);
        }
    }

    FENCE;
    return 0;
}
