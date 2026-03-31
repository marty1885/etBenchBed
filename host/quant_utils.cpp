#include "quant_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hostbench {

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = (bits >> 16) & 0x8000;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    const uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) {
        return sign;
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits = 0;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void quantize_row_q8_0(const float* src, void* dst_v, int K) {
    auto* dst = static_cast<block_q8_0*>(dst_v);
    for (int kb = 0; kb < K / kQk8_0; ++kb) {
        const float* blk = src + kb * kQk8_0;
        float amax = 0.0f;
        for (int i = 0; i < kQk8_0; ++i) {
            amax = std::max(amax, std::fabs(blk[i]));
        }
        const float scale = amax / 127.0f;
        dst[kb].d = fp32_to_fp16(scale);
        const float inv = (scale != 0.0f) ? 127.0f / amax : 0.0f;
        for (int i = 0; i < kQk8_0; ++i) {
            const float q = std::max(-128.0f, std::min(127.0f, std::round(blk[i] * inv)));
            dst[kb].qs[i] = static_cast<int8_t>(q);
        }
    }
}

void dequantize_row_q8_0(const void* src_v, float* dst, int K) {
    const auto* src = static_cast<const block_q8_0*>(src_v);
    for (int kb = 0; kb < K / kQk8_0; ++kb) {
        const float scale = fp16_to_fp32(src[kb].d);
        for (int i = 0; i < kQk8_0; ++i) {
            dst[kb * kQk8_0 + i] = scale * static_cast<float>(src[kb].qs[i]);
        }
    }
}

}  // namespace hostbench
