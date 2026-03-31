#pragma once

#include <cstdint>

namespace hostbench {

constexpr int kQk8_0 = 32;

#pragma pack(push, 1)
struct block_q8_0 {
    uint16_t d;
    int8_t qs[kQk8_0];
};
#pragma pack(pop)

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

uint16_t fp32_to_fp16(float f);
float fp16_to_fp32(uint16_t h);
void quantize_row_q8_0(const float* src, void* dst_v, int K);
void dequantize_row_q8_0(const void* src_v, float* dst, int K);

}  // namespace hostbench
