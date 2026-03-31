//******************************************************************************
// ET Quantization Block Structures
// Provides Q8_0 block definition for quantized matmul kernels
//******************************************************************************

#ifndef QUANTS_H
#define QUANTS_H

#include <stdint.h>

// Q8_0 quantization: 32 int8 values per block + 1 fp16 scale
#define QK8_0 32

// Q8_0 quantization block (matches GGML definition)
// Each block contains 32 quantized int8 values + 1 fp16 scale factor
// Total size: 2 bytes (scale) + 32 bytes (values) = 34 bytes
typedef struct {
    uint16_t d;           // Scale factor (delta) as fp16 - 2 bytes
    int8_t qs[QK8_0];     // Quantized values (32 x int8) - 32 bytes
} block_q8_0;

#endif // QUANTS_H
