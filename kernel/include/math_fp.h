//******************************************************************************
// ET Floating Point Conversion Library
// Provides FP16 <-> FP32 conversion using ET hardware instructions
//******************************************************************************

#ifndef MATH_FP_H
#define MATH_FP_H

#include <stdint.h>

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction
static inline float fp16_to_fp32(uint16_t h) {
    float result;
    unsigned long temp;
    uint32_t raw = (uint32_t)h;

    __asm__ volatile (
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "fbcx.ps   %[result], %[raw]    \n\t"
        "fcvt.ps.f16 %[result], %[result] \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [raw] "r"(raw)
    );

    return result;
}

#endif // MATH_FP_H
