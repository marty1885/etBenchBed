//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized Q8_0 dot product using ET hardware vector instructions
//******************************************************************************

#ifndef BLOCK_OPS_H
#define BLOCK_OPS_H

#include <stdint.h>
#include "math_fp.h"
#include "quants.h"

typedef struct {
    unsigned long saved_mask;
} q8_dot_state;

static inline void q8_dot_begin(q8_dot_state* state) {
    __asm__ volatile("mova.x.m %0" : "=r"(state->saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
}

static inline void q8_dot_end(const q8_dot_state* state) {
    __asm__ volatile("mova.m.x %0" :: "r"(state->saved_mask));
}

// Compute full-row dot product: sum over K_blocks Q8_0 blocks against F32 vector.
// Hoists mask save/restore and gather pattern load outside the block loop.
// Accumulates scaled partial products in a vector register and does a single
// horizontal reduce at the end.
//
// Uses fg32b.ps for aligned 8-byte gathers (fast path) and falls back to
// fgb.ps for chunks that straddle a 32-byte alignment boundary.
static inline float q8_dot_compute(const block_q8_0* q_row,
                                   const float* b_col,
                                   int64_t K_blocks) {
    const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint64_t gather_0_to_7 = 0x398a418820ULL;
    __asm__ volatile(
        "flw.ps f31, %[g]\n"
        :
        : [g] "m"(*(const int32_t(*)[8])gather_pattern)
        : "f31"
    );
    __asm__ volatile("fbci.pi f20, 0" ::: "f20");  // vector accumulator

    for (int64_t kb = 0; kb < K_blocks; kb++) {
        const block_q8_0* blk = q_row + kb;
        const float* b_ptr = b_col + (kb << 5);
        const uintptr_t qs_addr = (uintptr_t)blk->qs;
        const uintptr_t qs_aligned = qs_addr & ~(uintptr_t)31;
        const uintptr_t qs_low = qs_addr & 31;
        const int fast_chunks = (int)((32 - qs_low) >> 3);

        if (fast_chunks >= 3) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv1]\n"
                "fg32b.ps    f11, %[gi](%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv2]\n"
                "fg32b.ps    f11, %[gi](%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi]  "r"(gather_0_to_7),
                  [ap0] "r"(qs_addr),
                  [ap1] "r"(qs_aligned | ((qs_addr + 8)  & 31)),
                  [ap2] "r"(qs_aligned | ((qs_addr + 16) & 31)),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        } else if (fast_chunks == 2) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv1]\n"
                "fg32b.ps    f11, %[gi](%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi]  "r"(gather_0_to_7),
                  [ap0] "r"(qs_addr),
                  [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        } else if (fast_chunks == 1) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv1]\n"
                "fgb.ps      f11, f31(%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi]  "r"(gather_0_to_7),
                  [ap0] "r"(qs_addr),
                  [ap1] "r"(&blk->qs[8]),
                  [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        } else {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fgb.ps      f11, f31(%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv1]\n"
                "fgb.ps      f11, f31(%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"

                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [ap0] "r"(&blk->qs[0]),
                  [ap1] "r"(&blk->qs[8]),
                  [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        }

        // f20 += f10 * broadcast(scale) -- hardware fp16->fp32 via FCVT.PS.F16
        uint32_t scale_raw = (uint32_t)blk->d;
        __asm__ volatile(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20"
        );
    }

    // Single horizontal reduce
    float result;
    __asm__ __volatile__ (
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (result)
        :: "t0", "f1", "f2", "f3", "f4", "f5"
    );

    return result;
}

// Compute two row dots together while reusing the same loaded B chunks.
//
// Safe when every row starts at the same 32-byte offset, i.e. the Q8 row stride
// (`lda`) is a multiple of 32. In that case the gather/alignment pattern is the
// same for both rows at a given `kb`, so one set of B vector loads can feed both
// row accumulators.
static inline void q8_dot_compute_x2_aligned(const block_q8_0* q_row0,
                                             const block_q8_0* q_row1,
                                             const float* b_col,
                                             int64_t K_blocks,
                                             float* out0,
                                             float* out1) {
    const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint64_t gather_0_to_7 = 0x398a418820ULL;
    __asm__ volatile(
        "flw.ps f31, %[g]\n"
        :
        : [g] "m"(*(const int32_t(*)[8])gather_pattern)
        : "f31"
    );
    __asm__ volatile(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n"
        ::: "f20", "f21"
    );

    for (int64_t kb = 0; kb < K_blocks; kb++) {
        const block_q8_0* blk0 = q_row0 + kb;
        const block_q8_0* blk1 = q_row1 + kb;
        const float* b_ptr = b_col + (kb << 5);

        const uintptr_t qs_addr0 = (uintptr_t)blk0->qs;
        const uintptr_t qs_addr1 = (uintptr_t)blk1->qs;
        const uintptr_t qs_aligned0 = qs_addr0 & ~(uintptr_t)31;
        const uintptr_t qs_aligned1 = qs_addr1 & ~(uintptr_t)31;
        const int fast_chunks = (int)((32 - (qs_addr0 & 31)) >> 3);

        if (fast_chunks >= 3) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fg32b.ps    f16, %[gi](%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fg32b.ps    f16, %[gi](%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi]    "r"(gather_0_to_7),
                  [r0ap0] "r"(qs_addr0),
                  [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8)  & 31)),
                  [r0ap2] "r"(qs_aligned0 | ((qs_addr0 + 16) & 31)),
                  [r0ap3] "r"(&blk0->qs[24]),
                  [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8)  & 31)),
                  [r1ap2] "r"(qs_aligned1 | ((qs_addr1 + 16) & 31)),
                  [r1ap3] "r"(&blk1->qs[24]),
                  [bv0]   "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1]   "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2]   "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3]   "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17"
            );
        } else if (fast_chunks == 2) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fg32b.ps    f16, %[gi](%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi]    "r"(gather_0_to_7),
                  [r0ap0] "r"(qs_addr0),
                  [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8) & 31)),
                  [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]),
                  [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8) & 31)),
                  [r1ap2] "r"(&blk1->qs[16]),
                  [r1ap3] "r"(&blk1->qs[24]),
                  [bv0]   "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1]   "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2]   "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3]   "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17"
            );
        } else if (fast_chunks == 1) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fgb.ps      f16, f31(%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fgb.ps      f17, f31(%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi]    "r"(gather_0_to_7),
                  [r0ap0] "r"(qs_addr0),
                  [r0ap1] "r"(&blk0->qs[8]),
                  [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]),
                  [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(&blk1->qs[8]),
                  [r1ap2] "r"(&blk1->qs[16]),
                  [r1ap3] "r"(&blk1->qs[24]),
                  [bv0]   "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1]   "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2]   "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3]   "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17"
            );
        } else {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fgb.ps      f16, f31(%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fgb.ps      f17, f31(%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fgb.ps      f16, f31(%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fgb.ps      f17, f31(%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [r0ap0] "r"(&blk0->qs[0]),
                  [r0ap1] "r"(&blk0->qs[8]),
                  [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]),
                  [r1ap0] "r"(&blk1->qs[0]),
                  [r1ap1] "r"(&blk1->qs[8]),
                  [r1ap2] "r"(&blk1->qs[16]),
                  [r1ap3] "r"(&blk1->qs[24]),
                  [bv0]   "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1]   "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2]   "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3]   "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17"
            );
        }

        const uint32_t scale_raw0 = (uint32_t)blk0->d;
        const uint32_t scale_raw1 = (uint32_t)blk1->d;
        __asm__ volatile(
            "fbcx.ps     f24, %[s0]\n"
            "fcvt.ps.f16 f24, f24\n"
            "fmadd.ps    f20, f10, f24, f20\n"
            "fbcx.ps     f25, %[s1]\n"
            "fcvt.ps.f16 f25, f25\n"
            "fmadd.ps    f21, f11, f25, f21\n"
            :
            : [s0] "r"(scale_raw0),
              [s1] "r"(scale_raw1)
            : "f20", "f21", "f24", "f25"
        );
    }

    float result0;
    float result1;
    __asm__ __volatile__ (
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (result0)
        :: "t0", "f1", "f2", "f3", "f4", "f5"
    );
    __asm__ __volatile__ (
        "fswizz.ps f1, f21, 0xB1 \n\t"
        "fadd.ps   f2, f21, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (result1)
        :: "t0", "f1", "f2", "f3", "f4", "f5"
    );

    *out0 = result0;
    *out1 = result1;
}

static inline float compute_row_dot_q8_0(const block_q8_0* q_row,
                                         const float* b_col,
                                         int64_t K_blocks) {
    q8_dot_state state;
    q8_dot_begin(&state);
    const float result = q8_dot_compute(q_row, b_col, K_blocks);
    q8_dot_end(&state);
    return result;
}

#endif // BLOCK_OPS_H
