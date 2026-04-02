# Q8_0 Matrix Multiply Kernel Optimization Log

Target: ET-SoC-1 (32 shires x 32 minions x 2 harts = 2048 harts)
Kernel: `kernel/mul_mat_q8.c` (benchbed), `mul_mat_Q8_0.c` (llama.cpp)
Operation: C[M,N] = A[M,K] * B[N,K]^T, A is Q8_0, B/C are F32

## Architecture notes

- L1 cache is shared between both harts in a minion (~32KB usable with SCP split)
- L2 SCP: writes go through L1 (need flush/evict), reads bypass L1
- Scalar and vector FP registers are shared (MMX-style). Scalar float ops (flw/fadd.s/fsw) between vector .ps ops corrupt packed SIMD register state. **Never use float arrays (`accum[r]`) between vector calls** -- use scalar variables instead.
- Barrier primitives: ET_BARRIER_MINION (FLB=local_minion_id, FCC 0), ET_BARRIER_SHIRE, ET_BARRIER_GLOBAL
- Semaphore: et_sem_post/et_sem_wait (FCC only, no FLB, lighter than barriers)
- Q8_0 block: 32 int8 values + 1 fp16 scale = 34 bytes

## Baseline

**File:** `mmq8_perf.csv`

Original kernel: each hart independently processes rows m = hart_id, hart_id+2048, ...
Single `compute_row_dot_q8_0(q_row, b_col, K_blocks)` call per row.

Peak N=1 GEMV bandwidth: ~61 GB/s on small-K shapes (4096x4096).
Large-K shapes severely bandwidth-starved: 8192x28672 at 43 GFLOPS (23 GB/s).

## Attempt 1: K-split + tile-outer + adaptive routing

**File:** `mmq8_perf_opt1.csv`
**Status:** Working in both benchbed and llama.cpp (LLaMA 3, Qwen 2.5, Gemma 3)

Three-path kernel:

### Path 1: K-split (intra-minion collaboration)

Both harts in a minion process the same rows, each computing half of K.
Partial sums exchanged via L2SCP with balanced semaphore protocol.

**Conditions:** `K_blocks >= 256 AND (rows_per_minion <= 4 OR (rows_per_minion <= 8 AND k_half <= TILE_KB))`

**Exchange protocol (balanced FCC credits):**
1. Hart 1: store partial to L2SCP -> FENCE -> flush_to_l2 -> WAIT_CACHEOPS -> sem_post -> sem_wait
2. Hart 0: sem_wait -> read L2SCP -> sem_post -> combine -> store

Each (row, N) iteration: both harts post 1, consume 1. Zero residual on kernel exit.
One cache-line-aligned L2SCP slot per minion: `et_shire_l2scp_local(local_minion * 64)`.

**Previous K-split attempt failed in llama.cpp** using et_barrier instead of semaphores.
Root cause: likely unbalanced FCC credits across kernel invocations.

### Path 2: Tile-outer with scalar row groups

For large K shapes not eligible for K-split (e.g., 8192x28672 with 8 rows/minion and k_half > TILE_KB).
Groups of 4 rows share each B tile (TILE_KB=256 blocks = 32KB) before advancing.

**Key:** Uses scalar variables `s0, s1, s2, s3` (not an array!) to accumulate across tiles.
Array `accum[r]` generates flw/fadd.s/fsw stack ops that corrupt vector register state on ET-SoC-1.
Scalar variables stay in fixed FP registers -- compiler avoids conflicts with asm clobber lists.

**Condition:** `K_blocks > TILE_KB` (K > 8192) and not K-split eligible.

### Path 3: Simple (small K)

Single `compute_row_dot_q8_0` call per row. No tiling overhead.

**Condition:** `K_blocks <= TILE_KB` and not K-split eligible.

### Results (N=1 GEMV, baseline -> opt1)

| Shape (MxK) | Baseline GFLOPS | Opt1 GFLOPS | Gain | Path |
|-------------|----------------|-------------|------|------|
| 2048x16384 | 43.5 | 96.4 | +122% | K-split |
| 8192x28672 | 43.1 | 84.5 | +96% | Tile-outer |
| 4096x14336 | 61.6 | 97.4 | +58% | K-split |
| 2560x10240 | 78.6 | 118.6 | +51% | K-split |
| 3840x15360 | 73.3 | 97.4 | +33% | K-split |
| 8192x8192 N=2 | 75.5 | 98.5 | +30% | K-split |
| 3584x18944 | 77.6 | 96.6 | +25% | K-split |
| 2048x11008 | 107.9 | 128.1 | +19% | K-split |
| 1536x8960 | 82.8 | 92.0 | +11% | K-split |
| 4096x4096 | 115.7 | 114.2 | flat | Simple |
| 10240x2560 | 134.3 | 132.9 | flat | Simple |

## What didn't work

1. **TILE_KB=128** -- 23% regression on 8192x8192. Per-tile overhead (mask save/restore, gather pattern load, horizontal reduce) called 2x more often. Fixed by TILE_KB=256.

2. **Tile-inner (row-major with tiled K)** -- no improvement for large shapes. Each row still streams ALL of B; tiling just splits the call into smaller chunks with the same total footprint. Only tile-outer (B reuse across rows) helps.

3. **Software L2 prefetching** -- no improvement. Hardware sequential prefetcher already handles it, or DRAM bandwidth is saturated. Added ~4% instruction overhead for zero gain.

4. **KSPLIT_MAX_ROWS=8 unconditionally** -- 8192x28672 regressed from 84 to 64 GFLOPS. L1 thrashing when both harts x 8 rows stream 57KB of B each. Fixed with adaptive condition: rows>4 only when k_half <= TILE_KB (each hart's half fits in one tile).

5. **Tile-outer with `accum[]` array** -- worked on LLaMA 3 and Qwen 2.5 but corrupted Gemma 3 output. Root cause: `accum[r] += q8_dot_reduce()` generates scalar float stack ops (flw + fadd.s + fsw) between vector .ps calls, corrupting MMX-style shared FP registers. Fixed by using scalar variables instead of array.

6. **Separate `q8_dot_tile()` and `q8_dot_reduce()` functions** -- compiler could insert scalar float ops between f20 write (tile accumulation) and f20 read (horizontal reduce), clobbering f20. Must fuse tile+reduce or keep them in the same inline scope.

7. **DRAM-backed C buffer with evict_to_l2 for K-split exchange** -- verification failed. L2SCP has well-defined semantics; regular DRAM addresses do not guarantee visibility between harts via L1 flush/evict.

## Next steps

1. **Split-phase API in benchbed** -- llama.cpp hoists mask save/restore outside loops via q8_dot_begin/teardown. Benchbed calls compute_row_dot_q8_0 which re-saves/restores per call. ~20 cycles wasted per call x 16 calls per tile-outer group = ~320 cycles.

2. **Batched K-split exchange** -- group 4 rows with tile-outer inside K-split, batch to 1 exchange per group instead of 1 per row. Reduces sync overhead and gets B reuse within each K half. Would help all K-split shapes.

3. **N>1 A-reuse tiling** -- for batch inference (N=2-8), tile N so multiple B columns share the same A row data in L1. Currently each N iteration re-streams A.

4. **Larger tile-outer groups (8 rows)** -- 8 scalar accumulators s0-s7 for more B reuse per tile. More register pressure (8 of 32 FP regs) but still feasible given clobber list uses ~12.

5. **Lower KSPLIT_MIN_K_BLOCKS** -- shapes like 1536x8960 (K_blocks=280, rows/minion=2) are just above threshold. Could benefit from a lower cutoff, but need to verify the K-split overhead doesn't eat the gain for smaller K.
