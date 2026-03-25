# etBenchBed: ET-SoC-1 Benchmark Suite

Standalone benchmarks for DRAM bandwidth and F32 matrix multiply on the ET-SoC-1

## Project Structure

```
etBenchBed/
├── analysis/
│   ├── perf_surface.C      ROOT macro for MM GFLOPS surfaces with one fixed dimension
│   └── plot_dram_bw_surface.C
│                            ROOT macro for BW sweep surface + ridge fitting
├── host/
│   └── main.cpp            Host program: CLI parsing, PCIe init, kernel launch, timing, verify, CSV output
├── kernel/
│   ├── dram_bw.c           DRAM read bandwidth benchmark (L2SCP loads, no compute)
│   ├── mul_mat_f32.c       F32 matmul using TensorFMA32 (K-parallel + hardware reduce)
│   ├── include/
│   │   ├── et_blas.h       Shared parameter struct (host ↔ kernel ABI)
│   │   ├── platform.h      SCP setup, ucache_control, fence macros
│   │   └── tensor.h        tensor_load/fma/store/reduce APIs
│   ├── crt.S               Bare-metal C runtime startup
│   └── sections.ld         Linker script (entry @ 0x8005801000)
└── CMakeLists.txt          Top-level build (cross-compiles kernels + builds host)
```

## Kernels

### `dram_bw.c` - DRAM Read Bandwidth

Pure read benchmark. All 1024 even harts stream through src0 and src1 using `et_tensor_load_l2scp`, bypassing L1/L2 caches. This benchmark is designed to measure the achivable read bandwidth for a given pattern,

**Tuning knobs** (compile-time `#define`s):
| Knob | Default | What it does |
|------|---------|-------------|
| `STAGGER_LINES` | 15 | Cache lines per L2SCP load. 15 is coprime to 16 DRAM channels and 4 L2 banks - breaks channel lockstep. Set 0 for baseline (16 lines). |
| `CACHEOP_MAX` | 4 | Max outstanding L2 requests per minion. Prevents L2 bank flooding. 0=unlimited. |
| `REP_RATE` | 0 | Injection delay between L2SCP ops (0=none, 1-7=2^N cycles). No measured benefit. |

**Reference results** (default (measured optmimal valies) vs 16-line/unlimited baseline):
- Peak (25 MB): 127 GB/s, was 83 -> +53%
- Floor (269 MB): 66 GB/s, was 54 -> +21%

### `mul_mat_f32.c` - F32 Matrix Multiply

Full matmul TensorFMA32 (16×16 hardware engine).

Implementation:
- K-parallel: When fewer tiles than harts (small N), splits K across adjacent minions within each shire. Ring-reduces partial sums via TensorSend/TensorRecv.
- Interleaved tile mapping: Tiles distributed across shires in interleaved order (not contiguous blocks) for DRAM channel balance.
- Tuning knobs: `TILE_N` (16=full, 15=staggered), `CACHEOP_MAX`, `REP_RATE`

Measured results:
- Common prefill shap: N=512, 1078 GFLOPS
- Common Decode shape: N=1 M=K=4096, 10 GFLOPS / 20 GB/s

## Building

Requires et-platform installed to `/opt/et`.

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/opt/et -Wno-dev
cmake --build build
```

This configures the host build normally and cross-compiles the RISC-V kernels as in-tree CMake targets under `build/kernel`.

## Running

**Prerequisite**: load the ET PCIe driver before running.

```bash
sudo modprobe et_dma
```

### Single point measurement

```bash
# DRAM bandwidth
./build/host/bench_host --bench bw -m 4096 -k 4096 -n 16

# Matrix multiply
./build/host/bench_host --bench mm -m 4096 -k 4096 -n 128

# Matrix multiply with host-side BLAS verification
./build/host/bench_host --bench mm -m 256 -k 256 -n 256 --verify
```

### Sweep mode

```bash
# Full BW surface: M=K in {16..8192}, N in {16..8192} (~8700 points, ~2-3 hours)
./build/host/bench_host --bench bw --sweep --csv bw_sweep.csv

# MM sweep: 21 LLM-relevant sizes (decode/prefill/various models)
./build/host/bench_host --bench mm --sweep --csv mm_sweep.csv
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--bench bw\|mm` | required | Benchmark type |
| `-m M` | 4096 | Rows (single run) |
| `-k K` | 4096 | Reduction dimension (single run) |
| `-n N` | 16 | Columns (single run) |
| `--sweep` | off | Enable sweep over predefined sizes |
| `--verify` | off | For `mm` only: run one launch per point and compare device output against OpenBLAS `cblas_sgemm` |
| `--runs N` | 10 | Minimum runs per measurement |
| `--warmup N` | 2 | Warmup runs (discarded) |
| `--csv FILE` | none | Write results to CSV |
| `--kernel PATH` | auto | Override kernel ELF path |
| `--seed SEED` | 42 | sets random seed |

### Timing methodology

Each test point:
1. Warmup - `--warmup` runs discarded
2. Batched dispatch - launches are enqueued back-to-back and synchronized once per batch to amortize host/PCIe overhead
3. Adaptive sampling - additional batches are launched until both `--runs` minimum reached AND 500ms total elapsed
4. Reported metric - average time per run = total measured batch time / total runs

Output shows the run count and batch count used for the measurement: `(4096 runs, 2 batches)`

### Verify mode

`--verify` is supported only for `--bench mm`.

For each test point the host:
1. Launches the device kernel once
2. Copies `dst` back to host memory
3. Computes a reference result with OpenBLAS `cblas_sgemm`
4. Reports max absolute error and max relative error
5. Fails the run if `max_rel >= sqrt(K) * 1e-5`

### CSV format

**BW**: `M,K,N,BYTES_READ,US_PER_RUN,BW_GB_S,RUNS`

**MM**: `M,K,N,US_PER_RUN,MFLOPS,GFLOPS,BW_GB_S,RUNS`

## Plotting

The repo includes two [ROOT](https://root.cern/) macros under `analysis/` for visualizing sweep output.

### MM performance surface

`[analysis/perf_surface.C` reads an MM CSV with a `GFLOPS` column and plots a 3D surface while holding one dimension fixed.

```bash
root -l 'analysis/perf_surface.C("N", 16, "mm_sweep.csv")'
root -l 'analysis/perf_surface.C("M", 4096, "mm_sweep.csv")'
root -l 'analysis/perf_surface.C("K", 4096, "mm_sweep.csv")'
```

### DRAM bandwidth surface

`analysis/plot_dram_bw_surface.C` reads a BW sweep CSV, plots the 3D bandwidth surface for square cases (`M == K`), highlights the per-`N` ridge, and fits simple ridge models.

```bash
root -l 'analysis/plot_dram_bw_surface.C("bw_sweep.csv")'
root -l 'analysis/plot_dram_bw_surface.C("bw_sweep.csv", "dram_bw_surface.png", true)'
```

## Notes

- `dram_bw.c` uses L2 scratchpad loads (`et_tensor_load_l2scp`). `mul_mat_f32.c` uses L1 scratchpad loads (`tensor_load`) + TensorFMA32. In theory you can get better bandwidth by L2 load then L1SCP. But that is yet to be implemented.
- The `CacheOpMax` tuning knob in `ucache_control` only affects L2SCP loads and cache ops - it does not affect L1 tensor_load path used by the matmul kernel.
- TensorSend/Recv `TARGET`/`SOURCE` fields use **global minion IDs** (`hart_id >> 1`), not shire-local IDs.
