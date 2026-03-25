/*
 * ET-SoC-1 Benchmark Host Program
 *
 * Launches DRAM BW or F32 matmul benchmark kernels on real hardware via PCIe.
 * Supports single-point runs and full sweeps with CSV output.
 *
 * Usage:
 *   bench_host --bench bw  -m 4096 -k 4096 -n 16
 *   bench_host --bench mm  -m 4096 -k 4096 -n 512
 *   bench_host --bench bw  --sweep --csv results.csv
 *   bench_host --bench mm  --sweep --csv results.csv
 *   bench_host --bench mm  -m 256 -k 256 -n 256 --verify
 *   bench_host --bench mm  -m 256 -k 256 -n 256 --seed 123
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cblas.h>

#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>

#include "Constants.h"

namespace fs = std::filesystem;

struct et_sgemm_params {
    int64_t M, N, K;
    const void* A;   int64_t lda;
    const void* B;   int64_t ldb;
    void*       C;   int64_t ldc;
    int64_t batch_count;
    int64_t stride_A, stride_B, stride_C;
};

struct et_bw_params {
    const void* src0;  int64_t src0_bytes;
    const void* src1;  int64_t src1_bytes;
};

struct TestPoint {
    int m;
    int k;
    int n;
};

struct BenchResult {
    double us_per_run;
    double gflops;
    double bw_gbs;
    double bytes_read;
    int num_runs;
    int num_batches;
};

static std::vector<std::byte> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "Cannot open: %s\n", path.c_str());
        std::exit(1);
    }

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

static std::vector<TestPoint> bw_sweep_points() {
    std::vector<TestPoint> points;

    std::vector<int> mk_vals = {16, 32};
    for (int value = 64; value <= 8192; value += 64) {
        mk_vals.push_back(value);
    }

    std::vector<int> n_vals = {16, 32, 64};
    for (int value = 128; value <= 8192; value += 128) {
        n_vals.push_back(value);
    }

    for (int mk : mk_vals) {
        for (int n : n_vals) {
            points.push_back({mk, mk, n});
        }
    }

    return points;
}

static std::vector<TestPoint> mm_sweep_points() {
    return {
        {4096, 4096, 1}, {4096, 4096, 2}, {4096, 4096, 4},
        {4096, 11008, 1}, {4096, 11008, 4},
        {4096, 4096, 16}, {4096, 4096, 32}, {4096, 4096, 64},
        {4096, 11008, 16},
        {4096, 4096, 128}, {4096, 4096, 256}, {4096, 4096, 512},
        {2048, 2048, 1}, {2048, 2048, 16}, {2048, 5504, 1},
        {8192, 8192, 1}, {8192, 8192, 16},
        {2496, 2496, 16}, {3072, 3072, 16},
        {4096, 14336, 1}, {4096, 14336, 16},
    };
}

int main(int argc, char** argv) {
    enum BenchType { BW, MM };

    constexpr double MIN_TOTAL_US = 500000.0;
    constexpr int MAX_RUNS = 50000;
    constexpr int INITIAL_BATCH_RUNS = 32;
    constexpr int MAX_BATCH_RUNS = 4096;
    constexpr uint64_t SHIRE_MASK = 0xFFFFFFFF;

    BenchType bench_type = BW;
    bool sweep = false;
    bool verify = false;
    int M = 4096;
    int K = 4096;
    int N = 16;
    int min_runs = 10;
    int warmup_runs = 2;
    std::string csv_path;
    unsigned seed = 42;
    bool custom_seed = false;
    std::string kernel_override;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench") && i + 1 < argc) {
            ++i;
            if (!std::strcmp(argv[i], "bw")) {
                bench_type = BW;
            } else if (!std::strcmp(argv[i], "mm")) {
                bench_type = MM;
            } else {
                fprintf(stderr, "Unknown bench: %s\n", argv[i]);
                return 1;
            }
        } else if (!std::strcmp(argv[i], "--sweep")) {
            sweep = true;
        } else if (!std::strcmp(argv[i], "--verify")) {
            verify = true;
        } else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-k") && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) {
            min_runs = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup_runs = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--kernel") && i + 1 < argc) {
            kernel_override = argv[++i];
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 0));
            custom_seed = true;
        } else {
            fprintf(stderr,
                    "Usage: bench_host --bench bw|mm [-m M -k K -n N] [--sweep] "
                    "[--verify] [--runs N] [--warmup N] [--csv FILE] [--kernel ELF] "
                    "[--seed SEED]\n");
            return 1;
        }
    }

    if (verify && bench_type != MM) {
        fprintf(stderr, "--verify is only supported with --bench mm\n");
        return 1;
    }

    printf("Initializing PCIe device...\n");
    std::shared_ptr<dev::IDeviceLayer> device_layer(
        dev::IDeviceLayer::createPcieDeviceLayer().release());
    rt::RuntimePtr runtime = rt::IRuntime::create(device_layer);

    const std::vector<rt::DeviceId> devices = runtime->getDevices();
    if (devices.empty()) {
        fprintf(stderr, "No ET devices found\n");
        return 1;
    }

    const rt::DeviceId device = devices[0];
    const rt::StreamId stream = runtime->createStream(device);
    printf("Device ready.\n");

    const char* kernel_name = (bench_type == BW) ? "dram_bw" : "mul_mat_f32";
    const std::string kernel_path = kernel_override.empty()
        ? (fs::path(KERNELS_DIR) / (std::string(kernel_name) + ".elf")).string()
        : kernel_override;

    printf("Loading kernel: %s\n", kernel_path.c_str());
    const std::vector<std::byte> elf_data = readFile(kernel_path);
    const rt::LoadCodeResult load_result =
        runtime->loadCode(stream, elf_data.data(), elf_data.size());
    runtime->waitForEvent(load_result.event_);
    const rt::KernelId kernel = load_result.kernel_;
    printf("Kernel loaded: %s\n", kernel_name);

    std::vector<TestPoint> points;
    if (sweep) {
        points = (bench_type == BW) ? bw_sweep_points() : mm_sweep_points();
    } else {
        points.push_back({M, K, N});
    }

    int64_t max_mk = 0;
    int64_t max_nk = 0;
    int64_t max_mn = 0;
    for (const TestPoint& p : points) {
        max_mk = std::max(max_mk, static_cast<int64_t>(p.m) * p.k);
        max_nk = std::max(max_nk, static_cast<int64_t>(p.n) * p.k);
        max_mn = std::max(max_mn, static_cast<int64_t>(p.m) * p.n);
    }

    const size_t src0_bytes = static_cast<size_t>(max_mk * sizeof(float));
    const size_t src1_bytes = static_cast<size_t>(max_nk * sizeof(float));
    const size_t dst_bytes = static_cast<size_t>(max_mn * sizeof(float));

    if (custom_seed) {
        printf("Using custom seed: %u\n", seed);
    }

    printf("Allocating: src0=%.1fMB src1=%.1fMB dst=%.1fMB\n",
           src0_bytes / 1e6,
           src1_bytes / 1e6,
           dst_bytes / 1e6);

    std::byte* d_src0 = runtime->mallocDevice(device, src0_bytes);
    std::byte* d_src1 = runtime->mallocDevice(device, src1_bytes);
    std::byte* d_dst = runtime->mallocDevice(device, dst_bytes);

    std::vector<float> h_A;
    std::vector<float> h_B;
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> a_data(max_mk);
        for (float& value : a_data) {
            value = dist(rng);
        }
        runtime->memcpyHostToDevice(stream,
                                    reinterpret_cast<const std::byte*>(a_data.data()),
                                    d_src0,
                                    src0_bytes);

        std::vector<float> b_data(max_nk);
        for (float& value : b_data) {
            value = dist(rng);
        }
        runtime->memcpyHostToDevice(stream,
                                    reinterpret_cast<const std::byte*>(b_data.data()),
                                    d_src1,
                                    src1_bytes);

        runtime->waitForStream(stream);

        if (verify) {
            h_A = std::move(a_data);
            h_B = std::move(b_data);
        }
    }

    rt::KernelLaunchOptions opts;
    opts.setShireMask(SHIRE_MASK);
    opts.setBarrier(true);
    opts.setFlushL3(false);

    FILE* csv = nullptr;
    if (!csv_path.empty()) {
        csv = fopen(csv_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "Cannot open CSV: %s\n", csv_path.c_str());
            return 1;
        }

        if (bench_type == BW) {
            fprintf(csv, "M,K,N,BYTES_READ,US_PER_RUN,BW_GB_S,RUNS\n");
        } else {
            fprintf(csv, "M,K,N,US_PER_RUN,MFLOPS,GFLOPS,BW_GB_S,RUNS\n");
        }
    }

    const int total = static_cast<int>(points.size());
    int verify_pass = 0;
    int verify_fail = 0;
    const auto t_start = std::chrono::steady_clock::now();

    printf("\nRunning %d test points (%s%s):\n\n",
           total,
           (bench_type == BW) ? "BW" : "MM",
           verify ? ", verify" : "");

    for (int idx = 0; idx < total; ++idx) {
        const TestPoint& p = points[idx];

        if (verify) {
            et_sgemm_params params = {};
            params.M = p.m;
            params.N = p.n;
            params.K = p.k;
            params.A = d_src0;
            params.lda = static_cast<int64_t>(p.k) * sizeof(float);
            params.B = d_src1;
            params.ldb = static_cast<int64_t>(p.k) * sizeof(float);
            params.C = d_dst;
            params.ldc = static_cast<int64_t>(p.m) * sizeof(float);
            params.batch_count = 1;

            runtime->kernelLaunch(stream,
                                  kernel,
                                  reinterpret_cast<const std::byte*>(&params),
                                  sizeof(params),
                                  opts);
            runtime->waitForStream(stream);

            const size_t c_elems = static_cast<size_t>(p.m) * p.n;
            std::vector<float> h_C_dev(c_elems);
            runtime->memcpyDeviceToHost(stream,
                                        d_dst,
                                        reinterpret_cast<std::byte*>(h_C_dev.data()),
                                        c_elems * sizeof(float));
            runtime->waitForStream(stream);

            std::vector<float> h_C_ref(c_elems, 0.0f);
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        p.n,
                        p.m,
                        p.k,
                        1.0f,
                        h_B.data(),
                        p.k,
                        h_A.data(),
                        p.k,
                        0.0f,
                        h_C_ref.data(),
                        p.m);

            const float rel_tol = std::sqrt(static_cast<float>(p.k)) * 1e-5f;
            const float abs_tol = rel_tol;
            float max_abs = 0.0f;
            float max_rel = 0.0f;
            float max_violation = 0.0f;
            int worst_abs_idx = 0;
            int worst_rel_idx = 0;
            int worst_violation_idx = 0;
            for (size_t i = 0; i < c_elems; ++i) {
                const float diff = std::fabs(h_C_dev[i] - h_C_ref[i]);
                const float mag = std::fabs(h_C_ref[i]);
                if (diff > max_abs) {
                    max_abs = diff;
                    worst_abs_idx = static_cast<int>(i);
                }

                const float rel = (mag > 0.0f) ? diff / mag : 0.0f;
                if (rel > max_rel) {
                    max_rel = rel;
                    worst_rel_idx = static_cast<int>(i);
                }

                const float allowed = abs_tol + rel_tol * mag;
                const float violation = diff / allowed;
                if (violation > max_violation) {
                    max_violation = violation;
                    worst_violation_idx = static_cast<int>(i);
                }
            }

            const bool pass = max_violation <= 1.0f;
            if (pass) {
                ++verify_pass;
            } else {
                ++verify_fail;
            }

            printf("  [%d/%d]  M=%5d K=%5d N=%4d  %s  "
                   "max_abs=%.3e max_rel=%.3e max_v=%.3e (atol=%.1e rtol=%.1e)\n",
                   idx + 1,
                   total,
                   p.m,
                   p.k,
                   p.n,
                   pass ? "PASS" : "FAIL",
                   max_abs,
                   max_rel,
                   max_violation,
                   abs_tol,
                   rel_tol);
            if (!pass) {
                printf("           worst_abs @ %d: device=%.6e ref=%.6e\n",
                       worst_abs_idx,
                       h_C_dev[worst_abs_idx],
                       h_C_ref[worst_abs_idx]);
                printf("           worst_rel @ %d: device=%.6e ref=%.6e\n",
                       worst_rel_idx,
                       h_C_dev[worst_rel_idx],
                       h_C_ref[worst_rel_idx]);
                printf("           worst_v   @ %d: device=%.6e ref=%.6e\n",
                       worst_violation_idx,
                       h_C_dev[worst_violation_idx],
                       h_C_ref[worst_violation_idx]);
            }
            continue;
        }

        BenchResult out = {};
        auto measure_batched = [&](const void* params, size_t params_size) {
            if (warmup_runs > 0) {
                for (int warmup = 0; warmup < warmup_runs; ++warmup) {
                    runtime->kernelLaunch(
                        stream,
                        kernel,
                        reinterpret_cast<const std::byte*>(params),
                        params_size,
                        opts);
                }
                runtime->waitForStream(stream);
            }

            double total_us = 0.0;
            int total_runs_measured = 0;
            int batch_count = 0;
            int next_batch_runs = std::max(min_runs, INITIAL_BATCH_RUNS);

            while (total_runs_measured < min_runs || total_us < MIN_TOTAL_US) {
                if (total_runs_measured >= MAX_RUNS) {
                    break;
                }

                int batch_runs =
                    std::min(next_batch_runs, MAX_RUNS - total_runs_measured);
                batch_runs = std::max(batch_runs, 1);

                const auto t0 = std::chrono::high_resolution_clock::now();
                for (int run = 0; run < batch_runs; ++run) {
                    runtime->kernelLaunch(
                        stream,
                        kernel,
                        reinterpret_cast<const std::byte*>(params),
                        params_size,
                        opts);
                }
                runtime->waitForStream(stream);
                const auto t1 = std::chrono::high_resolution_clock::now();

                total_us +=
                    std::chrono::duration<double, std::micro>(t1 - t0).count();
                total_runs_measured += batch_runs;
                ++batch_count;

                if (total_us < MIN_TOTAL_US && total_runs_measured < MAX_RUNS) {
                    const double avg_us =
                        total_us / std::max(total_runs_measured, 1);
                    const double remaining_us = MIN_TOTAL_US - total_us;
                    const int runs_for_time = static_cast<int>(
                        std::ceil(remaining_us / std::max(avg_us, 1.0)));
                    const int runs_for_min = min_runs - total_runs_measured;
                    next_batch_runs = std::max({1, runs_for_time, runs_for_min});
                    next_batch_runs =
                        std::min(next_batch_runs, MAX_BATCH_RUNS);
                }
            }

            out.us_per_run = total_us / std::max(total_runs_measured, 1);
            out.num_runs = total_runs_measured;
            out.num_batches = batch_count;
        };

        if (bench_type == MM) {
            et_sgemm_params params = {};
            params.M = p.m;
            params.N = p.n;
            params.K = p.k;
            params.A = d_src0;
            params.lda = static_cast<int64_t>(p.k) * sizeof(float);
            params.B = d_src1;
            params.ldb = static_cast<int64_t>(p.k) * sizeof(float);
            params.C = d_dst;
            params.ldc = static_cast<int64_t>(p.m) * sizeof(float);
            params.batch_count = 1;

            measure_batched(&params, sizeof(params));
            const double bytes =
                static_cast<double>(static_cast<int64_t>(p.m) * p.k +
                                    static_cast<int64_t>(p.n) * p.k) *
                sizeof(float);
            const double flops = 2.0 * p.m * p.n * p.k;

            out.gflops = flops / (out.us_per_run * 1e-6) / 1e9;
            out.bw_gbs = bytes / (out.us_per_run * 1e-6) / 1e9;
            out.bytes_read = bytes;
        } else {
            et_bw_params params = {};
            params.src0 = d_src0;
            params.src0_bytes = static_cast<int64_t>(p.m) * p.k * sizeof(float);
            params.src1 = d_src1;
            params.src1_bytes = static_cast<int64_t>(p.n) * p.k * sizeof(float);

            measure_batched(&params, sizeof(params));
            const double bytes =
                static_cast<double>(params.src0_bytes + params.src1_bytes);

            out.gflops = 0.0;
            out.bw_gbs = bytes / (out.us_per_run * 1e-6) / 1e9;
            out.bytes_read = bytes;
        }

        const auto elapsed = std::chrono::steady_clock::now() - t_start;
        const double elapsed_s = std::chrono::duration<double>(elapsed).count();
        const double rate = (idx + 1) / elapsed_s;
        const double eta_s = (total - idx - 1) / std::max(rate, 0.001);

        if (bench_type == BW) {
            printf("  [%d/%d %.0fs ETA %.0fs]  M=%5d K=%5d N=%5d  "
                   "%6.1fMB  %8.1fus  %6.2f GB/s  (%d runs, %d batches)\n",
                   idx + 1,
                   total,
                   elapsed_s,
                   eta_s,
                   p.m,
                   p.k,
                   p.n,
                   out.bytes_read / 1e6,
                   out.us_per_run,
                   out.bw_gbs,
                   out.num_runs,
                   out.num_batches);
            if (csv) {
                fprintf(csv,
                        "%d,%d,%d,%.0f,%.2f,%.4f,%d\n",
                        p.m,
                        p.k,
                        p.n,
                        out.bytes_read,
                        out.us_per_run,
                        out.bw_gbs,
                        out.num_runs);
            }
        } else {
            printf("  [%d/%d %.0fs ETA %.0fs]  M=%5d K=%5d N=%4d  "
                   "%7.1f GFLOPS  %5.1f GB/s  (%d runs, %d batches)\n",
                   idx + 1,
                   total,
                   elapsed_s,
                   eta_s,
                   p.m,
                   p.k,
                   p.n,
                   out.gflops,
                   out.bw_gbs,
                   out.num_runs,
                   out.num_batches);
            if (csv) {
                const double mflops = 2.0 * p.m * p.n * p.k / 1e6;
                fprintf(csv,
                        "%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%d\n",
                        p.m,
                        p.k,
                        p.n,
                        out.us_per_run,
                        mflops,
                        out.gflops,
                        out.bw_gbs,
                        out.num_runs);
            }
        }

        if (csv) {
            fflush(csv);
        }
    }

    if (csv) {
        fclose(csv);
    }

    runtime->freeDevice(device, d_src0);
    runtime->freeDevice(device, d_src1);
    runtime->freeDevice(device, d_dst);
    runtime->unloadCode(kernel);
    runtime->destroyStream(stream);

    if (verify) {
        printf("\nVerify: %d passed, %d failed.\n", verify_pass, verify_fail);
    } else {
        printf("\nDone. %d points measured.\n", total);
    }

    return verify_fail ? 1 : 0;
}
