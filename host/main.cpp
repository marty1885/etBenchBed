/*
 * ET-SoC-1 Benchmark Host Program
 *
 * main() stays script-like, but the bench catalog and non-core helpers live
 * in a central registry header instead of being smeared across this file.
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>

#include "benchmark_runner.h"
#include "Constants.h"
#include "bench_registry.h"

namespace fs = std::filesystem;
using namespace hostbench;

namespace {

struct CliOptions {
    const BenchDesc* bench = default_bench();
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
    bool diag = false;
};

struct AllocationPlan {
    int64_t max_mk = 0;
    int64_t max_nk = 0;
    int64_t max_mn = 0;
    int64_t max_m = 0;
    int64_t max_k = 0;
    size_t src0_bytes = 0;
    size_t src1_bytes = 0;
    size_t dst_bytes = 0;
};

bool parse_cli(int argc, char** argv, CliOptions& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            options.bench = find_bench(argv[++i]);
            if (!options.bench) {
                error = "Unknown bench '" + std::string(argv[i]) + "'. Available: " + bench_names_str();
                return false;
            }
        } else if (std::strcmp(argv[i], "--sweep") == 0) {
            options.sweep = true;
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            options.verify = true;
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            options.M = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            options.K = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            options.N = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            options.min_runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            options.warmup_runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            options.csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            options.kernel_override = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            options.seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 0));
            options.custom_seed = true;
        } else if (std::strcmp(argv[i], "--diag") == 0) {
            options.diag = true;
        } else {
            error = "Invalid arguments";
            return false;
        }
    }
    return true;
}

void print_usage(const char* bench_names) {
    fprintf(stderr,
        "Usage: bench_host [--bench %s] [options]\n"
        "       bench_host --diag\n"
        "\n"
        "Options:\n"
        "  --bench NAME    Benchmark to run: %s\n"
        "  -m M            M dimension (default: 4096)\n"
        "  -k K            K dimension (default: 4096)\n"
        "  -n N            N dimension (default: 16)\n"
        "  --sweep         Run predefined sweep points\n"
        "  --verify        Verify device output against OpenBLAS reference\n"
        "  --runs N        Minimum runs per point (default: 10)\n"
        "  --warmup N      Warmup runs (default: 2)\n"
        "  --csv FILE      Write results to CSV\n"
        "  --kernel ELF    Override kernel binary path\n"
        "  --seed SEED     Random seed for input data (default: 42)\n"
        "  --diag          Run device diagnostic (upload, kernel exec, readback)\n",
        bench_names, bench_names);
}

std::vector<std::byte> read_file(const std::string& path) {
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

std::vector<TestPoint> build_points(const CliOptions& options) {
    if (options.sweep) {
        return options.bench->sweep_points();
    }
    return {{options.M, options.K, options.N}};
}

AllocationPlan plan_allocations(const std::vector<TestPoint>& points, const BenchDesc& bench) {
    AllocationPlan plan;
    for (const TestPoint& point : points) {
        plan.max_mk = std::max(plan.max_mk, static_cast<int64_t>(point.m) * point.k);
        plan.max_nk = std::max(plan.max_nk, static_cast<int64_t>(point.n) * point.k);
        plan.max_mn = std::max(plan.max_mn, static_cast<int64_t>(point.m) * point.n);
        plan.max_m = std::max(plan.max_m, static_cast<int64_t>(point.m));
        plan.max_k = std::max(plan.max_k, static_cast<int64_t>(point.k));
    }

    if (bench.kind == BenchKind::Matmul && bench.src_format->is_quantized()) {
        plan.src0_bytes = static_cast<size_t>(plan.max_m) * bench.src_format->row_bytes(static_cast<int>(plan.max_k));
    } else {
        plan.src0_bytes = static_cast<size_t>(plan.max_mk) * sizeof(float);
    }
    plan.src1_bytes = static_cast<size_t>(plan.max_nk) * sizeof(float);
    plan.dst_bytes = static_cast<size_t>(plan.max_mn) * sizeof(float);
    return plan;
}

FILE* open_csv(const CliOptions& options, const BenchDesc& bench) {
    if (options.csv_path.empty()) {
        return nullptr;
    }
    FILE* csv = fopen(options.csv_path.c_str(), "w");
    if (!csv) {
        fprintf(stderr, "Cannot open CSV: %s\n", options.csv_path.c_str());
        return nullptr;
    }
    if (bench.perf_mode.csv_header) {
        fputs(bench.perf_mode.csv_header, csv);
    }
    return csv;
}

void write_csv_row(FILE* csv, const BenchDesc& bench, const TestPoint& point, const BenchResult& result) {
    if (!csv) {
        return;
    }
    if (bench.kind == BenchKind::Bandwidth) {
        fprintf(
            csv, "%d,%d,%d,%.0f,%.2f,%.4f,%d\n",
            point.m, point.k, point.n,
            result.bytes_read, result.us_per_run, result.bw_gbs, result.num_runs);
        return;
    }
    const double mflops = 2.0 * point.m * point.n * point.k / 1e6;
    fprintf(
        csv, "%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%d\n",
        point.m, point.k, point.n,
        result.us_per_run, mflops, result.gflops, result.bw_gbs, result.num_runs);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(bench_names_str().c_str());
            return 0;
        }
    }

    CliOptions options;
    std::string cli_error;
    if (!parse_cli(argc, argv, options, cli_error)) {
        if (cli_error.rfind("Unknown bench", 0) == 0) {
            std::fprintf(stderr, "%s\n", cli_error.c_str());
        } else {
            print_usage(bench_names_str().c_str());
        }
        return 1;
    }

    const RunMode run_mode = options.verify ? RunMode::Verify : RunMode::Perf;
    if (!supports_mode(*options.bench, run_mode)) {
        std::fprintf(stderr, "--verify is not supported with --bench %s\n", options.bench->cli_name);
        return 1;
    }

    if (options.bench->validate) {
        const char* err = options.bench->validate(options.M, options.K, options.N);
        if (err) {
            std::fprintf(stderr, "%s\n", err);
            return 1;
        }
    }

    printf("Initializing PCIe device...\n");
    std::shared_ptr<dev::IDeviceLayer> device_layer(
        dev::IDeviceLayer::createPcieDeviceLayer().release());
    rt::RuntimePtr runtime = rt::IRuntime::create(device_layer);

    const std::vector<rt::DeviceId> devices = runtime->getDevices();
    if (devices.empty()) {
        std::fprintf(stderr, "No ET devices found\n");
        return 1;
    }

    const rt::DeviceId device = devices[0];
    const rt::StreamId stream = runtime->createStream(device);
    printf("Device ready.\n");

    if (options.diag) {
        struct { const void* buf; int64_t count; } diag_params;
        constexpr int N = 64;
        const size_t bytes = N * sizeof(uint64_t);

        std::byte* d_buf = runtime->mallocDevice(device, bytes);
        printf("diag: device buffer = %p (%zu bytes)\n", (void*)d_buf, bytes);

        // Fill with sentinel on host, copy to device
        std::vector<uint64_t> h_buf(N, 0xDEADBEEFDEADBEEFULL);
        runtime->memcpyHostToDevice(stream,
            reinterpret_cast<const std::byte*>(h_buf.data()), d_buf, bytes);
        runtime->waitForStream(stream);

        // Read back sentinel to confirm upload works
        std::vector<uint64_t> h_check(N, 0);
        runtime->memcpyDeviceToHost(stream, d_buf,
            reinterpret_cast<std::byte*>(h_check.data()), bytes);
        runtime->waitForStream(stream);
        bool upload_ok = true;
        for (int i = 0; i < N; i++) {
            if (h_check[i] != 0xDEADBEEFDEADBEEFULL) { upload_ok = false; break; }
        }
        printf("diag: upload roundtrip: %s\n", upload_ok ? "OK" : "FAIL");

        // Load and launch diag kernel
        const std::string kpath = (fs::path(KERNELS_DIR) / "diag.elf").string();
        printf("diag: loading %s\n", kpath.c_str());
        const std::vector<std::byte> elf = read_file(kpath);
        const rt::LoadCodeResult lr = runtime->loadCode(stream, elf.data(), elf.size());
        runtime->waitForEvent(lr.event_);

        diag_params.buf = d_buf;
        diag_params.count = N;

        rt::KernelLaunchOptions opts;
        opts.setShireMask(kShireMask);
        opts.setBarrier(true);
        opts.setFlushL3(false);

        printf("diag: launching kernel (buf=%p, count=%ld)\n",
               diag_params.buf, (long)diag_params.count);
        runtime->kernelLaunch(stream, lr.kernel_,
            reinterpret_cast<const std::byte*>(&diag_params),
            sizeof(diag_params), opts);
        runtime->waitForStream(stream);
        check_stream_errors(runtime, stream);
        printf("diag: kernel done\n");

        // Read back
        std::vector<uint64_t> h_result(N, 0);
        runtime->memcpyDeviceToHost(stream, d_buf,
            reinterpret_cast<std::byte*>(h_result.data()), bytes);
        runtime->waitForStream(stream);

        int pass = 0, fail = 0;
        for (int i = 0; i < N; i++) {
            uint64_t expected = (uint64_t)(i + 1);
            if (h_result[i] == expected) {
                pass++;
            } else {
                if (fail < 8) {
                    printf("diag: [%d] got 0x%016lx expected 0x%016lx%s\n",
                           i, h_result[i], expected,
                           h_result[i] == 0xDEADBEEFDEADBEEFULL ? " (sentinel)" : "");
                }
                fail++;
            }
        }
        printf("diag: %d/%d pass, %d fail\n", pass, N, fail);

        runtime->unloadCode(lr.kernel_);
        runtime->freeDevice(device, d_buf);
        runtime->destroyStream(stream);
        return fail ? 1 : 0;
    }

    const std::string kernel_path = options.kernel_override.empty()
        ? (fs::path(KERNELS_DIR) / (std::string(options.bench->kernel_name) + ".elf")).string()
        : options.kernel_override;

    printf("Loading kernel: %s\n", kernel_path.c_str());
    const std::vector<std::byte> elf_data = read_file(kernel_path);
    const rt::LoadCodeResult load_result =
        runtime->loadCode(stream, elf_data.data(), elf_data.size());
    runtime->waitForEvent(load_result.event_);
    const rt::KernelId kernel = load_result.kernel_;
    printf("Kernel loaded: %s\n", options.bench->kernel_name);

    const std::vector<TestPoint> points = build_points(options);
    const AllocationPlan allocation = plan_allocations(points, *options.bench);

    if (options.custom_seed) {
        printf("Using custom seed: %u\n", options.seed);
    }
    printf(
        "Allocating: src0=%.1fMB src1=%.1fMB dst=%.1fMB\n",
        allocation.src0_bytes / 1e6,
        allocation.src1_bytes / 1e6,
        allocation.dst_bytes / 1e6);

    DeviceBuffers device_buffers = {
        runtime->mallocDevice(device, allocation.src0_bytes),
        runtime->mallocDevice(device, allocation.src1_bytes),
        runtime->mallocDevice(device, allocation.dst_bytes),
    };
    HostData host_data = initialize_host_data(
        runtime,
        stream,
        device_buffers,
        allocation.max_mk,
        allocation.max_nk,
        allocation.max_m,
        allocation.max_k,
        allocation.src0_bytes,
        allocation.src1_bytes,
        *options.bench,
        options.verify,
        options.seed);

    rt::KernelLaunchOptions opts;
    opts.setShireMask(kShireMask);
    opts.setBarrier(true);
    opts.setFlushL3(false);

    FILE* csv = open_csv(options, *options.bench);
    if (!options.csv_path.empty() && !csv) {
        runtime->freeDevice(device, device_buffers.src0);
        runtime->freeDevice(device, device_buffers.src1);
        runtime->freeDevice(device, device_buffers.dst);
        runtime->unloadCode(kernel);
        runtime->destroyStream(stream);
        return 1;
    }

    const int total = static_cast<int>(points.size());
    int verify_pass = 0;
    int verify_fail = 0;
    const auto t_start = std::chrono::steady_clock::now();

    printf(
        "\nRunning %d test points (%s%s):\n\n",
        total,
        options.bench->label,
        options.verify ? ", verify" : "");

    for (int idx = 0; idx < total; ++idx) {
        const TestPoint& p = points[idx];

        const LaunchSpec launch_spec = build_launch_spec(p, *options.bench, device_buffers);

        if (options.verify) {
            const VerifyResult verify_result = run_verify(
                runtime, stream, kernel, opts, *options.bench, p, launch_spec, device_buffers, host_data);
            verify_result.pass ? ++verify_pass : ++verify_fail;

            printf(
                "  [%d/%d]  M=%5d K=%5d N=%4d  %s  max_abs=%.6f max_rel=%.6f "
                "max_v=%.4f (atol=%.6f rtol=%.6f)\n",
                idx + 1,
                total,
                p.m,
                p.k,
                p.n,
                verify_result.pass ? "PASS" : "FAIL",
                verify_result.max_abs,
                verify_result.max_rel,
                verify_result.max_violation,
                verify_result.abs_tol,
                verify_result.rel_tol);
            if (!verify_result.pass) {
                printf(
                    "           worst_abs @ %d: device=%.6f ref=%.6f diff=%.6f\n",
                    verify_result.worst_abs_idx,
                    verify_result.h_C_dev[verify_result.worst_abs_idx],
                    verify_result.h_C_ref[verify_result.worst_abs_idx],
                    verify_result.h_C_dev[verify_result.worst_abs_idx] -
                        verify_result.h_C_ref[verify_result.worst_abs_idx]);
                printf(
                    "           worst_rel @ %d: device=%.6f ref=%.6f rel=%.6f\n",
                    verify_result.worst_rel_idx,
                    verify_result.h_C_dev[verify_result.worst_rel_idx],
                    verify_result.h_C_ref[verify_result.worst_rel_idx],
                    std::fabs(verify_result.h_C_ref[verify_result.worst_rel_idx]) > 0.0f
                        ? std::fabs(
                              verify_result.h_C_dev[verify_result.worst_rel_idx] -
                              verify_result.h_C_ref[verify_result.worst_rel_idx]) /
                              std::fabs(verify_result.h_C_ref[verify_result.worst_rel_idx])
                        : 0.0f);
                printf(
                    "           worst_v   @ %d: device=%.6f ref=%.6f violation=%.4f\n",
                    verify_result.worst_violation_idx,
                    verify_result.h_C_dev[verify_result.worst_violation_idx],
                    verify_result.h_C_ref[verify_result.worst_violation_idx],
                    verify_result.max_violation);
            }
            continue;
        }

        const BenchResult result = run_perf(
            runtime,
            stream,
            kernel,
            opts,
            *options.bench,
            p,
            launch_spec,
            options.min_runs,
            options.warmup_runs);

        const auto elapsed = std::chrono::steady_clock::now() - t_start;
        const double elapsed_s = std::chrono::duration<double>(elapsed).count();
        const double rate = (idx + 1) / elapsed_s;
        const double eta_s = (total - idx - 1) / std::max(rate, 0.001);

        if (options.bench->kind == BenchKind::Bandwidth) {
            printf(
                "  [%d/%d %.0fs ETA %.0fs]  M=%5d K=%5d N=%5d  %6.1fMB  %8.1fus  "
                "%6.2f GB/s  (%d runs, %d batches)\n",
                idx + 1,
                total,
                elapsed_s,
                eta_s,
                p.m,
                p.k,
                p.n,
                result.bytes_read / 1e6,
                result.us_per_run,
                result.bw_gbs,
                result.num_runs,
                result.num_batches);
        } else {
            printf(
                "  [%d/%d %.0fs ETA %.0fs]  M=%5d K=%5d N=%4d  %7.1f GFLOPS  %5.1f GB/s  "
                "(%d runs, %d batches)\n",
                idx + 1,
                total,
                elapsed_s,
                eta_s,
                p.m,
                p.k,
                p.n,
                result.gflops,
                result.bw_gbs,
                result.num_runs,
                result.num_batches);
        }

        write_csv_row(csv, *options.bench, p, result);
        if (csv) fflush(csv);
    }

    if (csv) fclose(csv);
    runtime->freeDevice(device, device_buffers.src0);
    runtime->freeDevice(device, device_buffers.src1);
    runtime->freeDevice(device, device_buffers.dst);
    runtime->unloadCode(kernel);
    runtime->destroyStream(stream);

    if (options.verify) {
        printf("\nVerify: %d passed, %d failed.\n", verify_pass, verify_fail);
    } else {
        printf("\nDone. %d points measured.\n", total);
    }

    return verify_fail ? 1 : 0;
}
