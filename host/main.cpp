/*
 * ET-SoC-1 Benchmark Host Program
 *
 * main() stays script-like, but the bench catalog and non-core helpers live
 * in a central registry header instead of being smeared across this file.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tuple>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
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
    bool hang = false;
    bool overhead = false;
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
        } else if (std::strcmp(argv[i], "--hang") == 0) {
            options.hang = true;
        } else if (std::strcmp(argv[i], "--overhead") == 0) {
            options.overhead = true;
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
        "  --diag          Run device diagnostic\n  --hang          Trigger the hang repro kernel (uses -k for CACHEOP_MAX, -m for REP_RATE) (upload, kernel exec, readback)\n  --overhead      Measure kernel op-op gap and launch overhead via empty kernels (uses -m for batch size B, -n for outer repeats)\n",
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
    if (options.hang) {
        struct { uint64_t rep_rate; uint64_t cacheop_max; } hang_params;
        hang_params.rep_rate = options.M; // Using -m for REP_RATE
        hang_params.cacheop_max = options.K; // Using -k for CACHEOP_MAX

        const std::string kpath = (fs::path(KERNELS_DIR) / "hang.elf").string();
        printf("hang: loading %s (REP_RATE=%ld, CACHEOP_MAX=%ld)\n",
               kpath.c_str(), (long)hang_params.rep_rate, (long)hang_params.cacheop_max);

        const std::vector<std::byte> elf = read_file(kpath);
        const rt::LoadCodeResult lr = runtime->loadCode(stream, elf.data(), elf.size());
        runtime->waitForEvent(lr.event_);

        rt::KernelLaunchOptions opts;
        opts.setShireMask(kShireMask);
        opts.setBarrier(true);
        opts.setFlushL3(false);

        printf("hang: launching kernel...\n");
        runtime->kernelLaunch(stream, lr.kernel_,
            reinterpret_cast<const std::byte*>(&hang_params),
            sizeof(hang_params), opts);

        printf("hang: waiting for kernel (it might hang here)...\n");
        runtime->waitForStream(stream);
        printf("hang: kernel returned (no hang!)\n");

        runtime->unloadCode(lr.kernel_);
        runtime->destroyStream(stream);
        return 0;
    }
    if (options.overhead) {
        // Batch B of empty kernels per launch group; repeat R outer groups.
        // Each kernel writes (t_start, t_end) mcycle pair to slot[idx].
        const int B = std::max(2, options.M);          // -m: batch size, need >=2 for gap
        const int R = std::max(1, options.N);          // -n: outer repeats
        constexpr size_t kSlot = 64;                    // one cacheline per slot
        const size_t bytes = static_cast<size_t>(B) * kSlot;

        std::byte* d_buf = runtime->mallocDevice(device, bytes);

        const std::string kpath = (fs::path(KERNELS_DIR) / "overhead.elf").string();
        const std::vector<std::byte> elf = read_file(kpath);
        const rt::LoadCodeResult lr = runtime->loadCode(stream, elf.data(), elf.size());
        runtime->waitForEvent(lr.event_);

        rt::KernelLaunchOptions opts;
        opts.setShireMask(kShireMask);
        opts.setBarrier(true);
        opts.setFlushL3(false);   // kernel-boundary horizon already covers visibility

        struct overhead_params { const void* out; uint64_t idx; };

        // Capture runtime trace so we can recover device-wall (CommandSent ->
        // ResponseReceived) per launch. body_cyc is read from kernel via
        // hpmcounter3; device_wall - body gives the firmware envelope (M-mode
        // boot + teardown + transport).
        std::stringstream prof_ss;
        runtime->getProfiler()->start(prof_ss, rt::IProfiler::OutputType::Json);

        // Warmup (one full batch, discarded).
        for (int i = 0; i < B; ++i) {
            overhead_params p{d_buf, static_cast<uint64_t>(i)};
            runtime->kernelLaunch(stream, lr.kernel_,
                reinterpret_cast<const std::byte*>(&p), sizeof(p), opts);
        }
        runtime->waitForStream(stream);
        check_stream_errors(runtime, stream);

        std::vector<uint8_t> ts_raw(bytes, 0);
        // helpers indexing into the cacheline-strided buffer
        auto T0 = [&](int i) -> uint64_t {
            uint64_t v;
            std::memcpy(&v, ts_raw.data() + i * kSlot + 0, sizeof(v));
            return v;
        };
        auto T1 = [&](int i) -> uint64_t {
            uint64_t v;
            std::memcpy(&v, ts_raw.data() + i * kSlot + 8, sizeof(v));
            return v;
        };

        // Stats accumulators.
        double host_wall_total_us = 0.0;        // batched per-kernel host wall
        double solo_wall_total_us = 0.0;        // single-kernel host wall
        long long solo_count = 0;
        std::vector<double> batched_per_kernel_us_samples;  // us/kernel per batch
        std::vector<double> solo_us_samples;                // us per solo launch

        // Per-pair stats across all R*((B-1)) gap samples.
        std::vector<uint64_t> gap_samples;
        gap_samples.reserve(static_cast<size_t>(R) * (B - 1));
        std::vector<uint64_t> body_samples;
        body_samples.reserve(static_cast<size_t>(R) * B);
        std::vector<uint64_t> body_in_order;            // launch-order, for fw_env subtraction
        body_in_order.reserve(static_cast<size_t>(R) * B);
        // For freq derivation: span of (t_start[B-1]-t_start[0]) per batch vs host wall span of that batch
        double freq_num_cycles = 0.0;
        double freq_den_us = 0.0;

        // --- Batched runs ---
        int wrap_body = 0;
        int wrap_gap = 0;
        for (int r = 0; r < R; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < B; ++i) {
                overhead_params p{d_buf, static_cast<uint64_t>(i)};
                runtime->kernelLaunch(stream, lr.kernel_,
                    reinterpret_cast<const std::byte*>(&p), sizeof(p), opts);
            }
            runtime->waitForStream(stream);
            const auto t1 = std::chrono::high_resolution_clock::now();
            check_stream_errors(runtime, stream);

            const double batch_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            host_wall_total_us += batch_us;
            batched_per_kernel_us_samples.push_back(batch_us / B);

            runtime->memcpyDeviceToHost(stream, d_buf,
                reinterpret_cast<std::byte*>(ts_raw.data()), bytes);
            runtime->waitForStream(stream);

            if (r == 0) {
                const int dump_n = std::min(B, 32);
                printf("\nraw[r=0] %-3s %20s %20s %14s %14s\n",
                       "i", "t_start", "t_end", "body_cyc", "gap_to_next");
                for (int i = 0; i < dump_n; ++i) {
                    const uint64_t a = T0(i);
                    const uint64_t b = T1(i);
                    const int64_t body_d = (int64_t)(b - a);
                    int64_t gap_d = 0;
                    if (i + 1 < B) gap_d = (int64_t)(T0(i + 1) - b);
                    printf("raw[r=0] %-3d %20lu %20lu %14ld %14ld\n",
                           i, (unsigned long)a, (unsigned long)b,
                           (long)body_d, (long)gap_d);
                }
            }

            for (int i = 0; i < B; ++i) {
                const uint64_t a = T0(i);
                const uint64_t b = T1(i);
                if (b >= a && (b - a) < (1ULL << 32)) {
                    body_samples.push_back(b - a);
                    body_in_order.push_back(b - a);
                } else {
                    ++wrap_body;
                    body_in_order.push_back(0);  // sentinel; skipped in fw_env
                }
            }
            for (int i = 0; i + 1 < B; ++i) {
                const uint64_t t_end_i = T1(i);
                const uint64_t t_start_next = T0(i + 1);
                if (t_start_next >= t_end_i && (t_start_next - t_end_i) < (1ULL << 32)) {
                    gap_samples.push_back(t_start_next - t_end_i);
                } else {
                    ++wrap_gap;
                }
            }

            const uint64_t span_cycles = T0(B - 1) - T0(0);
            freq_num_cycles += static_cast<double>(span_cycles);
            freq_den_us += batch_us; // approximate: full batch wall ~ span + tail
        }

        // --- Solo runs (one kernel at a time, individually waited).
        // Each solo writes to slot i so we can recover per-launch body cycles.
        // Cap at B so each gets a distinct slot.
        const int solo_launches = std::min({64, std::max(8, R), B});
        // discard a couple
        for (int i = 0; i < 2; ++i) {
            overhead_params p{d_buf, 0};
            runtime->kernelLaunch(stream, lr.kernel_,
                reinterpret_cast<const std::byte*>(&p), sizeof(p), opts);
            runtime->waitForStream(stream);
        }
        check_stream_errors(runtime, stream);
        for (int i = 0; i < solo_launches; ++i) {
            overhead_params p{d_buf, static_cast<uint64_t>(i)};
            const auto t0 = std::chrono::high_resolution_clock::now();
            runtime->kernelLaunch(stream, lr.kernel_,
                reinterpret_cast<const std::byte*>(&p), sizeof(p), opts);
            runtime->waitForStream(stream);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            solo_wall_total_us += us;
            solo_us_samples.push_back(us);
            ++solo_count;
        }
        check_stream_errors(runtime, stream);
        // Readback solo timestamps; body_solo[i] = T1(i) - T0(i).
        runtime->memcpyDeviceToHost(stream, d_buf,
            reinterpret_cast<std::byte*>(ts_raw.data()), bytes);
        runtime->waitForStream(stream);
        std::vector<uint64_t> body_solo_in_order;
        body_solo_in_order.reserve(solo_launches);
        for (int i = 0; i < solo_launches; ++i) {
            const uint64_t a = T0(i);
            const uint64_t b = T1(i);
            if (b >= a && (b - a) < (1ULL << 32)) {
                body_solo_in_order.push_back(b - a);
            } else {
                body_solo_in_order.push_back(0);  // sentinel
            }
        }

        runtime->getProfiler()->stop();

        // --- Parse the captured trace to recover per-launch device-wall ---
        // Trace events are JSON value blocks with a "class" string, a
        // timeStamp.time_since_epoch.count (nanoseconds), and an extras list.
        // For KernelLaunch we want the "event" id; for CommandSent/
        // ResponseReceived (Instant) we get the device-side timestamps via
        // the same event id. We collect (cs_ns, rr_ns) pairs in launch order.
        std::vector<uint64_t> cs_in_order;
        std::vector<uint64_t> rr_in_order;
        {
            const std::string& s = prof_ss.str();
            size_t pos = 0;
            // Helper: find next occurrence of needle starting at pos.
            auto find_after = [&](const char* needle, size_t from) -> size_t {
                return s.find(needle, from);
            };
            auto extract_count_after = [&](size_t from) -> uint64_t {
                size_t p = s.find("\"count\":", from);
                if (p == std::string::npos) return 0;
                p += 8;
                while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n')) ++p;
                uint64_t v = 0;
                while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
                    v = v * 10 + (uint64_t)(s[p] - '0');
                    ++p;
                }
                return v;
            };
            // Walk the buffer block-by-block by anchoring on "\"class\":".
            while (pos < s.size()) {
                size_t cp = s.find("\"class\":", pos);
                if (cp == std::string::npos) break;
                // Read class string
                size_t q1 = s.find('"', cp + 8);
                if (q1 == std::string::npos) break;
                size_t q2 = s.find('"', q1 + 1);
                if (q2 == std::string::npos) break;
                std::string klass = s.substr(q1 + 1, q2 - q1 - 1);
                pos = q2 + 1;
                // The block's timestamp is the first "count" after the class.
                if (klass == "CommandSent") {
                    uint64_t ts = extract_count_after(pos);
                    if (ts) cs_in_order.push_back(ts);
                } else if (klass == "ResponseReceived") {
                    uint64_t ts = extract_count_after(pos);
                    if (ts) rr_in_order.push_back(ts);
                }
            }
        }

        // Pair CS/RR by encounter order. On a single stream with barrier,
        // they appear in launch order.
        std::vector<uint64_t> dev_wall_ns;        // RR - CS per launch
        const size_t pair_n = std::min(cs_in_order.size(), rr_in_order.size());
        dev_wall_ns.reserve(pair_n);
        for (size_t i = 0; i < pair_n; ++i) {
            if (rr_in_order[i] >= cs_in_order[i]) {
                dev_wall_ns.push_back(rr_in_order[i] - cs_in_order[i]);
            } else {
                dev_wall_ns.push_back(0);
            }
        }

        // Partition: first B are warmup, next R*B are batched, next 2 are
        // solo-discard, remainder are solo. Anything beyond is ignored.
        const size_t off_batched = (size_t)B;
        const size_t off_solo    = off_batched + (size_t)R * (size_t)B + 2;
        std::vector<uint64_t> dev_wall_batched;
        std::vector<uint64_t> dev_wall_solo;
        for (size_t i = off_batched; i < dev_wall_ns.size() && i < off_solo - 2; ++i) {
            dev_wall_batched.push_back(dev_wall_ns[i]);
        }
        for (size_t i = off_solo; i < dev_wall_ns.size(); ++i) {
            dev_wall_solo.push_back(dev_wall_ns[i]);
        }

        struct S {
            size_t n;
            double mean, stdev;
            uint64_t mn, p1, p10, p50, p90, p99, mx;
        };
        auto stats = [](std::vector<uint64_t>& v) {
            std::sort(v.begin(), v.end());
            S s{};
            s.n = v.size();
            if (s.n == 0) return s;
            double sum = 0.0, sq = 0.0;
            for (uint64_t x : v) { sum += (double)x; sq += (double)x * (double)x; }
            s.mean = sum / s.n;
            s.stdev = std::sqrt(std::max(0.0, sq / s.n - s.mean * s.mean));
            auto pct = [&](double p) -> uint64_t {
                size_t i = static_cast<size_t>(p * (s.n - 1));
                return v[i];
            };
            s.mn  = v.front();
            s.p1  = pct(0.01);
            s.p10 = pct(0.10);
            s.p50 = pct(0.50);
            s.p90 = pct(0.90);
            s.p99 = pct(0.99);
            s.mx  = v.back();
            return s;
        };

        const S g = stats(gap_samples);
        const S b = stats(body_samples);
        S dwall_b_ns = stats(dev_wall_batched);
        S dwall_s_ns = stats(dev_wall_solo);

        const double batched_per_kernel_us = host_wall_total_us / (static_cast<double>(R) * B);
        const double solo_per_kernel_us = solo_count ? solo_wall_total_us / solo_count : 0.0;
        const double cyc_per_us = freq_den_us > 0 ? freq_num_cycles / freq_den_us : 0.0;
        auto u = [&](double c) { return cyc_per_us > 0 ? c / cyc_per_us : 0.0; };

        printf(R"(
Measurment (time flows left -> right):

  Device sequence across adjacent kernels:
  ...   kernel k-1    | xxxxxxxxxxxxxxxxxxxxxx |           kernel k         | xxxxxxxxxxxxxxxxxxxxxx | kernel k + 1 .....
  ------------------------------------------------------------------------------------------------------------------------
                      |<-     op-op gap      ->|   fw  | kernel body |  fw  |<-     op-op gap      ->|

Where the kernel body is near empty (reads timer, write and leave).

)");
        printf("[overhead] B=%d R=%d batched=%d solo=%lld discard_body=%d discard_gap=%d cyc_per_us=%.2f\n",
               B, R, R * B, solo_count, wrap_body, wrap_gap, cyc_per_us);

        printf("\n%-32s %8s %14s %14s %12s %12s %12s %12s %12s %12s %12s\n",
               "metric", "n", "mean", "stdev", "min", "p1", "p10", "p50", "p90", "p99", "max");
        auto row_cyc = [&](const char* name, const S& s) {
            printf("%-32s %8zu %14.1f %14.1f %12lu %12lu %12lu %12lu %12lu %12lu %12lu\n",
                   name, s.n, s.mean, s.stdev,
                   (unsigned long)s.mn, (unsigned long)s.p1, (unsigned long)s.p10,
                   (unsigned long)s.p50, (unsigned long)s.p90, (unsigned long)s.p99,
                   (unsigned long)s.mx);
        };
        auto row_us = [&](const char* name, const S& s) {
            printf("%-32s %8zu %14.3f %14.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n",
                   name, s.n, u(s.mean), u(s.stdev),
                   u((double)s.mn), u((double)s.p1), u((double)s.p10),
                   u((double)s.p50), u((double)s.p90), u((double)s.p99),
                   u((double)s.mx));
        };
        // Headline row 1: estimated real op-op gap (on-device, batch-invariant)
        row_cyc("op-op gap cycles",   g);
        row_us ("op-op gap us",       g);
        // Reference: kernel body itself (so the gap has context).
        row_cyc("kernel body cycles", b);
        row_us ("kernel body us",     b);

        // Convert ns to us in stats rows.
        auto row_ns = [&](const char* name, const S& s) {
            printf("%-32s %8zu %14.3f %14.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n",
                   name, s.n, s.mean / 1000.0, s.stdev / 1000.0,
                   s.mn / 1000.0, s.p1 / 1000.0, s.p10 / 1000.0,
                   s.p50 / 1000.0, s.p90 / 1000.0, s.p99 / 1000.0,
                   s.mx / 1000.0);
        };

        // Headline row 2: single-kernel wall time = solo dev_wall (RR - CS,
        // no queue depth contamination). This is what one kernel costs end
        // to end, host-visible.
        row_ns("single kernel wall time us", dwall_s_ns);

        // Headline row 3: firmware overhead per kernel execution =
        // dev_wall_solo[i] - body_solo[i] in ns, per-launch. Includes host
        // <-> device transport; excludes queueing. Stable across batch size.
        std::vector<uint64_t> fw_overhead_ns;
        if (cyc_per_us > 0.0) {
            const size_t n = std::min(body_solo_in_order.size(), dev_wall_solo.size());
            fw_overhead_ns.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (body_solo_in_order[i] == 0) continue;
                const double body_ns = (double)body_solo_in_order[i] / cyc_per_us * 1000.0;
                const double dw = (double)dev_wall_solo[i];
                if (dw > body_ns) fw_overhead_ns.push_back((uint64_t)(dw - body_ns));
            }
        }
        S fw_overhead = stats(fw_overhead_ns);
        row_ns("firmware overhead per kernel us", fw_overhead);

        auto dstats = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            struct D { size_t n; double mean, stdev, mn, p50, p90, mx; } d{};
            d.n = v.size();
            if (!d.n) return d;
            double sum = 0.0, sq = 0.0;
            for (double x : v) { sum += x; sq += x * x; }
            d.mean  = sum / d.n;
            d.stdev = std::sqrt(std::max(0.0, sq / d.n - d.mean * d.mean));
            d.mn  = v.front();
            d.p50 = v[d.n / 2];
            d.p90 = v[(d.n * 9) / 10];
            d.mx  = v.back();
            return d;
        };
        const auto bw = dstats(batched_per_kernel_us_samples);
        const auto sw = dstats(solo_us_samples);

        printf("\n%-32s %8s %12s %12s %12s %12s %12s %12s\n",
               "host wall metric (us)", "n", "mean", "stdev", "min", "p50", "p90", "max");
        printf("%-32s %8zu %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n",
               "host wall batched per-kernel",
               bw.n, bw.mean, bw.stdev, bw.mn, bw.p50, bw.p90, bw.mx);
        printf("%-32s %8zu %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n",
               "host wall solo",
               sw.n, sw.mean, sw.stdev, sw.mn, sw.p50, sw.p90, sw.mx);
        printf("%-32s %8s %12.3f %12s %12s %12s %12s %12s\n",
               "host wall solo minus batched",
               "-", sw.mean - bw.mean, "-", "-", "-", "-", "-");

        runtime->unloadCode(lr.kernel_);
        runtime->freeDevice(device, d_buf);
        runtime->destroyStream(stream);
        return 0;
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
