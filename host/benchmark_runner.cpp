#include "benchmark_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>

#include <cblas.h>

namespace hostbench {

void check_stream_errors(rt::RuntimePtr& runtime, rt::StreamId stream) {
    auto errors = runtime->retrieveStreamErrors(stream);
    if (errors.empty()) {
        return;
    }
    fprintf(stderr, "\nFATAL: %zu device error(s) after kernel launch:\n", errors.size());
    for (const auto& e : errors) {
        fprintf(stderr, "  %s\n", e.getString().c_str());
    }
    fprintf(stderr, "The device may need a reboot.\n");
    std::exit(2);
}

HostData initialize_host_data(rt::RuntimePtr& runtime,
                              rt::StreamId stream,
                              const DeviceBuffers& device_buffers,
                              int64_t max_mk,
                              int64_t max_nk,
                              int64_t max_m,
                              int64_t max_k,
                              size_t src0_bytes,
                              size_t src1_bytes,
                              const BenchDesc& bench,
                              bool verify,
                              unsigned seed) {
    HostData host_data;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a_f32(static_cast<size_t>(max_m) * max_k);
    for (float& value : a_f32) {
        value = dist(rng);
    }

    if (bench.kind == BenchKind::Matmul && bench.src_format->is_quantized()) {
        const SrcFormat* format = bench.src_format;
        const int64_t row_bytes = format->row_bytes(static_cast<int>(max_k));
        host_data.h_A_raw.resize(static_cast<size_t>(max_m) * row_bytes);

        for (int64_t row = 0; row < max_m; ++row) {
            format->quantize_row(
                a_f32.data() + row * max_k,
                host_data.h_A_raw.data() + row * row_bytes,
                static_cast<int>(max_k));
        }

        runtime->memcpyHostToDevice(
            stream,
            reinterpret_cast<const std::byte*>(host_data.h_A_raw.data()),
            device_buffers.src0,
            src0_bytes);

        if (verify) {
            host_data.h_A.resize(static_cast<size_t>(max_m) * max_k);
            for (int64_t row = 0; row < max_m; ++row) {
                format->dequantize_row(
                    host_data.h_A_raw.data() + row * row_bytes,
                    host_data.h_A.data() + row * max_k,
                    static_cast<int>(max_k));
            }
        }
    } else {
        runtime->memcpyHostToDevice(
            stream,
            reinterpret_cast<const std::byte*>(a_f32.data()),
            device_buffers.src0,
            src0_bytes);
        if (verify) {
            host_data.h_A = a_f32;
        }
    }

    host_data.h_B.resize(max_nk);
    for (float& value : host_data.h_B) {
        value = dist(rng);
    }

    runtime->memcpyHostToDevice(
        stream,
        reinterpret_cast<const std::byte*>(host_data.h_B.data()),
        device_buffers.src1,
        src1_bytes);
    runtime->waitForStream(stream);

    if (!verify) {
        host_data.h_A.clear();
        host_data.h_B.clear();
    }

    return host_data;
}

LaunchSpec build_launch_spec(const TestPoint& point,
                             const BenchDesc& bench,
                             const DeviceBuffers& device_buffers) {
    LaunchSpec spec;
    if (bench.kind == BenchKind::Bandwidth) {
        spec.bw_params.src0 = device_buffers.src0;
        spec.bw_params.src0_bytes = static_cast<int64_t>(point.m) * point.k * sizeof(float);
        spec.bw_params.src1 = device_buffers.src1;
        spec.bw_params.src1_bytes = static_cast<int64_t>(point.n) * point.k * sizeof(float);
        spec.params_size = sizeof(spec.bw_params);
    } else {
        spec.mm_params.M = point.m;
        spec.mm_params.N = point.n;
        spec.mm_params.K = point.k;
        spec.mm_params.A = device_buffers.src0;
        spec.mm_params.lda = bench.src_format->row_bytes(point.k);
        spec.mm_params.B = device_buffers.src1;
        spec.mm_params.ldb = static_cast<int64_t>(point.k) * sizeof(float);
        spec.mm_params.C = device_buffers.dst;
        spec.mm_params.ldc = static_cast<int64_t>(point.m) * sizeof(float);
        spec.mm_params.batch_count = 1;
        spec.params_size = sizeof(spec.mm_params);
    }
    return spec;
}

VerifyResult run_verify(rt::RuntimePtr& runtime,
                        rt::StreamId stream,
                        rt::KernelId kernel,
                        const rt::KernelLaunchOptions& launch_options,
                        const BenchDesc& bench,
                        const TestPoint& point,
                        const LaunchSpec& launch_spec,
                        const DeviceBuffers& device_buffers,
                        const HostData& host_data) {
    VerifyResult result;

    runtime->kernelLaunch(
        stream,
        kernel,
        reinterpret_cast<const std::byte*>(launch_spec.params_ptr(bench.kind)),
        launch_spec.params_size,
        launch_options);
    runtime->waitForStream(stream);
    check_stream_errors(runtime, stream);

    const size_t c_elems = static_cast<size_t>(point.m) * point.n;
    result.h_C_dev.resize(c_elems);
    runtime->memcpyDeviceToHost(
        stream,
        device_buffers.dst,
        reinterpret_cast<std::byte*>(result.h_C_dev.data()),
        c_elems * sizeof(float));
    runtime->waitForStream(stream);

    result.h_C_ref.assign(c_elems, 0.0f);
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        point.n,
        point.m,
        point.k,
        1.0f,
        host_data.h_B.data(),
        point.k,
        host_data.h_A.data(),
        point.k,
        0.0f,
        result.h_C_ref.data(),
        point.m);

    const float base_tol = std::sqrt(static_cast<float>(point.k)) * 1e-5f;
    result.rel_tol = base_tol * bench.src_format->verify_tol_scale;
    result.abs_tol = result.rel_tol;

    for (size_t i = 0; i < c_elems; ++i) {
        const float diff = std::fabs(result.h_C_dev[i] - result.h_C_ref[i]);
        const float mag = std::fabs(result.h_C_ref[i]);
        if (diff > result.max_abs) {
            result.max_abs = diff;
            result.worst_abs_idx = static_cast<int>(i);
        }
        const float rel = (mag > 0.0f) ? diff / mag : 0.0f;
        if (rel > result.max_rel) {
            result.max_rel = rel;
            result.worst_rel_idx = static_cast<int>(i);
        }
        const float violation = diff / (result.abs_tol + result.rel_tol * mag);
        if (violation > result.max_violation) {
            result.max_violation = violation;
            result.worst_violation_idx = static_cast<int>(i);
        }
    }

    result.pass = result.max_violation <= 1.0f;
    return result;
}

BenchResult run_perf(rt::RuntimePtr& runtime,
                     rt::StreamId stream,
                     rt::KernelId kernel,
                     const rt::KernelLaunchOptions& launch_options,
                     const BenchDesc& bench,
                     const TestPoint& point,
                     const LaunchSpec& launch_spec,
                     int min_runs,
                     int warmup_runs) {
    BenchResult result;

    for (int i = 0; i < warmup_runs; ++i) {
        runtime->kernelLaunch(
            stream,
            kernel,
            reinterpret_cast<const std::byte*>(launch_spec.params_ptr(bench.kind)),
            launch_spec.params_size,
            launch_options);
    }
    if (warmup_runs > 0) {
        runtime->waitForStream(stream);
        check_stream_errors(runtime, stream);
    }

    double total_us = 0.0;
    int total_runs = 0;
    int batch_count = 0;
    int next_batch = std::max(min_runs, kInitialBatchRuns);

    while (total_runs < min_runs || total_us < kMinTotalUs) {
        if (total_runs >= kMaxRuns) {
            break;
        }
        const int batch = std::min(next_batch, kMaxRuns - total_runs);

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < batch; ++i) {
            runtime->kernelLaunch(
                stream,
                kernel,
                reinterpret_cast<const std::byte*>(launch_spec.params_ptr(bench.kind)),
                launch_spec.params_size,
                launch_options);
        }
        runtime->waitForStream(stream);
        const auto t1 = std::chrono::high_resolution_clock::now();

        check_stream_errors(runtime, stream);

        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        total_runs += batch;
        ++batch_count;

        if (total_us < kMinTotalUs && total_runs < kMaxRuns) {
            const double avg = total_us / std::max(total_runs, 1);
            const int need_time = static_cast<int>(std::ceil((kMinTotalUs - total_us) / std::max(avg, 1.0)));
            const int need_min = min_runs - total_runs;
            next_batch = std::min(std::max({1, need_time, need_min}), kMaxBatchRuns);
        }
    }

    result.us_per_run = total_us / std::max(total_runs, 1);
    result.num_runs = total_runs;
    result.num_batches = batch_count;

    if (bench.kind == BenchKind::Bandwidth) {
        const double bytes = static_cast<double>(
            launch_spec.bw_params.src0_bytes + launch_spec.bw_params.src1_bytes);
        result.bytes_read = bytes;
        result.bw_gbs = bytes / (result.us_per_run * 1e-6) / 1e9;
    } else {
        const double bytes =
            static_cast<double>(point.m) * bench.src_format->row_bytes(point.k) +
            static_cast<double>(point.n) * point.k * sizeof(float);
        const double flops = 2.0 * point.m * point.n * point.k;
        result.bytes_read = bytes;
        result.gflops = flops / (result.us_per_run * 1e-6) / 1e9;
        result.bw_gbs = bytes / (result.us_per_run * 1e-6) / 1e9;
    }

    return result;
}

}  // namespace hostbench
