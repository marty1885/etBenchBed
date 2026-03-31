#pragma once

#include <cstddef>
#include <cstdio>
#include <vector>

#include <runtime/IRuntime.h>
#include <runtime/Types.h>

#include "bench_registry.h"

namespace hostbench {

struct DeviceBuffers {
    std::byte* src0 = nullptr;
    std::byte* src1 = nullptr;
    std::byte* dst = nullptr;
};

struct HostData {
    std::vector<float> h_A;
    std::vector<float> h_B;
    std::vector<std::byte> h_A_raw;
};

struct LaunchSpec {
    et_sgemm_params mm_params = {};
    et_bw_params bw_params = {};
    size_t params_size = 0;

    const void* params_ptr(BenchKind kind) const {
        return (kind == BenchKind::Bandwidth)
            ? static_cast<const void*>(&bw_params)
            : static_cast<const void*>(&mm_params);
    }
};

struct VerifyResult {
    bool pass = false;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float max_violation = 0.0f;
    float abs_tol = 0.0f;
    float rel_tol = 0.0f;
    int worst_abs_idx = 0;
    int worst_rel_idx = 0;
    int worst_violation_idx = 0;
    std::vector<float> h_C_dev;
    std::vector<float> h_C_ref;
};

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
                              unsigned seed);

LaunchSpec build_launch_spec(const TestPoint& point,
                             const BenchDesc& bench,
                             const DeviceBuffers& device_buffers);

VerifyResult run_verify(rt::RuntimePtr& runtime,
                        rt::StreamId stream,
                        rt::KernelId kernel,
                        const rt::KernelLaunchOptions& launch_options,
                        const BenchDesc& bench,
                        const TestPoint& point,
                        const LaunchSpec& launch_spec,
                        const DeviceBuffers& device_buffers,
                        const HostData& host_data);

BenchResult run_perf(rt::RuntimePtr& runtime,
                     rt::StreamId stream,
                     rt::KernelId kernel,
                     const rt::KernelLaunchOptions& launch_options,
                     const BenchDesc& bench,
                     const TestPoint& point,
                     const LaunchSpec& launch_spec,
                     int min_runs,
                     int warmup_runs);

}  // namespace hostbench
