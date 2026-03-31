#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hostbench {

struct et_sgemm_params {
    int64_t M, N, K;
    const void* A;
    int64_t lda;
    const void* B;
    int64_t ldb;
    void* C;
    int64_t ldc;
    int64_t batch_count;
    int64_t stride_A, stride_B, stride_C;
};

struct et_bw_params {
    const void* src0;
    int64_t src0_bytes;
    const void* src1;
    int64_t src1_bytes;
};

struct TestPoint {
    int m;
    int k;
    int n;
};

struct BenchResult {
    double us_per_run = 0.0;
    double gflops = 0.0;
    double bw_gbs = 0.0;
    double bytes_read = 0.0;
    int num_runs = 0;
    int num_batches = 0;
};

struct SrcFormat {
    const char* name;
    int block_elems;
    int block_bytes;
    float verify_tol_scale;
    void (*quantize_row)(const float* src, void* dst, int K);
    void (*dequantize_row)(const void* src, float* dst, int K);

    int64_t row_bytes(int K) const {
        return static_cast<int64_t>(K / block_elems) * block_bytes;
    }

    bool is_quantized() const { return quantize_row != nullptr; }
};

enum class BenchKind {
    Bandwidth,
    Matmul,
};

enum class RunMode {
    Perf,
    Verify,
};

struct ModeRegistration {
    bool enabled;
    const char* csv_header;
};

struct BenchDesc {
    const char* cli_name;
    const char* label;
    const char* kernel_name;
    std::vector<TestPoint> (*sweep_points)();
    const SrcFormat* src_format;
    const char* (*validate)(int M, int K, int N);
    BenchKind kind;
    ModeRegistration perf_mode;
    ModeRegistration verify_mode;
};

constexpr double kMinTotalUs = 500000.0;
constexpr int kMaxRuns = 50000;
constexpr int kInitialBatchRuns = 32;
constexpr int kMaxBatchRuns = 4096;
constexpr uint64_t kShireMask = 0xFFFFFFFF;

const std::array<BenchDesc, 3>& benchmark_registry();
const BenchDesc* default_bench();
bool supports_mode(const BenchDesc& bench, RunMode mode);
const BenchDesc* find_bench(const char* name);
std::string bench_names_str();

}  // namespace hostbench
