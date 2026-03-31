#include "bench_registry.h"

#include <array>
#include <cstring>

#include "quant_utils.h"

namespace hostbench {

namespace {

std::vector<TestPoint> bw_sweep_points() {
    std::vector<TestPoint> points;
    std::vector<int> mk_vals = {16, 32};
    for (int v = 64; v <= 8192; v += 64) {
        mk_vals.push_back(v);
    }
    std::vector<int> n_vals = {16, 32, 64};
    for (int v = 128; v <= 8192; v += 128) {
        n_vals.push_back(v);
    }
    for (int mk : mk_vals) {
        for (int n : n_vals) {
            points.push_back({mk, mk, n});
        }
    }
    return points;
}

std::vector<TestPoint> mm_sweep_points() {
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

std::vector<TestPoint> q8_sweep_points() {
    std::vector<TestPoint> points;
    const int mk[][2] = {
        {4096, 4096}, {4096, 11008}, {4096, 14336},
        {8192, 8192}, {8192, 28672},
        {2048, 2048}, {2048, 5504},
        {2496, 2496}, {3072, 3072},
    };
    const int n_vals[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    for (const auto& dims : mk) {
        for (int n : n_vals) {
            points.push_back({dims[0], dims[1], n});
        }
    }
    return points;
}

const char* validate_q8(int, int K, int) {
    return (K % kQk8_0 != 0) ? "Q8 requires K to be a multiple of 32" : nullptr;
}

const SrcFormat kFmtF32 = {
    "f32", 1, static_cast<int>(sizeof(float)), 1.0f, nullptr, nullptr
};

const SrcFormat kFmtQ8_0 = {
    "q8_0", kQk8_0, static_cast<int>(sizeof(block_q8_0)), 10.0f,
    quantize_row_q8_0, dequantize_row_q8_0
};

const std::array<BenchDesc, 3> kRegistry = {{
    {
        "bw",
        "BW",
        "dram_bw",
        bw_sweep_points,
        nullptr,
        nullptr,
        BenchKind::Bandwidth,
        {true, "M,K,N,BYTES_READ,US_PER_RUN,BW_GB_S,RUNS\n"},
        {false, nullptr},
    },
    {
        "mm",
        "MM",
        "mul_mat_f32",
        mm_sweep_points,
        &kFmtF32,
        nullptr,
        BenchKind::Matmul,
        {true, "M,K,N,US_PER_RUN,MFLOPS,GFLOPS,BW_GB_S,RUNS\n"},
        {true, nullptr},
    },
    {
        "mmq8",
        "MMQ8",
        "mul_mat_q8",
        q8_sweep_points,
        &kFmtQ8_0,
        validate_q8,
        BenchKind::Matmul,
        {true, "M,K,N,US_PER_RUN,MFLOPS,GFLOPS,BW_GB_S,RUNS\n"},
        {true, nullptr},
    },
}};

}  // namespace

const std::array<BenchDesc, 3>& benchmark_registry() {
    return kRegistry;
}

const BenchDesc* default_bench() {
    return &kRegistry.front();
}

bool supports_mode(const BenchDesc& bench, RunMode mode) {
    const ModeRegistration& registration =
        (mode == RunMode::Perf) ? bench.perf_mode : bench.verify_mode;
    return registration.enabled;
}

const BenchDesc* find_bench(const char* name) {
    for (const auto& bench : kRegistry) {
        if (std::strcmp(bench.cli_name, name) == 0) {
            return &bench;
        }
    }
    return nullptr;
}

std::string bench_names_str() {
    std::string names;
    for (const auto& bench : kRegistry) {
        if (!names.empty()) {
            names += '|';
        }
        names += bench.cli_name;
    }
    return names;
}

}  // namespace hostbench
