# etBenchBed: ET-SoC-1 Benchmark Suite
#
# Build:
#   cmake -B build -DCMAKE_PREFIX_PATH=/opt/et
#   cmake --build build
#
# Run:
#   ./build/host/bench_host --bench bw -m 4096 -k 4096 -n 16
#   ./build/host/bench_host --bench mm -m 4096 -k 4096 -n 16
#   ./build/host/bench_host --bench bw --sweep --csv bw_sweep.csv

cmake_minimum_required(VERSION 3.17)
project(etBenchBed VERSION 0.2.0)

if(DEFINED ENV{ET_PLATFORM})
    set(ET_PLATFORM_PATH $ENV{ET_PLATFORM})
else()
    set(ET_PLATFORM_PATH "/opt/et")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ET_PLATFORM_PATH}/lib/cmake")
list(APPEND CMAKE_MODULE_PATH "${ET_PLATFORM_PATH}/lib/cmake")
list(APPEND CMAKE_MODULE_PATH "${ET_PLATFORM_PATH}/lib/cmake/cmake-modules")

# --- Device kernels (cross-compiled in-tree) ---
set(KERNEL_BINARY_DIR ${CMAKE_BINARY_DIR}/kernel)
add_subdirectory(kernel)

# --- Host program ---
add_subdirectory(host)
