# etBenchBed: Device kernel build
#
# Cross-compiled in-tree with the RISC-V toolchain scoped to this directory.

include(${ET_PLATFORM_PATH}/lib/cmake/riscv64-ec-toolchain.cmake)
enable_language(ASM)

list(APPEND CMAKE_PREFIX_PATH "${ET_PLATFORM_PATH}/lib/cmake")
list(APPEND CMAKE_MODULE_PATH "${ET_PLATFORM_PATH}/lib/cmake/cmake-modules")

include(DeviceUtils)

find_package(et-common-libs REQUIRED)
find_package(esperantoTrace REQUIRED)

set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/sections.ld)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ── DRAM Bandwidth Benchmark Kernel ─────────────────────────────────
add_riscv_executable(dram_bw)
target_sources(dram_bw.elf PRIVATE dram_bw.c crt.S)
target_include_directories(dram_bw.elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dram_bw.elf PRIVATE
    et-common-libs::cm-umode
    esperantoTrace::et_trace
)
set_target_properties(dram_bw.elf PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# ── F32 Matrix Multiply Benchmark Kernel ────────────────────────────
add_riscv_executable(mul_mat_f32)
target_sources(mul_mat_f32.elf PRIVATE mul_mat_f32.c crt.S)
target_include_directories(mul_mat_f32.elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mul_mat_f32.elf PRIVATE
    et-common-libs::cm-umode
    esperantoTrace::et_trace
)
set_target_properties(mul_mat_f32.elf PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
