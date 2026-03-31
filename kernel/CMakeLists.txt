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

add_riscv_executable(dram_bw)
target_sources(dram_bw.elf PRIVATE dram_bw.c crt.S)
target_include_directories(dram_bw.elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dram_bw.elf PRIVATE
    et-common-libs::cm-umode
    esperantoTrace::et_trace
)
set_target_properties(dram_bw.elf PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

add_riscv_executable(mul_mat_f32)
target_sources(mul_mat_f32.elf PRIVATE mul_mat_f32.c crt.S)
target_include_directories(mul_mat_f32.elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mul_mat_f32.elf PRIVATE
    et-common-libs::cm-umode
    esperantoTrace::et_trace
)
set_target_properties(mul_mat_f32.elf PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# The Q8 kernel includes static const data in headers (gather patterns,
# quantization tables) that need -fdata-sections and -ffat-lto-objects
# to survive LTO without R_RISCV_HI20 relocation failures at 0x80058xxxxx.
add_riscv_executable(mul_mat_q8)
target_sources(mul_mat_q8.elf PRIVATE mul_mat_q8.c crt.S)
target_include_directories(mul_mat_q8.elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(mul_mat_q8.elf PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffat-lto-objects>
    $<$<COMPILE_LANGUAGE:C>:-mcmodel=medany>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
)
target_link_libraries(mul_mat_q8.elf PRIVATE
    et-common-libs::cm-umode
    esperantoTrace::et_trace
)
set_target_properties(mul_mat_q8.elf PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

add_riscv_executable(diag)
target_sources(diag.elf PRIVATE diag.c crt.S)
target_include_directories(diag.elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(diag.elf PRIVATE
    et-common-libs::cm-umode
    esperantoTrace::et_trace
)
set_target_properties(diag.elf PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
