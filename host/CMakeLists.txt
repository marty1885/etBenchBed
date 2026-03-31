find_package(runtime REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(esperantoTrace REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)

if(NOT KERNELS_DIR)
    set(KERNELS_DIR "${CMAKE_BINARY_DIR}/kernel")
endif()

configure_file(Constants.h.in ${CMAKE_CURRENT_BINARY_DIR}/Constants.h)

add_executable(bench_host
    main.cpp
    bench_registry.cpp
    benchmark_runner.cpp
    quant_utils.cpp
)
target_include_directories(bench_host PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(bench_host PRIVATE cxx_std_17)
target_link_libraries(bench_host PRIVATE
    runtime::etrt
    deviceLayer::deviceLayer
    esperantoTrace::et_trace
    PkgConfig::OPENBLAS
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0.0>>:stdc++fs>
)
