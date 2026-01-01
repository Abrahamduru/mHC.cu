cmake_minimum_required(VERSION 3.18)
project(mHC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100)

find_package(CUDAToolkit REQUIRED)

add_library(mhc_kernels INTERFACE)
target_include_directories(mhc_kernels INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
)
target_link_libraries(mhc_kernels INTERFACE
    CUDA::cudart
    CUDA::cublasLt
)

option(MHC_ENABLE_PDL "Enable PDL for H100/B200 (SM_90 / SM_100)" ON)

add_executable(test_rmsnorm
    csrc/tests/test_rmsnorm.cu
)
target_link_libraries(test_rmsnorm PRIVATE mhc_kernels)
target_compile_options(test_rmsnorm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(MHC_ENABLE_PDL)
    target_compile_definitions(test_rmsnorm PRIVATE MHC_ENABLE_PDL)
endif()

add_executable(bench_rmsnorm
    csrc/benchmarks/bench_rmsnorm.cu
)
target_link_libraries(bench_rmsnorm PRIVATE mhc_kernels)
target_compile_options(bench_rmsnorm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(MHC_ENABLE_PDL)
    target_compile_definitions(bench_rmsnorm PRIVATE MHC_ENABLE_PDL)
endif()

add_executable(test_sinkhorn_knopp
    csrc/tests/test_sinkhorn_knopp.cu
)
target_link_libraries(test_sinkhorn_knopp PRIVATE mhc_kernels)
target_compile_options(test_sinkhorn_knopp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(MHC_ENABLE_PDL)
    target_compile_definitions(test_sinkhorn_knopp PRIVATE MHC_ENABLE_PDL)
endif()

add_executable(bench_sinkhorn_knopp
    csrc/benchmarks/bench_sinkhorn_knopp.cu
)
target_link_libraries(bench_sinkhorn_knopp PRIVATE mhc_kernels)
target_compile_options(bench_sinkhorn_knopp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(MHC_ENABLE_PDL)
    target_compile_definitions(bench_sinkhorn_knopp PRIVATE MHC_ENABLE_PDL)
endif()

add_executable(test_mhc_layer
    csrc/tests/test_mhc_layer.cu
)
target_link_libraries(test_mhc_layer PRIVATE mhc_kernels)
target_compile_options(test_mhc_layer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(MHC_ENABLE_PDL)
    target_compile_definitions(test_mhc_layer PRIVATE MHC_ENABLE_PDL)
endif()

add_executable(bench_mhc_layer
    csrc/benchmarks/bench_mhc_layer.cu
)
target_link_libraries(bench_mhc_layer PRIVATE mhc_kernels)
target_compile_options(bench_mhc_layer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(MHC_ENABLE_PDL)
    target_compile_definitions(bench_mhc_layer PRIVATE MHC_ENABLE_PDL)
endif()

add_executable(test_stream_mix_tc
    csrc/tests/test_stream_mix_tc.cu
)
target_link_libraries(test_stream_mix_tc PRIVATE mhc_kernels)
target_compile_options(test_stream_mix_tc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if(MHC_ENABLE_PDL)
    target_compile_definitions(test_stream_mix_tc PRIVATE MHC_ENABLE_PDL)
endif()
