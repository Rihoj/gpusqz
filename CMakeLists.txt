cmake_minimum_required(VERSION 3.20)

# Must be set before project()/enable_language(CUDA): "native" arch
# detection is unreliable under WSL (falls back silently to a very old
# default), so target the known GPU (RTX 5060 Ti / Blackwell, sm_120)
# explicitly. Override with -DCMAKE_CUDA_ARCHITECTURES=... for other
# hardware.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(gpu_zip LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(gzp
  src/main.cu
  src/kernels.cu
)

set_target_properties(gzp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# main.cu's output writer thread.
find_package(Threads REQUIRED)
target_link_libraries(gzp PRIVATE Threads::Threads)

target_compile_options(gzp PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Occupancy A/B testing. Both kernels carry an explicit __launch_bounds__,
# which takes precedence over a bare --maxrregcount, so that flag alone
# cannot force fewer registers here; GZP_MIN_BLOCKS_PER_SM instead sets
# __launch_bounds__'s second (minBlocksPerSM) argument, which does force
# ptxas to cut register usage until that many blocks fit (spilling to
# local memory if it can't). Compare with e.g.:
#   cmake -S . -B build-b2 -DGZP_MIN_BLOCKS_PER_SM=2 && cmake --build build-b2
#   bash bench/run_bench.sh <file>   # vs. a build with this unset
# and check `nvcc -Xptxas -v` for registers/spills alongside GZP_VERBOSE
# kernel MB/s. None of the per-chunk kernels use shared memory any more
# (the match-finding table moved to global memory), and at 38 (parse), 64
# (encode) and 58 (decode) registers per thread with no spills, their
# one-warp blocks are limited by the per-SM block count rather than by
# registers, so this knob is mainly a regression check.
set(GZP_MIN_BLOCKS_PER_SM "" CACHE STRING "minBlocksPerSM for __launch_bounds__ on gzp's kernels (empty = compiler default)")
if(GZP_MIN_BLOCKS_PER_SM)
  target_compile_definitions(gzp PRIVATE GZP_MIN_BLOCKS_PER_SM=${GZP_MIN_BLOCKS_PER_SM})
endif()

# CPU reference decoder used by tests/round_trip.sh.
add_executable(gzp_refdec tests/ref_decode.cpp)
target_include_directories(gzp_refdec PRIVATE src)
target_compile_options(gzp_refdec PRIVATE -O2 -Wall -Wextra)
