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

target_compile_options(gzp PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
