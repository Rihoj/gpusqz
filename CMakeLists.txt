cmake_minimum_required(VERSION 3.20)

# Backends (see src/backend.h). gpusqz is built with whichever are on:
#   GPUSQZ_BUILD_CUDA    NVIDIA GPUs; needs the CUDA toolkit.
#   GPUSQZ_BUILD_VULKAN  AMD, Apple (through MoltenVK), Intel and NVIDIA
#                        GPUs; needs the Vulkan headers and glslangValidator
#                        (Vulkan SDK, or distro packages) to build. The
#                        Vulkan library itself is loaded at run time.
# With both off, only the CPU reference decoder (gpusqz_refdec) is built.
if(APPLE)
  set(GPUSQZ_CUDA_DEFAULT OFF) # CUDA does not exist on macOS
else()
  set(GPUSQZ_CUDA_DEFAULT ON)
endif()
option(GPUSQZ_BUILD_CUDA "Build the CUDA backend (NVIDIA; needs the CUDA toolkit)" ${GPUSQZ_CUDA_DEFAULT})
option(GPUSQZ_BUILD_VULKAN "Build the Vulkan backend (needs Vulkan headers and glslangValidator)" ON)

# Must be set before enable_language(CUDA): "native" arch detection is
# unreliable under WSL (falls back silently to a very old default), so
# target the known GPU (RTX 5060 Ti / Blackwell, sm_120) explicitly.
# Release builds pass a list covering every supported generation (see
# .github/workflows/build.yml); gpusqz needs sm_70 or newer (__match_any_sync).
if(GPUSQZ_BUILD_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(GPUSQZ_VERSION "0.5.0" CACHE STRING "Version stamped into packages (CI sets it from the git tag)")
project(gpusqz VERSION ${GPUSQZ_VERSION} DESCRIPTION "GPU file compressor" LANGUAGES CXX)
if(GPUSQZ_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(WIN32)
  # std::getenv/fopen are fine here; silence MSVC's deprecation warnings.
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  # 64-bit off_t for fseeko/ftello on 32-bit platforms (see src/file_io.h).
  add_compile_definitions(_FILE_OFFSET_BITS=64)
endif()

if(GPUSQZ_BUILD_CUDA OR GPUSQZ_BUILD_VULKAN)
  set(GPUSQZ_BUILD_GPU ON)
  add_executable(gpusqz src/main.cpp)
  # main.cpp's output writer thread.
  find_package(Threads REQUIRED)
  target_link_libraries(gpusqz PRIVATE Threads::Threads)
  if(MSVC)
    target_compile_options(gpusqz PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
  else()
    target_compile_options(gpusqz PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
  endif()
else()
  set(GPUSQZ_BUILD_GPU OFF)
endif()

if(GPUSQZ_BUILD_CUDA)
  target_sources(gpusqz PRIVATE src/cuda_backend.cu src/kernels.cu)
  target_compile_definitions(gpusqz PRIVATE GPUSQZ_HAVE_CUDA)
  target_link_libraries(gpusqz PRIVATE CUDA::cudart_static)

  # All kernels live in kernels.cu, so no relocatable device code is needed;
  # with it, the device link step dropped the PTX that lets newer GPUs run.
  set_target_properties(gpusqz PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
  )

  target_compile_options(gpusqz PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )

  # Occupancy experiments: sets __launch_bounds__'s minBlocksPerSM on the
  # per-chunk kernels, forcing ptxas to cut registers until that many blocks
  # fit (a bare --maxrregcount is overridden by __launch_bounds__). Check
  # `nvcc -Xptxas -v` for spills alongside GPUSQZ_VERBOSE kernel MB/s.
  set(GPUSQZ_MIN_BLOCKS_PER_SM "" CACHE STRING "minBlocksPerSM for __launch_bounds__ on gpusqz's kernels (empty = compiler default)")
  if(GPUSQZ_MIN_BLOCKS_PER_SM)
    target_compile_definitions(gpusqz PRIVATE GPUSQZ_MIN_BLOCKS_PER_SM=${GPUSQZ_MIN_BLOCKS_PER_SM})
  endif()
endif()

if(GPUSQZ_BUILD_VULKAN)
  find_path(GPUSQZ_VULKAN_INCLUDE vulkan/vulkan.h HINTS $ENV{VULKAN_SDK}/include)
  find_program(GPUSQZ_GLSLANG glslangValidator HINTS $ENV{VULKAN_SDK}/bin)
  if(NOT GPUSQZ_VULKAN_INCLUDE OR NOT GPUSQZ_GLSLANG)
    message(FATAL_ERROR "The Vulkan backend needs the Vulkan headers and glslangValidator (install the Vulkan SDK, "
                        "or e.g. libvulkan-dev + glslang-tools), or configure with -DGPUSQZ_BUILD_VULKAN=OFF")
  endif()

  # Every compute shader becomes an embedded SPIR-V array spv_<name>_<lg>;
  # lg is common.glsl's LG_SUBGROUP. Only the shaders using lane-group
  # operations need the subgroup build.
  set(GPUSQZ_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/vk)
  set(GPUSQZ_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
  file(MAKE_DIRECTORY ${GPUSQZ_SPV_DIR})
  set(GPUSQZ_SHADER_INCLUDES ${GPUSQZ_SHADER_DIR}/common.glsl ${GPUSQZ_SHADER_DIR}/compress.glsl
                             ${GPUSQZ_SHADER_DIR}/decompress.glsl)
  set(GPUSQZ_SPV_HEADERS)
  foreach(spec parse_hist:0 parse_hist:1 rans_encode:0 rans_encode:1 decompress:0 decompress:1 probe:0 probe:1
               build_table:0 scan:0 compact:0 expand_tables:0)
    string(REPLACE ":" ";" parts ${spec})
    list(GET parts 0 name)
    list(GET parts 1 lg)
    set(out ${GPUSQZ_SPV_DIR}/spv_${name}_${lg}.h)
    add_custom_command(
      OUTPUT ${out}
      COMMAND ${GPUSQZ_GLSLANG} -V --target-env vulkan1.2 -DLG_SUBGROUP=${lg} --vn spv_${name}_${lg}
              -o ${out} ${GPUSQZ_SHADER_DIR}/${name}.comp
      DEPENDS ${GPUSQZ_SHADER_DIR}/${name}.comp ${GPUSQZ_SHADER_INCLUDES}
      COMMENT "Compiling ${name}.comp (LG_SUBGROUP=${lg})"
      VERBATIM)
    list(APPEND GPUSQZ_SPV_HEADERS ${out})
  endforeach()
  add_custom_target(gpusqz_shaders DEPENDS ${GPUSQZ_SPV_HEADERS})

  target_sources(gpusqz PRIVATE src/vk_backend.cpp)
  # Vulkan structs are initialised as {sType} with the rest zeroed.
  if(NOT MSVC)
    set_source_files_properties(src/vk_backend.cpp PROPERTIES COMPILE_OPTIONS -Wno-missing-field-initializers)
  endif()
  add_dependencies(gpusqz gpusqz_shaders)
  target_include_directories(gpusqz PRIVATE ${GPUSQZ_VULKAN_INCLUDE} ${GPUSQZ_SPV_DIR})
  target_compile_definitions(gpusqz PRIVATE GPUSQZ_HAVE_VULKAN)
  target_link_libraries(gpusqz PRIVATE ${CMAKE_DL_LIBS})
endif()

# CPU reference decoder: used by the tests, and the only tool built when
# both backends are off.
add_executable(gpusqz_refdec tests/ref_decode.cpp)
target_include_directories(gpusqz_refdec PRIVATE src)
if(MSVC)
  target_compile_options(gpusqz_refdec PRIVATE /W4)
else()
  target_compile_options(gpusqz_refdec PRIVATE -O2 -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Tests (ctest). Labelled "gpu" if they need an NVIDIA GPU; CI runners have
# none, so they run `ctest -LE gpu`.
# ---------------------------------------------------------------------------
enable_testing()

# Every tests/fixtures/<name>__<variant>.gsz must decode, with the CPU
# reference decoder, to tests/fixtures/<name>.orig. The fixtures were
# made by the GPU compressor (tests/fixtures/make_fixtures.sh), so this
# checks the file format and the decoder on every platform, GPU or not.
file(GLOB GPUSQZ_FIXTURES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/*.gsz)
foreach(fixture ${GPUSQZ_FIXTURES})
  get_filename_component(fname ${fixture} NAME_WE)
  string(REGEX REPLACE "__.*$" "" orig_name ${fname})
  add_test(NAME refdec_${fname}
           COMMAND ${CMAKE_COMMAND}
                   -DREFDEC=$<TARGET_FILE:gpusqz_refdec>
                   -DINPUT=${fixture}
                   -DEXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/${orig_name}.orig
                   -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/fixture_${fname}.out
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/refdec_check.cmake)
endforeach()
# ...and corrupt ones (*.gszbad) must be rejected.
file(GLOB GPUSQZ_BAD_FIXTURES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/*.gszbad)
foreach(fixture ${GPUSQZ_BAD_FIXTURES})
  get_filename_component(fname ${fixture} NAME_WE)
  add_test(NAME refdec_rejects_${fname}
           COMMAND gpusqz_refdec ${fixture} ${CMAKE_CURRENT_BINARY_DIR}/fixture_${fname}.bad.out)
  set_tests_properties(refdec_rejects_${fname} PROPERTIES WILL_FAIL TRUE)
endforeach()

if(GPUSQZ_BUILD_GPU)
  # Runs without a GPU: argument errors are reported before any CUDA call.
  add_test(NAME gpusqz_usage COMMAND gpusqz)
  set_tests_properties(gpusqz_usage PROPERTIES PASS_REGULAR_EXPRESSION "usage:")

  # The full GPU round-trip suite (needs an NVIDIA GPU and bash).
  find_program(BASH_PROGRAM bash)
  if(BASH_PROGRAM AND NOT WIN32)
    add_test(NAME round_trip
             COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/tests/round_trip.sh $<TARGET_FILE:gpusqz>)
    set_tests_properties(round_trip PROPERTIES LABELS gpu)
  endif()
endif()

# ---------------------------------------------------------------------------
# Install and packages (cpack -G DEB | RPM | ZIP | TGZ)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
if(GPUSQZ_BUILD_GPU)
  install(TARGETS gpusqz RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS gpusqz_refdec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/gpusqz)
if(MSVC)
  # Ship the MSVC runtime DLLs next to the executables so the zip runs on
  # machines without the Visual C++ redistributable.
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${CMAKE_INSTALL_BINDIR})
  include(InstallRequiredSystemLibraries)
endif()

set(GPUSQZ_PACKAGE_MAINTAINER "gpusqz maintainers" CACHE STRING "Maintainer field for the deb/rpm packages")
if(GPUSQZ_BUILD_GPU)
  set(CPACK_PACKAGE_NAME "gpusqz")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU (CUDA) file compressor")
else()
  set(CPACK_PACKAGE_NAME "gpusqz-refdec")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPU decoder for files made by the gpusqz GPU compressor")
endif()
set(CPACK_PACKAGE_VENDOR "gpusqz")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT ${GPUSQZ_PACKAGE_MAINTAINER})
set(CPACK_PACKAGE_DESCRIPTION
    "gpusqz compresses and decompresses files on an NVIDIA GPU with CUDA: a warp-parallel LZ parse and a 32-way interleaved rANS coder. It needs an NVIDIA GPU of compute capability 7.0 or newer and an NVIDIA driver that supports CUDA 12 (the CUDA runtime is linked in). gpusqz_refdec is a CPU-only decoder for the same format.")
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_STRIP_FILES ON)

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${GPUSQZ_PACKAGE_MAINTAINER})
set(CPACK_DEBIAN_PACKAGE_SECTION utils)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_GROUP "Applications/Archiving")
set(CPACK_RPM_PACKAGE_LICENSE "Unspecified")

include(CPack)
