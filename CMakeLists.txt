cmake_minimum_required(VERSION 3.20)

# GZP_BUILD_GPU=OFF builds only the CPU reference decoder (gzp_refdec),
# which needs no CUDA toolkit. That is what the macOS package contains:
# CUDA does not exist on macOS, so the compressor itself cannot run there,
# but .gzp files made elsewhere can still be decompressed.
option(GZP_BUILD_GPU "Build the CUDA compressor gzp (needs the CUDA toolkit)" ON)

# Must be set before enable_language(CUDA): "native" arch detection is
# unreliable under WSL (falls back silently to a very old default), so
# target the known GPU (RTX 5060 Ti / Blackwell, sm_120) explicitly.
# Release builds pass a list covering every supported generation (see
# .github/workflows/build.yml); gzp needs sm_70 or newer (__match_any_sync).
if(GZP_BUILD_GPU AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(GZP_VERSION "0.5.0" CACHE STRING "Version stamped into packages (CI sets it from the git tag)")
project(gpu_zip VERSION ${GZP_VERSION} DESCRIPTION "GPU (CUDA) file compressor" LANGUAGES CXX)
if(GZP_BUILD_GPU)
  enable_language(CUDA)
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

if(GZP_BUILD_GPU)
  add_executable(gzp
    src/main.cu
    src/kernels.cu
  )

  set_target_properties(gzp PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
  )

  # main.cu's output writer thread.
  find_package(Threads REQUIRED)
  target_link_libraries(gzp PRIVATE Threads::Threads)

  target_compile_options(gzp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )

  # Occupancy A/B testing. The per-chunk kernels carry an explicit
  # __launch_bounds__, which takes precedence over a bare --maxrregcount,
  # so that flag alone cannot force fewer registers here;
  # GZP_MIN_BLOCKS_PER_SM instead sets __launch_bounds__'s second
  # (minBlocksPerSM) argument, which does force ptxas to cut register usage
  # until that many blocks fit (spilling to local memory if it can't).
  # Compare with e.g.:
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
endif()

# CPU reference decoder: used by the tests, and the only tool in the macOS
# package (see GZP_BUILD_GPU above).
add_executable(gzp_refdec tests/ref_decode.cpp)
target_include_directories(gzp_refdec PRIVATE src)
if(MSVC)
  target_compile_options(gzp_refdec PRIVATE /W4)
else()
  target_compile_options(gzp_refdec PRIVATE -O2 -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Tests (ctest). Labelled "gpu" if they need an NVIDIA GPU; CI runners have
# none, so they run `ctest -LE gpu`.
# ---------------------------------------------------------------------------
enable_testing()

# Every tests/fixtures/<name>__<variant>.gzp must decode, with the CPU
# reference decoder, to tests/fixtures/<name>.orig. The fixtures were
# made by the GPU compressor (tests/fixtures/make_fixtures.sh), so this
# checks the file format and the decoder on every platform, GPU or not.
file(GLOB GZP_FIXTURES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/*.gzp)
foreach(fixture ${GZP_FIXTURES})
  get_filename_component(fname ${fixture} NAME_WE)
  string(REGEX REPLACE "__.*$" "" orig_name ${fname})
  add_test(NAME refdec_${fname}
           COMMAND ${CMAKE_COMMAND}
                   -DREFDEC=$<TARGET_FILE:gzp_refdec>
                   -DINPUT=${fixture}
                   -DEXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/${orig_name}.orig
                   -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/fixture_${fname}.out
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/refdec_check.cmake)
endforeach()
# ...and corrupt ones (*.gzpbad) must be rejected.
file(GLOB GZP_BAD_FIXTURES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/*.gzpbad)
foreach(fixture ${GZP_BAD_FIXTURES})
  get_filename_component(fname ${fixture} NAME_WE)
  add_test(NAME refdec_rejects_${fname}
           COMMAND gzp_refdec ${fixture} ${CMAKE_CURRENT_BINARY_DIR}/fixture_${fname}.bad.out)
  set_tests_properties(refdec_rejects_${fname} PROPERTIES WILL_FAIL TRUE)
endforeach()

if(GZP_BUILD_GPU)
  # Runs without a GPU: argument errors are reported before any CUDA call.
  add_test(NAME gzp_usage COMMAND gzp)
  set_tests_properties(gzp_usage PROPERTIES PASS_REGULAR_EXPRESSION "usage:")

  # The full GPU round-trip suite (needs an NVIDIA GPU and bash).
  find_program(BASH_PROGRAM bash)
  if(BASH_PROGRAM AND NOT WIN32)
    add_test(NAME round_trip
             COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/tests/round_trip.sh $<TARGET_FILE:gzp>)
    set_tests_properties(round_trip PROPERTIES LABELS gpu)
  endif()
endif()

# ---------------------------------------------------------------------------
# Install and packages (cpack -G DEB | RPM | ZIP | TGZ)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
if(GZP_BUILD_GPU)
  install(TARGETS gzp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS gzp_refdec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/gzp)
if(MSVC)
  # Ship the MSVC runtime DLLs next to the executables so the zip runs on
  # machines without the Visual C++ redistributable.
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${CMAKE_INSTALL_BINDIR})
  include(InstallRequiredSystemLibraries)
endif()

set(GZP_PACKAGE_MAINTAINER "gzp maintainers" CACHE STRING "Maintainer field for the deb/rpm packages")
if(GZP_BUILD_GPU)
  set(CPACK_PACKAGE_NAME "gzp")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU (CUDA) file compressor")
else()
  set(CPACK_PACKAGE_NAME "gzp-refdec")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPU decoder for files made by the gzp GPU compressor")
endif()
set(CPACK_PACKAGE_VENDOR "gzp")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT ${GZP_PACKAGE_MAINTAINER})
set(CPACK_PACKAGE_DESCRIPTION
    "gzp compresses and decompresses files on an NVIDIA GPU with CUDA: a warp-parallel LZ parse and a 32-way interleaved rANS coder. It needs an NVIDIA GPU of compute capability 7.0 or newer and an NVIDIA driver that supports CUDA 12 (the CUDA runtime is linked in). gzp_refdec is a CPU-only decoder for the same format.")
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_STRIP_FILES ON)

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${GZP_PACKAGE_MAINTAINER})
set(CPACK_DEBIAN_PACKAGE_SECTION utils)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_GROUP "Applications/Archiving")
set(CPACK_RPM_PACKAGE_LICENSE "Unspecified")

include(CPack)
