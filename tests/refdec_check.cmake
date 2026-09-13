# ctest helper: decode INPUT with the CPU reference decoder REFDEC into
# OUTPUT and require it to match EXPECTED byte for byte. Portable across
# Linux, macOS and Windows (no shell, cmp or diff needed).
execute_process(COMMAND ${REFDEC} ${INPUT} ${OUTPUT} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "gzp_refdec failed on ${INPUT} (exit ${rc})")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files ${OUTPUT} ${EXPECTED} RESULT_VARIABLE diff)
if(NOT diff EQUAL 0)
  message(FATAL_ERROR "decoding ${INPUT} did not reproduce ${EXPECTED}")
endif()
file(REMOVE ${OUTPUT})
