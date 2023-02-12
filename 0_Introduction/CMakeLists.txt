# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/0_Introduction/)

add_subdirectory(asyncAPI)
add_subdirectory(clock)
add_subdirectory(concurrentKernels)
add_subdirectory(cudaOpenMP)