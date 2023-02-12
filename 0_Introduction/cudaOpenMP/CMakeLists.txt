find_package(OpenMP REQUIRED)

add_compile_options(-Xcompiler -fopenmp)

add_executable(cudaOpenMP cudaOpenMP.cu)
set_target_properties(cudaOpenMP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cudaOpenMP PUBLIC OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "openmp not found!")
endif()