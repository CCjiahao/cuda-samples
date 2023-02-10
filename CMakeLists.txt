cmake_minimum_required(VERSION 3.8)

project(cuda-samples LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

include_directories(/usr/local/cuda/include)
include_directories(./common/include)