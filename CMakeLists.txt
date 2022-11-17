cmake_minimum_required(VERSION 3.14)

project(sti2 LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

#add_compile_options(-Xcompiler -fPIC)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS};-std=c++14)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -fPIC)
#
include_directories(include)
#include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(/target/TensorRT-8.2.5.1/include)

link_directories(${CUDA_LIBRARIES})
#link_directories(/target/TensorRT-8.2.5.1/lib)

add_subdirectory(plugins)
