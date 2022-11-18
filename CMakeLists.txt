cmake_minimum_required(VERSION 3.14)

project(sti2 LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -ldl")

set(SM_SETS 52 60 61 70 75 80 86)
foreach(SM_NUM IN LISTS SM_SETS)
  string(FIND "${SM}" "${SM_NUM}" SM_POS)
  if(SM_POS GREATER -1)
    if(FIND_SM STREQUAL False)
      set(ENV{TORCH_CUDA_ARCH_LIST} "")
    endif()
    set(FIND_SM True)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM_NUM},code=\\\"sm_${SM_NUM},compute_${SM_NUM}\\\"")

    if (SM_NUM STREQUAL 70 OR SM_NUM STREQUAL 75 OR SM_NUM STREQUAL 80 OR SM_NUM STREQUAL 86)
      set(USING_WMMA True)
    endif()

    if(BUILD_PYT)
      string(SUBSTRING ${SM_NUM} 0 1 SM_MAJOR)
      string(SUBSTRING ${SM_NUM} 1 1 SM_MINOR)
      set(ENV{TORCH_CUDA_ARCH_LIST} "$ENV{TORCH_CUDA_ARCH_LIST}\;${SM_MAJOR}.${SM_MINOR}")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message("-- Assign GPU architecture (sm=${SM_NUM})")
  endif()
endforeach()

if(NOT (FIND_SM STREQUAL True))
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
  #                      -rdc=true")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  if(BUILD_PYT)
    set(ENV{TORCH_CUDA_ARCH_LIST} "7.0;7.5;8.0;8.6")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  message("-- Assign GPU architecture (sm=70,75,80,86)")
endif()

if(BUILD_PYT)
  set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
endif()

include_directories(include)
#include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(/target/TensorRT-8.2.5.1/include)

link_directories(${CUDA_LIBRARIES})
#link_directories(/target/TensorRT-8.2.5.1/lib)

add_subdirectory(plugins)
