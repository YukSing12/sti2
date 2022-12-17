# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(Ernie STATIC Ernie.cc ErnieWeight.cc ErnieLayerWeight.cc)
set_property(TARGET Ernie PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Ernie PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Ernie PUBLIC -lcudart -lcublas
bert_preprocess_kernels 
cublasMMWrapper 
TensorParallelUnfusedAttentionLayer 
FusedAttentionLayer 
TensorParallelReluFfnLayer
TensorParallelGeluFfnLayer 
TensorParallelSiluFfnLayer 
layernorm_kernels 
add_residual_kernels 
decoding_kernels
ernie_kernels
nccl_utils tensor)

add_executable(ernie_gemm ernie_gemm.cc)
target_link_libraries(ernie_gemm PUBLIC -lcudart ernie_gemm_func memory_utils)
