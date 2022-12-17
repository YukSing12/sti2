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

add_library(ErnieINT8 STATIC ErnieINT8.cc ErnieINT8Weight.cc ErnieINT8LayerWeight.cc)
set_property(TARGET ErnieINT8 PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ErnieINT8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ErnieINT8 PUBLIC -lcudart -lcublas
bert_preprocess_kernels 
cublasMMWrapper 
cublasINT8MMWrapper 
UnfusedAttentionLayerINT8 
FusedAttentionLayerINT8 
FfnLayerINT8
layernorm_int8_kernels 
add_residual_kernels 
decoding_kernels 
quantization_int8_kernels 
dequantize_kernels 
ernie_kernels 
nccl_utils 
tensor)
