# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
message("-- ERNIE_BIN_DIR: ${PROJECT_SOURCE_DIR}/../../model/bin")
set(ERNIE_BIN_DIR "\"${PROJECT_SOURCE_DIR}/../../model/bin\"") 
configure_file("${PROJECT_SOURCE_DIR}/src/fastertransformer/tensorrt_plugin/ernie/ErnieBinPath.h.in"  "${PROJECT_SOURCE_DIR}/src/fastertransformer/tensorrt_plugin/ernie/ErnieBinPath.h")
add_library(ErniePlugin SHARED ErniePluginGemm.cc ErniePlugin.cu)
target_link_libraries(ErniePlugin PRIVATE 
Ernie 
ernie_gemm_func 
-lcublas 
-lcublasLt 
-lcudart 
-lnvinfer)

install(TARGETS ErniePlugin LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/so/plugins)