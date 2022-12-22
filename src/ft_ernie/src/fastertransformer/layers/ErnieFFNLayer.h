/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

template<typename T>
class ErnieFFNLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;

    // int8_mode_ == 1 for weight quantized only gemm for GPT
    int int8_mode_ = 0;

    // calculated data
    size_t hidden_units_;

    // gated activation
    bool use_gated_activation_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidTokenNum(size_t token_num);
    void allocateBuffer(size_t token_num);
    void initialize();

protected:
    T* inter_buf_ = nullptr;
    T* inter_buf_2_ = nullptr;  // for gated activation
    // the inter size for runtime ffn layer
    size_t inter_size_;
    /* used to allocater memory buffers
       different ffn layers (inter_size) will
       reuse the same ffn layer with the max inter size.
       max_inter_size will be passed as inter_size when initializing the ffn layer
    */
    size_t max_inter_size_;
    void invokeAddBiasActivation(const int m, const T* bias);
    void invokeAddBiasGatedActivation(const int m, const T* bias1, const T* bias2);

public:
    ErnieFFNLayer(size_t max_batch_size,
                  size_t max_seq_len,
                  size_t head_num,
                  size_t size_per_head,
                  size_t inter_size,
                  cudaStream_t stream,
                  cublasMMWrapper* cublas_wrapper,
                  IAllocator* allocator,
                  bool is_free_buffer_after_forward,
                  bool sparse = false,
                  int int8_mode = 0,
                  bool use_gated_activation = false);

    ErnieFFNLayer(ErnieFFNLayer<T> const& ffn_layer);

    ~ErnieFFNLayer();

    void resetInterSize(size_t runtime_inter_size)
    {
        inter_size_ = runtime_inter_size;
    }

    void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const FfnWeight<T>* ffn_weights);
};

}  // namespace fastertransformer
