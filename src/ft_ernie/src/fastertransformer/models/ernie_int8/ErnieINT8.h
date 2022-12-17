/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <unordered_map>
#include <vector>

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelSiluFfnLayer.h"
#include "src/fastertransformer/layers/FfnLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/FusedAttentionLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/UnfusedAttentionLayerINT8.h"
#include "src/fastertransformer/models/ernie_int8/ErnieINT8Weight.h"
#include "src/fastertransformer/models/ernie_int8/ErnieINT8LayerWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/kernels/quantization_int8_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/ernie_kernels.h"

namespace fastertransformer {

template<typename T>
class ErnieINT8Encoder: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_    = 0;

    // meta data
    const size_t           head_num_;
    const size_t           size_per_head_;
    const size_t           inter_size_;
    const size_t           hidden_units_;
    const size_t           d_model_;
    const size_t           num_layer_;
    const size_t           word_size_;
    const size_t           pos_size_;
    const size_t           sent_size_;
    int                    sm_;
    constexpr static float layernorm_eps_ = 1e-6f;
    float                  q_scaling_;
    AttentionType          attention_type_;
    int                    int8_mode_;
    bool                   sparse_;

    BaseAttentionLayer<T>* attention_layer_;
    FfnLayerINT8<T>*           ffn_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);
    void freeBuffer() override;
    void initialize();
    bool isValidLayerParallelId(uint l);
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();

    const ActivationType activation_type_;
    const LayerNormType  layernorm_type_;

    const NcclParam tensor_para_;
    const NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

protected:
    // model params
    size_t* token_num_               = nullptr;
    int*    padding_offset_          = nullptr;
    int*    trt_mha_padding_offset_  = nullptr;
    T*      attention_mask_          = nullptr;
    T*      relative_attention_bias_ = nullptr;
    T*      ernie_encoder_emb_buf_      = nullptr;
    T*      ernie_encoder_in_buffer_    = nullptr;
    int32_t*     attn_out_buf_            = nullptr;
    int8_t*      int8_buf_            = nullptr;
    
    T*      ernie_encoder_out_buffer_   = nullptr;

    T* normed_from_tensor_  = nullptr;
    T* normed_attn_out_buf_ = nullptr;

public:
    ErnieINT8Encoder(size_t                              max_batch_size,
                size_t                              max_seq_len,
                size_t                              head_num,
                size_t                              size_per_head,
                size_t                              inter_size,
                size_t                              d_model,
                size_t                              num_layer,
                size_t                              word_size,
                size_t                              pos_size,
                size_t                              sent_size,
                int                                 sm,
                float                               q_scaling,
                int                                 int8_mode,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                IAllocator*                         allocator,
                bool                                is_free_buffer_after_forward,
                AttentionType                       attention_type,
                bool                                sparse,
                ActivationType                      activation_type,
                LayerNormType                       layernorm_type,
                NcclParam                           tensor_para,
                NcclParam                           pipeline_para,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                int                                 enable_custom_all_reduce = 0);

    ErnieINT8Encoder(ErnieINT8Encoder<T> const& ernie_layer);

    ~ErnieINT8Encoder();

    void forward(std::vector<Tensor>*       output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const ErnieINT8EncoderWeight<T>*  ernie_weights);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const ErnieINT8EncoderWeight<T>*                      ernie_weights);

    inline size_t getDModel()
    {
        return d_model_;
    }
    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
