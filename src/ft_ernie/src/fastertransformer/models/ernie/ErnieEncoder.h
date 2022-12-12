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
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelSiluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/FusedAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelUnfusedAttentionLayer.h"
#include "src/fastertransformer/models/ernie/ErnieEncoderWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/models/ernie/FTCudaGraph.h"

namespace fastertransformer {

template<typename T>
class ErnieEncoder: public BaseLayer {
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
    size_t                 h_token_num_;
    int                    sm_;
    constexpr static float layernorm_eps_ = 1e-6f;
    float                  q_scaling_;
    AttentionType          attention_type_;
    bool                   sparse_;

    std::vector<BaseAttentionLayer<T>*> attention_layer_;
    std::vector<FfnLayer<T>*>           ffn_layer_;

    bool is_allocate_buffer_ = false;

    // for cuda graph
    FTCudaGraph* cur_graph_ptr_ = nullptr;
    bool is_enqueue_init_ = false;
    bool use_cuda_graph_ = true;
    std::unordered_map<std::string, FTCudaGraph*> cuda_graph_pool_;

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
    size_t* token_num_                  = nullptr;
    int*    padding_offset_             = nullptr;
    int*    trt_mha_padding_offset_     = nullptr;
    T*      attention_mask_             = nullptr;
    T*      relative_attention_bias_    = nullptr;
    T*      ernie_encoder_emb_buf_      = nullptr;
    T*      ernie_encoder_in_buffer_    = nullptr;
    T*      attn_out_buf_               = nullptr;
    T*      ernie_encoder_out_buffer_   = nullptr;
    T*      ernie_layer_out_buffer_     = nullptr;
    T*      ernie_slice_out_buffer_     = nullptr;
    T*      post_emb_out_buffer_        = nullptr;
    T*      fea_emb_fc_out_buffer_      = nullptr;
    T*      cls_out_buffer_      = nullptr;
    T*      cls_out_aside_buffer_      = nullptr;

    T* normed_from_tensor_  = nullptr;
    T* normed_attn_out_buf_ = nullptr;

public:
    ErnieEncoder(size_t                              max_batch_size,
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

    ErnieEncoder(ErnieEncoder<T> const& ernie_layer);

    ~ErnieEncoder();
    void forward(std::vector<Tensor>*       output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const ErnieEncoderWeight<T>*  ernie_weights);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const ErnieEncoderWeight<T>*                      ernie_weights);

    inline size_t getDModel()
    {
        return d_model_;
    }
    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
