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

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/ernie_kernels.h"
#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/kernels/quantization_int8_kernels.h"
#include "src/fastertransformer/layers/FfnLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/FusedAttentionLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/UnfusedAttentionLayerINT8.h"
#include "src/fastertransformer/models/ernie/CudaGraph.h"
#include "src/fastertransformer/models/ernie_int8/ErnieINT8LayerWeight.h"
#include "src/fastertransformer/models/ernie_int8/ErnieINT8Weight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

namespace fastertransformer {

template<typename T>
class ErnieINT8: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;

    // meta data
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t hidden_units_;
    const size_t d_model_;
    const size_t num_layer_;
    const size_t word_size_;
    const size_t pos_size_;
    const size_t sent_size_;
    size_t h_token_num_;
    size_t request_batch_size_;
    size_t request_seq_len_;
    int sm_;
    constexpr static float layernorm_eps_ = 1e-6f;
    float q_scaling_;
    AttentionType attention_type_;
    int int8_mode_;
    bool sparse_;
    bool is_host_ptr_ = false;

    // five inputs
    int* d_word_ids_;
    int* d_pos_ids_;
    int* d_sent_ids_;
    int* d_seq_len_;
    int* d_multi_ids_;
    // one output
    float* d_attn_out_;

    // feature_stream
    cudaStream_t stream_fea_;
    std::mutex* cublas_wrapper_mutex_fea_;
    cublasAlgoMap* cublas_algo_map_fea_;
    cublasMMWrapper* cublas_wrapper_fea_;
    Allocator<AllocatorType::CUDA>* allocator_fea_;
    cublasHandle_t cublas_handle_fea_;
    cublasLtHandle_t cublaslt_handle_fea_;

    // BaseAttentionLayer<T>* attention_layer_;
    // FfnLayerINT8<T>*           ffn_layer_;
    BaseAttentionLayer<T>* attention_layer_;
    FfnLayerINT8<T>* ffn_layer_;

    bool is_allocate_buffer_ = false;

    CudaGraph* cur_graph_ptr_ = nullptr;
    bool is_enqueue_init_ = false;
    bool use_cuda_graph_ = true;
    std::unordered_map<std::string, CudaGraph*> cuda_graph_pool_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len);
    void freeBuffer() override;
    void initialize();

    const LayerNormType layernorm_type_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int enable_custom_all_reduce_;

protected:
    // model params
    size_t* token_num_ = nullptr;
    int* padding_offset_ = nullptr;
    int* trt_mha_padding_offset_ = nullptr;
    T* attention_mask_ = nullptr;
    T* relative_attention_bias_ = nullptr;
    T* ernie_encoder_emb_buf_ = nullptr;
    T* ernie_encoder_in_buffer_ = nullptr;
    int32_t* attn_out_buf_ = nullptr;
    int8_t* int8_buf_ = nullptr;

    T* ernie_encoder_out_buffer_ = nullptr;
    T* ernie_layer_out_buffer_ = nullptr;
    T* ernie_slice_out_buffer_ = nullptr;
    T* post_emb_out_buffer_ = nullptr;
    T* fea_emb_fc_out_buffer_ = nullptr;
    T* cls_out_buffer_ = nullptr;
    T* cls_out_aside_buffer_ = nullptr;

    T* normed_from_tensor_ = nullptr;
    T* normed_attn_out_buf_ = nullptr;

public:
    ErnieINT8(size_t max_batch_size,
              size_t max_seq_len,
              size_t head_num,
              size_t size_per_head,
              size_t inter_size,
              size_t d_model,
              size_t num_layer,
              size_t word_size,
              size_t pos_size,
              size_t sent_size,
              int sm,
              float q_scaling,
              int int8_mode,
              cudaStream_t stream,
              cublasINT8MMWrapper* cublas_wrapper,
              IAllocator* allocator,
              bool is_free_buffer_after_forward,
              AttentionType attention_type,
              bool sparse,
              LayerNormType layernorm_type,
              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm = nullptr,
              int enable_custom_all_reduce = 0);

    ErnieINT8(ErnieINT8<T> const& ernie_layer);

    ~ErnieINT8();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const ErnieINT8Weight<T>* ernie_weights);

    void forward(std::unordered_map<std::string, Tensor>* output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const ErnieINT8Weight<T>* ernie_weights);

    void forward(const int* h_word_ids_,
                 const int* h_pos_ids_,
                 const int* h_sent_ids_,
                 const int* h_seq_len_,
                 const int* h_multi_ids_,
                 const int request_batch_size,
                 const int request_seq_len,
                 const ErnieINT8Weight<T>* ernie_encoder_weights);
    void copyToCpu(float* h_attn_out_, const int request_batch_size);
    inline size_t getDModel()
    {
        return d_model_;
    }
    void setUseGraph(bool useGraph)
    {
        use_cuda_graph_ = useGraph;
    }
    void setHostMode(bool is_host_ptr)
    {
        is_host_ptr_ = is_host_ptr;
        d_word_ids_ = (int*)allocator_->reMalloc(d_word_ids_, sizeof(int) * max_batch_size_ * max_seq_len_, false);
        d_pos_ids_ = (int*)allocator_->reMalloc(d_pos_ids_, sizeof(int) * max_batch_size_ * max_seq_len_, false);
        d_sent_ids_ = (int*)allocator_->reMalloc(d_sent_ids_, sizeof(int) * max_batch_size_ * max_seq_len_, false);
        d_seq_len_ = (int*)allocator_->reMalloc(d_seq_len_, sizeof(int) * max_batch_size_ * 1, false);
        d_multi_ids_ = (int*)allocator_->reMalloc(d_multi_ids_, sizeof(int) * max_batch_size_ * 8, false);
        d_attn_out_ = (float*)allocator_->reMalloc(d_attn_out_, sizeof(float) * max_batch_size_ * 1, false);
    }
    void setStream(cudaStream_t stream) override;
};

}  // namespace fastertransformer
