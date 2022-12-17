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

#include "src/fastertransformer/models/ernie_int8/ErnieINT8.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

#define POSTGRAPH_IDX 4
#define PREGRAPH_IDX 3

namespace fastertransformer {

template<typename T>
void ErnieINT8<T>::initialize()
{
    cudaStreamCreate(&stream_fea_);
    cublasCreate(&cublas_handle_fea_);
    cublasLtCreate(&cublaslt_handle_fea_);
    allocator_fea_ = new Allocator<AllocatorType::CUDA>(getDevice());
    cublas_wrapper_mutex_fea_ = new std::mutex();
    std::string gemmFileName = std::string("gemm_config.in").substr(0, 11) + std::string("-SM") + std::to_string(sm_)
                               + std::string("-FP") + std::to_string(std::is_same<T, half>::value ? 16 : 32)
                               + std::string("-BS") + std::to_string(max_batch_size_) + std::string("-SL")
                               + std::to_string(max_seq_len_) + std::string("-BM") + std::to_string(1)
                               + std::string(".in");

    cublas_algo_map_fea_ = new cublasAlgoMap(gemmFileName, "");

    cublas_wrapper_fea_ = new cublasMMWrapper(cublas_handle_fea_,
                                              cublaslt_handle_fea_,
                                              stream_fea_,
                                              cublas_algo_map_fea_,
                                              cublas_wrapper_mutex_fea_,
                                              allocator_fea_);

    if (std::is_same<T, half>::value) {
        cublas_wrapper_fea_->setFP16GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper_fea_->setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_fea_->setFP32GemmConfig();
    }

    if ((attention_type_ == AttentionType::FUSED_MHA || attention_type_ == AttentionType::FUSED_PADDED_MHA)
        && std::is_same<T, half>::value == true && max_seq_len_ <= 384) {
        attention_layer_ = new FusedAttentionLayerINT8<T>(max_batch_size_,
                                                          max_seq_len_,
                                                          head_num_,
                                                          size_per_head_,
                                                          sm_,
                                                          int8_mode_,
                                                          q_scaling_,  // adjust according to checkpoint structure
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          sparse_);
    }
    else if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::UNFUSED_PADDED_MHA) {
        attention_layer_ = new UnfusedAttentionLayerINT8<T>(max_batch_size_,
                                                            max_seq_len_,
                                                            head_num_,
                                                            size_per_head_,
                                                            q_scaling_,  // adjust according to checkpoint structure
                                                            int8_mode_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            sparse_);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
    }

    ffn_layer_ = new ReluFfnLayerINT8<T>(max_batch_size_,
                                         max_seq_len_,
                                         1,
                                         d_model_,
                                         inter_size_,
                                         int8_mode_,
                                         stream_,
                                         cublas_wrapper_,
                                         allocator_,
                                         is_free_buffer_after_forward_);
}

template<typename T>
ErnieINT8<T>::ErnieINT8(size_t max_batch_size,
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
                        std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                        int enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    word_size_(word_size),
    pos_size_(pos_size),
    sent_size_(sent_size),
    sm_(sm),
    q_scaling_(q_scaling),
    int8_mode_(int8_mode),
    attention_type_(attention_type),
    sparse_(sparse),
    layernorm_type_(layernorm_type),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
ErnieINT8<T>::ErnieINT8(ErnieINT8<T> const& ernie_encoder):
    BaseLayer(ernie_encoder),
    max_batch_size_(ernie_encoder.max_batch_size_),
    max_seq_len_(ernie_encoder.max_seq_len_),
    head_num_(ernie_encoder.head_num_),
    size_per_head_(ernie_encoder.size_per_head_),
    inter_size_(ernie_encoder.inter_size_),
    d_model_(ernie_encoder.d_model_),
    hidden_units_(ernie_encoder.hidden_units_),
    num_layer_(ernie_encoder.num_layer_),
    word_size_(ernie_encoder.word_size_),
    pos_size_(ernie_encoder.pos_size_),
    sent_size_(ernie_encoder.sent_size_),
    sm_(ernie_encoder.sm_),
    q_scaling_(ernie_encoder.q_scaling_),
    int8_mode_(ernie_encoder.int8_mode_),
    attention_type_(ernie_encoder.attention_type_),
    sparse_(ernie_encoder.sparse_),
    layernorm_type_(ernie_encoder.layernorm_type_),
    custom_all_reduce_comm_(ernie_encoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(ernie_encoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
ErnieINT8<T>::~ErnieINT8()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (auto& it : cuda_graph_pool_)
        delete it.second;
    cuda_graph_pool_.clear();
    delete attention_layer_;
    delete ffn_layer_;

    delete cublas_wrapper_mutex_fea_;
    delete cublas_algo_map_fea_;
    delete cublas_wrapper_fea_;
    delete allocator_fea_;
    freeBuffer();
}

template<typename T>
void ErnieINT8<T>::setStream(cudaStream_t stream)
{

    attention_layer_->setStream(stream);
    ffn_layer_->setStream(stream);

    BaseLayer::setStream(stream);
}

template<typename T>
void ErnieINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        allocateBuffer(max_batch_size_, max_seq_len_);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void ErnieINT8<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    token_num_ = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
    padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    trt_mha_padding_offset_ =
        (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * batch_size + 1), false);

    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false);
    relative_attention_bias_ =
        (T*)allocator_->reMalloc(relative_attention_bias_, sizeof(T) * head_num_ * seq_len * seq_len, false);

    ernie_encoder_emb_buf_ =
        (T*)allocator_->reMalloc(ernie_encoder_emb_buf_, sizeof(T) * batch_size * seq_len * d_model_, false);
    ernie_encoder_in_buffer_ =
        (T*)allocator_->reMalloc(ernie_encoder_in_buffer_, sizeof(T) * batch_size * seq_len * d_model_, false);
    attn_out_buf_ = (int32_t*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * d_model_, false);
    ernie_encoder_out_buffer_ =
        (T*)allocator_->reMalloc(ernie_encoder_out_buffer_, sizeof(T) * batch_size * seq_len * d_model_, false);
    int8_buf_ = reinterpret_cast<int8_t*>(
        allocator_->reMalloc(int8_buf_, sizeof(int8_t) * batch_size * seq_len * d_model_, false));
    ernie_layer_out_buffer_ =
        (T*)allocator_->reMalloc(ernie_layer_out_buffer_, sizeof(T) * batch_size * seq_len * d_model_, false);
    ernie_slice_out_buffer_ =
        (T*)allocator_->reMalloc(ernie_slice_out_buffer_, sizeof(T) * batch_size * 1 * d_model_, false);
    post_emb_out_buffer_ = (T*)allocator_->reMalloc(post_emb_out_buffer_, sizeof(T) * batch_size * d_model_, false);
    fea_emb_fc_out_buffer_ = (T*)allocator_->reMalloc(fea_emb_fc_out_buffer_, sizeof(T) * batch_size * d_model_, false);
    cls_out_buffer_ = (T*)allocator_->reMalloc(cls_out_buffer_, sizeof(T) * batch_size * 1, false);
    cls_out_aside_buffer_ = (T*)allocator_->reMalloc(cls_out_aside_buffer_, sizeof(T) * batch_size * 1, false);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_ = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        normed_from_tensor_ =
            (T*)allocator_->reMalloc(normed_from_tensor_, sizeof(T) * batch_size * seq_len * d_model_, false);
        normed_attn_out_buf_ =
            (T*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T) * batch_size * seq_len * d_model_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void ErnieINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        if (is_host_ptr_) {
            allocator_->free((void**)(&d_word_ids_));
            allocator_->free((void**)(&d_pos_ids_));
            allocator_->free((void**)(&d_sent_ids_));
            allocator_->free((void**)(&d_seq_len_));
            allocator_->free((void**)(&d_multi_ids_));
            allocator_->free((void**)(&d_attn_out_));
        }
        allocator_->free((void**)(&token_num_));
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&trt_mha_padding_offset_));

        allocator_->free((void**)(&attention_mask_));
        allocator_->free((void**)(&relative_attention_bias_));
        allocator_->free((void**)(&ernie_encoder_emb_buf_));
        allocator_->free((void**)(&ernie_encoder_in_buffer_));
        allocator_->free((void**)(&attn_out_buf_));
        allocator_->free((void**)(&ernie_encoder_out_buffer_));
        allocator_->free((void**)(&int8_buf_));
        allocator_->free((void**)(&ernie_layer_out_buffer_));
        allocator_->free((void**)(&ernie_slice_out_buffer_));
        allocator_->free((void**)(&post_emb_out_buffer_));
        allocator_->free((void**)(&fea_emb_fc_out_buffer_));
        allocator_->free((void**)(&cls_out_buffer_));
        allocator_->free((void**)(&cls_out_aside_buffer_));

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_ = nullptr;
            normed_attn_out_buf_ = nullptr;
        }
        else {
            allocator_->free((void**)(&normed_from_tensor_));
            allocator_->free((void**)(&normed_attn_out_buf_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void ErnieINT8<T>::copyToCpu(float* h_attn_out_, const int request_batch_size_)
{
    cudaAutoCpy(h_attn_out_, d_attn_out_, request_batch_size_, stream_);
}
template<typename T>
void ErnieINT8<T>::forward(std::vector<Tensor>* output_tensors,
                           const std::vector<Tensor>* input_tensors,
                           const ErnieINT8Weight<T>* ernie_int8_weights)
{
    // input_tensors:
    //      word_ids [batch, seqlen, 1]
    //      pos_ids  [batch, seqlen, 1]
    //      sent_ids [batch, seqlen, 1]
    //      seq_len     [batch, 1, 1]
    // output tensors:
    //      attn_out [batch, seqlen, d_model]

    std::unordered_map<std::string, Tensor> input_tensors_map{{"word_ids", input_tensors->at(0)},
                                                              {"pos_ids", input_tensors->at(1)},
                                                              {"sent_ids", input_tensors->at(2)},
                                                              {"seq_len", input_tensors->at(3)}};

    std::unordered_map<std::string, Tensor> output_tensors_map{{"attn_out", output_tensors->at(0)}};
    forward(&output_tensors_map, &input_tensors_map, ernie_int8_weights);
}

template<typename T>
void ErnieINT8<T>::forward(std::unordered_map<std::string, Tensor>* output_tensors,
                           const std::unordered_map<std::string, Tensor>* input_tensors,
                           const ErnieINT8Weight<T>* ernie_int8_weights)
{
    // input_tensors:
    //      word_ids [batch, seqlen, 1]
    //      pos_ids  [batch, seqlen, 1]
    //      sent_ids [batch, seqlen, 1]
    //      seq_len     [batch, 1, 1]
    // output tensors:
    //      attn_out [batch, seqlen, d_model]

    const ErnieINT8LayerWeight<T>* ernie_layer_int8_weight = (const ErnieINT8LayerWeight<T>*)ernie_int8_weights;
    const ScaleList* scale_list = &(ernie_layer_int8_weight->scale_list_);

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(is_host_ptr_ == false);

    FT_CHECK(input_tensors->at("word_ids").shape.size() == 2);
    request_batch_size_ = input_tensors->at("word_ids").shape[0];
    request_seq_len_ = input_tensors->at("word_ids").shape[1];
    FT_CHECK(input_tensors->size() == 5);
    FT_CHECK(request_batch_size_ == input_tensors->at("pos_ids").shape[0]);
    FT_CHECK(request_seq_len_ == input_tensors->at("pos_ids").shape[1]);
    FT_CHECK(input_tensors->at("pos_ids").shape.size() == 2);

    FT_CHECK(request_batch_size_ == input_tensors->at("sent_ids").shape[0]);
    FT_CHECK(request_seq_len_ == input_tensors->at("sent_ids").shape[1]);
    FT_CHECK(input_tensors->at("sent_ids").shape.size() == 2);

    FT_CHECK(request_batch_size_ == input_tensors->at("seq_len").shape[0]);
    FT_CHECK(input_tensors->at("seq_len").shape.size() == 2);

    FT_CHECK(request_batch_size_ == input_tensors->at("multi_ids").shape[0]);
    FT_CHECK(input_tensors->at("multi_ids").shape.size() == 2);

    d_word_ids_ = input_tensors->at("word_ids").getPtr<int>();
    d_pos_ids_ = input_tensors->at("pos_ids").getPtr<int>();
    d_sent_ids_ = input_tensors->at("sent_ids").getPtr<int>();
    d_seq_len_ = input_tensors->at("seq_len").getPtr<int>();
    d_multi_ids_ = input_tensors->at("multi_ids").getPtr<int>();
    // allocateBuffer(request_batch_size_, request_seq_len_);

    // Ernie Structure Difference
    // PositionEmbeddingType position_embedding_type = ernie_int8_weights->position_embedding_type;
    std::string cur_graph_key_pre = CudaGraph::AppendShape2Key({PREGRAPH_IDX, request_batch_size_, request_seq_len_});
    CudaGraph* cur_graph_ptr_pre = nullptr;
    bool launched_ = false;
    if (is_enqueue_init_ && use_cuda_graph_) {
        FT_CHECK(is_free_buffer_after_forward_ == false);
        if (cuda_graph_pool_.find(cur_graph_key_pre) == cuda_graph_pool_.end()) {
            cur_graph_ptr_pre = new CudaGraph();
            cur_graph_ptr_pre->beginCapture(stream_);
        }
        else {
            cur_graph_ptr_pre = cuda_graph_pool_[cur_graph_key_pre];
            cur_graph_ptr_pre->launch(stream_);
            launched_ = true;
        }
    }

    // const bool use_inputs_embeds_buffer = false;
    if (!launched_) {
        if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::FUSED_MHA) {
            invokeGetPaddingOffsetErnie(
                token_num_, padding_offset_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);
        }
    }

    invokeEmbeddingLookupConcat(ernie_encoder_emb_buf_,
                                head_num_ * size_per_head_,
                                request_batch_size_,
                                request_seq_len_,
                                word_size_,
                                pos_size_,
                                sent_size_,
                                ernie_int8_weights->sent_embedding_table,
                                ernie_int8_weights->word_embedding_table,
                                ernie_int8_weights->pos_embedding_table,
                                d_sent_ids_,
                                d_word_ids_,
                                d_pos_ids_,
                                stream_);

    if (is_enqueue_init_ && use_cuda_graph_) {
        if (cuda_graph_pool_.find(cur_graph_key_pre) == cuda_graph_pool_.end()) {
            cur_graph_ptr_pre->endCapture(stream_);
            cuda_graph_pool_[cur_graph_key_pre] = cur_graph_ptr_pre;
            cur_graph_ptr_pre->launch(stream_);
        }
    }

    sync_check_cuda_error();
    // size_t  h_token_num_;
    T* ernie_encoder_input_ptr;
    T* ernie_encoder_output_ptr;
    Tensor* padding_offset_tensor_ptr;
    // preprocess (remove padding and build mask)
    switch (attention_type_) {
        case AttentionType::UNFUSED_MHA: {
            invokeBuildEncoderAttentionMask(
                attention_mask_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);

            sync_check_cuda_error();
            cudaMemcpyAsync(&h_token_num_, token_num_, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
            sync_check_cuda_error();
            invokeRemovePadding(
                ernie_encoder_in_buffer_, ernie_encoder_emb_buf_, padding_offset_, h_token_num_, d_model_, stream_);
            sync_check_cuda_error();
            ernie_encoder_input_ptr = ernie_encoder_in_buffer_;
            ernie_encoder_output_ptr = ernie_encoder_out_buffer_;

            padding_offset_tensor_ptr =
                new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num_}, padding_offset_);
            break;
        }
        case AttentionType::UNFUSED_PADDED_MHA: {
            invokeBuildEncoderAttentionMask(
                attention_mask_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);

            h_token_num_ = request_batch_size_ * request_seq_len_;

            sync_check_cuda_error();
            ernie_encoder_input_ptr = ernie_encoder_emb_buf_;
            ernie_encoder_output_ptr = ernie_layer_out_buffer_;
            padding_offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
            break;
        }
        case AttentionType::FUSED_MHA: {
            cudaMemcpyAsync(&h_token_num_, token_num_, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);

            invokeRemovePadding(
                ernie_encoder_in_buffer_, ernie_encoder_emb_buf_, padding_offset_, h_token_num_, d_model_, stream_);
            sync_check_cuda_error();

            sync_check_cuda_error();
            ernie_encoder_input_ptr = ernie_encoder_in_buffer_;
            ernie_encoder_output_ptr = ernie_encoder_out_buffer_;

            invokeGetTrtPaddingOffset(trt_mha_padding_offset_, d_seq_len_, request_batch_size_, stream_);

            padding_offset_tensor_ptr = new Tensor(
                MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size_ + 1}, trt_mha_padding_offset_);
            break;
        }
        case AttentionType::FUSED_PADDED_MHA: {
            h_token_num_ = request_batch_size_ * request_seq_len_;
            invokeGetTrtPaddingOffset(
                trt_mha_padding_offset_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);
            padding_offset_tensor_ptr = new Tensor(
                MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size_ * 2 + 1}, trt_mha_padding_offset_);
            ernie_encoder_input_ptr = ernie_encoder_emb_buf_;
            ernie_encoder_output_ptr = ernie_layer_out_buffer_;
            break;
        }
        default: {
            throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
        }
    }

    // invokeQuantization(int8_buf_, ernie_encoder_input_ptr, h_token_num_*d_model_, &(scale_list->d_scale_list_[3]),
    // stream_); std::vector<Tensor> int8_input_tensors{Tensor{MEMORY_GPU, TYPE_INT8, std::vector<size_t>{h_token_num_,
    // d_model_}, int8_buf_},
    //                                        input_tensors->at(1),
    //                                        input_tensors->at(2)};

    invokeGeneralLayerNorm(ernie_encoder_input_ptr,
                           ernie_encoder_input_ptr,
                           ernie_int8_weights->pre_transformer_layernorm_weights.gamma,
                           ernie_int8_weights->pre_transformer_layernorm_weights.beta,
                           layernorm_eps_,
                           h_token_num_,
                           d_model_,
                           stream_);
    // invokeAddBiasResidualLayerNormCol32(ernie_encoder_input_ptr,
    //                        int8_buf_,
    //                        int8_buf_,
    //                        (const T*)0,
    //                        ernie_int8_weights->pre_transformer_layernorm_weights.gamma,
    //                        ernie_int8_weights->pre_transformer_layernorm_weights.beta,
    //                        h_token_num_,
    //                        d_model_,
    //                        stream_,
    //                        &(scale_list->d_scale_list_[scale_list->p2_offset_ + 3 * hidden_units_]),
    //                       &(scale_list->d_scale_list_[36]));
    sync_check_cuda_error();

    DataType data_type = getTensorType<int>();

    for (uint i = 0; i < num_layer_; i++) {
        T* from_tensor = (i == 0 ? ernie_encoder_input_ptr : ernie_encoder_output_ptr);
        T* out_tensor = ernie_encoder_output_ptr;
        invokeQuantization(int8_buf_, from_tensor, h_token_num_ * d_model_, &(scale_list->d_scale_list_[3]), stream_);

        // attn
        {
            std::vector<Tensor> attn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, int8_buf_},
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{request_batch_size_, 1, request_seq_len_, request_seq_len_},
                       attention_mask_},
                *padding_offset_tensor_ptr,
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{1, head_num_, request_seq_len_, request_seq_len_},
                       nullptr}};
            std::vector<Tensor> attn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, attn_out_buf_}};

            attention_layer_->forward(&attn_output_tensors,
                                      &attn_input_tensors,
                                      &ernie_int8_weights->ernie_encoder_layer_weights[i]->attention_weights);
        }

        // ln
        invokeAddBiasResidualLayerNormCol32(
            out_tensor,
            (const int8_t*)attn_out_buf_,
            (const int8_t*)from_tensor,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->attention_weights.attention_output_weight.bias,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->attn_layernorm_weights.gamma,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->attn_layernorm_weights.beta,
            h_token_num_,
            d_model_,
            stream_,
            &(scale_list->d_scale_list_[scale_list->p2_offset_ + 3 * hidden_units_]),
            &(scale_list->d_scale_list_[36]));

        invokeQuantization(int8_buf_, out_tensor, h_token_num_ * d_model_, &(scale_list->d_scale_list_[3]), stream_);
        // invokeQuantization(normed_attn_out_buf_, normed_attn_out_buf_, h_token_num_ * d_model_,
        // &(scale_list->d_scale_list_[3]), stream_);

        // FFN
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, int8_buf_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, out_tensor}};
            ffn_layer_->forward(&ffn_output_tensors,
                                &ffn_input_tensors,
                                &ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_weights);
        }
        // ln
        invokeAddBiasResidualLayerNormCol32(
            out_tensor,
            (const int8_t*)out_tensor,
            (const int8_t*)attn_out_buf_,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_weights.output_weight.bias,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_layernorm_weights.gamma,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_layernorm_weights.beta,
            h_token_num_,
            d_model_,
            stream_,
            &(scale_list->d_scale_list_[scale_list->p2_offset_ + 3 * hidden_units_]),
            &(scale_list->d_scale_list_[36]));
        sync_check_cuda_error();
    }

    // exit(0);
    // postemb
    invokePostEmbedding(d_multi_ids_,
                        ernie_int8_weights->multi_field_1,
                        ernie_int8_weights->multi_field_3,
                        ernie_int8_weights->multi_field_6,
                        ernie_int8_weights->multi_field_0,
                        ernie_int8_weights->multi_field_5,
                        ernie_int8_weights->multi_field_7,
                        ernie_int8_weights->multi_field_4,
                        ernie_int8_weights->multi_field_2,
                        post_emb_out_buffer_,
                        request_batch_size_,
                        stream_fea_);

    // MatMul(fea_emb_fc)
    {
        int m = request_batch_size_;
        int n = d_model_;
        int k = 160;
        cublas_wrapper_fea_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  ernie_int8_weights->fea_emb_fc.kernel,
                                  n,
                                  post_emb_out_buffer_,
                                  k,
                                  fea_emb_fc_out_buffer_,
                                  n);
        invokeAddBiasRelu(fea_emb_fc_out_buffer_, ernie_int8_weights->fea_emb_fc.bias, m, n, stream_fea_);
    }

    // MatMul(fea_emb_fc2)
    {
        int m = request_batch_size_;
        int n = 384;
        int k = d_model_;
        cublas_wrapper_fea_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  ernie_int8_weights->fea_emb_fc2.kernel,
                                  n,
                                  fea_emb_fc_out_buffer_,
                                  k,
                                  post_emb_out_buffer_,
                                  n);
        invokeAddBiasRelu(post_emb_out_buffer_, ernie_int8_weights->fea_emb_fc2.bias, m, n, stream_fea_);
    }

    // MatMul(cls_out_aside)
    {
        int m = request_batch_size_;
        int n = 1;
        int k = 384;
        cublas_wrapper_fea_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  ernie_int8_weights->cls_out_aside.kernel,
                                  n,
                                  post_emb_out_buffer_,
                                  k,
                                  cls_out_aside_buffer_,
                                  n);
    }
    // exit(0);

    // post process (rebuild padding)
    switch (attention_type_) {
        case AttentionType::UNFUSED_MHA: {

            invokeRebuildPadding(
                ernie_layer_out_buffer_, ernie_encoder_out_buffer_, padding_offset_, h_token_num_, d_model_, stream_);

            break;
        }
        case AttentionType::UNFUSED_PADDED_MHA: {
            break;
        }
        case AttentionType::FUSED_MHA: {

            invokeRebuildPadding(
                ernie_layer_out_buffer_, ernie_encoder_out_buffer_, padding_offset_, h_token_num_, d_model_, stream_);

            break;
        }
        case AttentionType::FUSED_PADDED_MHA: {
            break;
        }
        default: {
            throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
        }
    }

    delete padding_offset_tensor_ptr;
    // todo postprocess
    std::string cur_graph_key_post = CudaGraph::AppendShape2Key({POSTGRAPH_IDX, request_batch_size_, request_seq_len_});
    CudaGraph* cur_graph_ptr_post = nullptr;

    if (is_enqueue_init_ && false) {
        FT_CHECK(is_free_buffer_after_forward_ == false);
        if (cuda_graph_pool_.find(cur_graph_key_post) == cuda_graph_pool_.end()) {
            cur_graph_ptr_post = new CudaGraph();
            cur_graph_ptr_post->beginCapture(stream_);
        }
        else {
            cur_graph_ptr_post = cuda_graph_pool_[cur_graph_key_post];
            cur_graph_ptr_post->launch(stream_);
            return;
        }
    }
    invokeSlice(
        ernie_slice_out_buffer_, ernie_layer_out_buffer_, request_batch_size_, request_seq_len_, d_model_, stream_);
    // MatMul(pooled_fc_matmul)
    {
        int m = request_batch_size_;
        int n = d_model_;
        int k = d_model_;
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              ernie_int8_weights->pooled_fc.kernel,
                              n,
                              ernie_slice_out_buffer_,
                              k,
                              ernie_layer_out_buffer_,
                              n);
        // Add(pooled_fc_add + tanh)
        invokeAddBiasTanh(ernie_layer_out_buffer_, ernie_int8_weights->pooled_fc.bias, m, n, stream_);
    }

    // MatMul(cls_out_matmul)
    {
        int m = request_batch_size_;
        int n = 1;
        int k = d_model_;
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              ernie_int8_weights->cls_out.kernel,
                              n,
                              ernie_layer_out_buffer_,
                              k,
                              cls_out_buffer_,
                              n);
    }

    invokeAddTwoAddBiasSigmoid(cls_out_buffer_,
                               cls_out_aside_buffer_,
                               ernie_int8_weights->cls_out.bias,
                               ernie_int8_weights->cls_out_aside.bias,
                               output_tensors->at("attn_out").getPtr<float>(),
                               request_batch_size_,
                               stream_);

    if (is_enqueue_init_ && false) {
        if (cuda_graph_pool_.find(cur_graph_key_post) == cuda_graph_pool_.end()) {
            cur_graph_ptr_post->endCapture(stream_);
            cuda_graph_pool_[cur_graph_key_post] = cur_graph_ptr_post;
            // NOTE(yuqingding): If we don't rerun the stream, the result will be wrong.  Graph capture will destroy the
            // result???
            cur_graph_ptr_post->launch(stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
    if (!is_enqueue_init_) {
        is_enqueue_init_ = true;
    }
}
template<typename T>
void ErnieINT8<T>::forward(const int* h_word_ids_,
                           const int* h_pos_ids_,
                           const int* h_sent_ids_,
                           const int* h_seq_len_,
                           const int* h_multi_ids_,
                           const int request_batch_size,
                           const int request_seq_len,
                           const ErnieINT8Weight<T>* ernie_int8_weights)
{
    // input_tensors:
    //      word_ids [batch, seqlen]
    //      pos_ids  [batch, seqlen]
    //      sent_ids [batch, seqlen]
    //      seq_len     [batch, 1]
    //      multi_ids   [batch, 8]
    // output tensors:
    //      attn_out [batch, 1]
    const ErnieINT8LayerWeight<T>* ernie_layer_int8_weight = (const ErnieINT8LayerWeight<T>*)ernie_int8_weights;
    const ScaleList* scale_list = &(ernie_layer_int8_weight->scale_list_);

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(is_host_ptr_ == true);

    request_batch_size_ = request_batch_size;
    request_seq_len_ = request_seq_len;
    // preprocess (build embedding and layernorm)
    std::string cur_graph_key_pre = CudaGraph::AppendShape2Key({PREGRAPH_IDX, request_batch_size_, request_seq_len_});
    CudaGraph* cur_graph_ptr_pre = nullptr;
    bool launched_ = false;
    if (is_enqueue_init_ && use_cuda_graph_) {
        FT_CHECK(is_free_buffer_after_forward_ == false);
        if (cuda_graph_pool_.find(cur_graph_key_pre) == cuda_graph_pool_.end()) {
            cur_graph_ptr_pre = new CudaGraph();
            cur_graph_ptr_pre->beginCapture(stream_);
        }
        else {
            cur_graph_ptr_pre = cuda_graph_pool_[cur_graph_key_pre];
            cur_graph_ptr_pre->launch(stream_);
            launched_ = true;
        }
    }

    if (!launched_) {
        cudaAutoCpy(d_word_ids_, h_word_ids_, request_batch_size_ * request_seq_len_, stream_);
        cudaAutoCpy(d_pos_ids_, h_pos_ids_, request_batch_size_ * request_seq_len_, stream_);
        cudaAutoCpy(d_sent_ids_, h_sent_ids_, request_batch_size_ * request_seq_len_, stream_);
        cudaAutoCpy(d_seq_len_, h_seq_len_, request_batch_size_ * 1, stream_);
        cudaAutoCpy(d_multi_ids_, h_multi_ids_, request_batch_size_ * 8, stream_);

        if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::FUSED_MHA) {
            // prevent undefined behavior of the padding parts
            invokeGetPaddingOffsetErnie(
                token_num_, padding_offset_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);
        }
        invokeEmbeddingLookupConcat(ernie_encoder_emb_buf_,
                                    head_num_ * size_per_head_,
                                    request_batch_size_,
                                    request_seq_len_,
                                    word_size_,
                                    pos_size_,
                                    sent_size_,
                                    ernie_int8_weights->sent_embedding_table,
                                    ernie_int8_weights->word_embedding_table,
                                    ernie_int8_weights->pos_embedding_table,
                                    d_sent_ids_,
                                    d_word_ids_,
                                    d_pos_ids_,
                                    stream_);

        if (is_enqueue_init_ && use_cuda_graph_) {
            if (cuda_graph_pool_.find(cur_graph_key_pre) == cuda_graph_pool_.end()) {
                cur_graph_ptr_pre->endCapture(stream_);
                cuda_graph_pool_[cur_graph_key_pre] = cur_graph_ptr_pre;
                cur_graph_ptr_pre->launch(stream_);
            }
        }
    }
    sync_check_cuda_error();

    T* ernie_encoder_input_ptr;
    T* ernie_encoder_output_ptr;
    Tensor* padding_offset_tensor_ptr;
    // preprocess (remove padding and build mask)
    switch (attention_type_) {
        case AttentionType::UNFUSED_MHA: {
            invokeBuildEncoderAttentionMask(
                attention_mask_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);

            sync_check_cuda_error();
            cudaMemcpyAsync(&h_token_num_, token_num_, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
            sync_check_cuda_error();
            invokeRemovePadding(
                ernie_encoder_in_buffer_, ernie_encoder_emb_buf_, padding_offset_, h_token_num_, d_model_, stream_);
            sync_check_cuda_error();
            ernie_encoder_input_ptr = ernie_encoder_in_buffer_;
            ernie_encoder_output_ptr = ernie_encoder_out_buffer_;

            padding_offset_tensor_ptr =
                new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num_}, padding_offset_);
            break;
        }
        case AttentionType::UNFUSED_PADDED_MHA: {
            invokeBuildEncoderAttentionMask(
                attention_mask_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);

            h_token_num_ = request_batch_size_ * request_seq_len_;

            sync_check_cuda_error();
            ernie_encoder_input_ptr = ernie_encoder_emb_buf_;
            ernie_encoder_output_ptr = ernie_layer_out_buffer_;
            padding_offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
            break;
        }
        case AttentionType::FUSED_MHA: {
            cudaMemcpyAsync(&h_token_num_, token_num_, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);

            invokeRemovePadding(
                ernie_encoder_in_buffer_, ernie_encoder_emb_buf_, padding_offset_, h_token_num_, d_model_, stream_);
            sync_check_cuda_error();

            sync_check_cuda_error();
            ernie_encoder_input_ptr = ernie_encoder_in_buffer_;
            ernie_encoder_output_ptr = ernie_encoder_out_buffer_;

            invokeGetTrtPaddingOffset(trt_mha_padding_offset_, d_seq_len_, request_batch_size_, stream_);

            padding_offset_tensor_ptr = new Tensor(
                MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size_ + 1}, trt_mha_padding_offset_);
            break;
        }
        case AttentionType::FUSED_PADDED_MHA: {
            h_token_num_ = request_batch_size_ * request_seq_len_;
            invokeGetTrtPaddingOffset(
                trt_mha_padding_offset_, d_seq_len_, request_batch_size_, request_seq_len_, stream_);
            padding_offset_tensor_ptr = new Tensor(
                MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size_ * 2 + 1}, trt_mha_padding_offset_);
            ernie_encoder_input_ptr = ernie_encoder_emb_buf_;
            ernie_encoder_output_ptr = ernie_layer_out_buffer_;
            break;
        }
        default: {
            throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
        }
    }

    invokeGeneralLayerNorm(ernie_encoder_input_ptr,
                           ernie_encoder_input_ptr,
                           ernie_int8_weights->pre_transformer_layernorm_weights.gamma,
                           ernie_int8_weights->pre_transformer_layernorm_weights.beta,
                           layernorm_eps_,
                           h_token_num_,
                           d_model_,
                           stream_);
    sync_check_cuda_error();

    DataType data_type = getTensorType<T>();

    for (uint i = 0; i < num_layer_; i++) {
        T* from_tensor = (i == 0 ? ernie_encoder_input_ptr : ernie_encoder_output_ptr);
        T* out_tensor = ernie_encoder_output_ptr;
        invokeQuantization(int8_buf_, from_tensor, h_token_num_ * d_model_, &(scale_list->d_scale_list_[3]), stream_);

        // attn
        {
            std::vector<Tensor> attn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, int8_buf_},
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{request_batch_size_, 1, request_seq_len_, request_seq_len_},
                       attention_mask_},
                *padding_offset_tensor_ptr,
                Tensor{MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{1, head_num_, request_seq_len_, request_seq_len_},
                       nullptr}};
            std::vector<Tensor> attn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, attn_out_buf_}};

            attention_layer_->forward(&attn_output_tensors,
                                      &attn_input_tensors,
                                      &ernie_int8_weights->ernie_encoder_layer_weights[i]->attention_weights);
        }

        // ln
        invokeAddBiasResidualLayerNormCol32(
            out_tensor,
            (const int8_t*)attn_out_buf_,
            (const int8_t*)from_tensor,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->attention_weights.attention_output_weight.bias,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->attn_layernorm_weights.gamma,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->attn_layernorm_weights.beta,
            h_token_num_,
            d_model_,
            stream_,
            &(scale_list->d_scale_list_[scale_list->p2_offset_ + 3 * hidden_units_]),
            &(scale_list->d_scale_list_[36]));

        invokeQuantization(int8_buf_, out_tensor, h_token_num_ * d_model_, &(scale_list->d_scale_list_[3]), stream_);
        // invokeQuantization(normed_attn_out_buf_, normed_attn_out_buf_, h_token_num_ * d_model_,
        // &(scale_list->d_scale_list_[3]), stream_);

        // FFN
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, int8_buf_}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num_, d_model_}, out_tensor}};
            ffn_layer_->forward(&ffn_output_tensors,
                                &ffn_input_tensors,
                                &ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_weights);
        }
        // ln
        invokeAddBiasResidualLayerNormCol32(
            out_tensor,
            (const int8_t*)out_tensor,
            (const int8_t*)attn_out_buf_,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_weights.output_weight.bias,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_layernorm_weights.gamma,
            ernie_int8_weights->ernie_encoder_layer_weights[i]->ffn_layernorm_weights.beta,
            h_token_num_,
            d_model_,
            stream_,
            &(scale_list->d_scale_list_[scale_list->p2_offset_ + 3 * hidden_units_]),
            &(scale_list->d_scale_list_[36]));
        sync_check_cuda_error();
    }
    // postemb
    invokePostEmbedding(d_multi_ids_,
                        ernie_int8_weights->multi_field_1,
                        ernie_int8_weights->multi_field_3,
                        ernie_int8_weights->multi_field_6,
                        ernie_int8_weights->multi_field_0,
                        ernie_int8_weights->multi_field_5,
                        ernie_int8_weights->multi_field_7,
                        ernie_int8_weights->multi_field_4,
                        ernie_int8_weights->multi_field_2,
                        post_emb_out_buffer_,
                        request_batch_size_,
                        stream_fea_);

    // MatMul(fea_emb_fc)
    {
        int m = request_batch_size_;
        int n = d_model_;
        int k = 160;
        cublas_wrapper_fea_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  ernie_int8_weights->fea_emb_fc.kernel,
                                  n,
                                  post_emb_out_buffer_,
                                  k,
                                  fea_emb_fc_out_buffer_,
                                  n);
        invokeAddBiasRelu(fea_emb_fc_out_buffer_, ernie_int8_weights->fea_emb_fc.bias, m, n, stream_fea_);
    }

    // MatMul(fea_emb_fc2)
    {
        int m = request_batch_size_;
        int n = 384;
        int k = d_model_;
        cublas_wrapper_fea_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  ernie_int8_weights->fea_emb_fc2.kernel,
                                  n,
                                  fea_emb_fc_out_buffer_,
                                  k,
                                  post_emb_out_buffer_,
                                  n);
        invokeAddBiasRelu(post_emb_out_buffer_, ernie_int8_weights->fea_emb_fc2.bias, m, n, stream_fea_);
    }

    // MatMul(cls_out_aside)
    {
        int m = request_batch_size_;
        int n = 1;
        int k = 384;
        cublas_wrapper_fea_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n,
                                  m,
                                  k,
                                  ernie_int8_weights->cls_out_aside.kernel,
                                  n,
                                  post_emb_out_buffer_,
                                  k,
                                  cls_out_aside_buffer_,
                                  n);
    }
    // exit(0);

    // post process (rebuild padding)
    switch (attention_type_) {
        case AttentionType::UNFUSED_MHA: {
            invokeRebuildPadding(
                ernie_layer_out_buffer_, ernie_encoder_out_buffer_, padding_offset_, h_token_num_, d_model_, stream_);

            break;
        }
        case AttentionType::UNFUSED_PADDED_MHA: {
            break;
        }
        case AttentionType::FUSED_MHA: {
            invokeRebuildPadding(
                ernie_layer_out_buffer_, ernie_encoder_out_buffer_, padding_offset_, h_token_num_, d_model_, stream_);

            break;
        }
        case AttentionType::FUSED_PADDED_MHA: {
            break;
        }
        default: {
            throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
        }
    }

    delete padding_offset_tensor_ptr;
    // todo postprocess
    std::string cur_graph_key_post = CudaGraph::AppendShape2Key({POSTGRAPH_IDX, request_batch_size_, request_seq_len_});
    CudaGraph* cur_graph_ptr_post = nullptr;

    if (is_enqueue_init_ && false) {
        FT_CHECK(is_free_buffer_after_forward_ == false);
        if (cuda_graph_pool_.find(cur_graph_key_post) == cuda_graph_pool_.end()) {
            cur_graph_ptr_post = new CudaGraph();
            cur_graph_ptr_post->beginCapture(stream_);
        }
        else {
            cur_graph_ptr_post = cuda_graph_pool_[cur_graph_key_post];
            cur_graph_ptr_post->launch(stream_);
            return;
        }
    }
    invokeSlice(
        ernie_slice_out_buffer_, ernie_layer_out_buffer_, request_batch_size_, request_seq_len_, d_model_, stream_);
    // MatMul(pooled_fc_matmul)
    {
        int m = request_batch_size_;
        int n = d_model_;
        int k = d_model_;
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              ernie_int8_weights->pooled_fc.kernel,
                              n,
                              ernie_slice_out_buffer_,
                              k,
                              ernie_layer_out_buffer_,
                              n);
        // Add(pooled_fc_add + tanh)
        invokeAddBiasTanh(ernie_layer_out_buffer_, ernie_int8_weights->pooled_fc.bias, m, n, stream_);
    }

    // MatMul(cls_out_matmul)
    {
        int m = request_batch_size_;
        int n = 1;
        int k = d_model_;
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              ernie_int8_weights->cls_out.kernel,
                              n,
                              ernie_layer_out_buffer_,
                              k,
                              cls_out_buffer_,
                              n);
    }

    invokeAddTwoAddBiasSigmoid(cls_out_buffer_,
                               cls_out_aside_buffer_,
                               ernie_int8_weights->cls_out.bias,
                               ernie_int8_weights->cls_out_aside.bias,
                               d_attn_out_,
                               request_batch_size_,
                               stream_);

    if (is_enqueue_init_ && false) {
        if (cuda_graph_pool_.find(cur_graph_key_post) == cuda_graph_pool_.end()) {
            cur_graph_ptr_post->endCapture(stream_);
            cuda_graph_pool_[cur_graph_key_post] = cur_graph_ptr_post;
            // NOTE(yuqingding): If we don't rerun the stream, the result will be wrong.  Graph capture will destroy the
            // result???
            cur_graph_ptr_post->launch(stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
    if (!is_enqueue_init_) {
        is_enqueue_init_ = true;
    }
}

template class ErnieINT8<float>;
template class ErnieINT8<half>;
}  // namespace fastertransformer
