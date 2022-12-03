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

#include "src/fastertransformer/models/ernie/ErnieEncoder.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/ernie_kernels.h"

namespace fastertransformer {

template<typename T>
void ErnieEncoder<T>::initialize()
{
    if ((attention_type_ == AttentionType::FUSED_MHA || attention_type_ == AttentionType::FUSED_PADDED_MHA)
        && false  // disable the fused mha
        && std::is_same<T, half>::value == true && max_seq_len_ <= 384) {
        FT_CHECK(false);  // fused mha does not support relatvie attention bias now.
        // attention_layer_ = new FusedAttentionLayer<T>(max_batch_size_,
        //                                               max_seq_len_,
        //                                               head_num_,
        //                                               size_per_head_,
        //                                               sm_,
        //                                               q_scaling_ * (1.0 / sqrtf(size_per_head_ * 1.0f)),
        //                                               stream_,
        //                                               cublas_wrapper_,
        //                                               allocator_,
        //                                               is_free_buffer_after_forward_,
        //                                               sparse_);
    }
    else if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::UNFUSED_PADDED_MHA) {
        attention_layer_ =
            new TensorParallelUnfusedAttentionLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       head_num_,
                                                       size_per_head_,
                                                       d_model_,
                                                       q_scaling_,  // adjust according to checkpoint structure
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
    }

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU
                                || activation_type_ == ActivationType::SiGLU;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       1,
                                                       d_model_,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       0,
                                                       use_gated_activation,  // don't use GeGLU
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       1,
                                                       d_model_,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Silu || activation_type_ == ActivationType::SiGLU) {
        ffn_layer_ = new TensorParallelSiluFfnLayer<T>(max_batch_size_,
                                                       max_seq_len_,
                                                       1,
                                                       d_model_,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
}

template<typename T>
ErnieEncoder<T>::ErnieEncoder(size_t                              max_batch_size,
                        size_t                              max_seq_len,
                        size_t                              head_num,
                        size_t                              size_per_head,
                        size_t                              inter_size,
                        size_t                              d_model,
                        size_t                              num_layer,
                        size_t                              num_bucket_or_max_seq_len,
                        size_t                              max_distance,
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
                        std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                        int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    num_bucket_or_max_seq_len_(num_bucket_or_max_seq_len),
    max_distance_(max_distance),
    word_size_(word_size),
    pos_size_(pos_size),
    sent_size_(sent_size),
    sm_(sm),
    q_scaling_(q_scaling),
    attention_type_(attention_type),
    sparse_(sparse),
    activation_type_(activation_type),
    layernorm_type_(layernorm_type),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
ErnieEncoder<T>::ErnieEncoder(ErnieEncoder<T> const& ernie_encoder):
    BaseLayer(ernie_encoder),
    max_batch_size_(ernie_encoder.max_batch_size_),
    max_seq_len_(ernie_encoder.max_seq_len_),
    head_num_(ernie_encoder.head_num_),
    size_per_head_(ernie_encoder.size_per_head_),
    inter_size_(ernie_encoder.inter_size_),
    d_model_(ernie_encoder.d_model_),
    hidden_units_(ernie_encoder.hidden_units_),
    num_layer_(ernie_encoder.num_layer_),
    num_bucket_or_max_seq_len_(ernie_encoder.num_bucket_or_max_seq_len_),
    max_distance_(ernie_encoder.max_distance_),
    word_size_(ernie_encoder.word_size_),
    pos_size_(ernie_encoder.pos_size_),
    sent_size_(ernie_encoder.sent_size_),
    sm_(ernie_encoder.sm_),
    q_scaling_(ernie_encoder.q_scaling_),
    attention_type_(ernie_encoder.attention_type_),
    sparse_(ernie_encoder.sparse_),
    activation_type_(ernie_encoder.activation_type_),
    layernorm_type_(ernie_encoder.layernorm_type_),
    tensor_para_(ernie_encoder.tensor_para_),
    pipeline_para_(ernie_encoder.pipeline_para_),
    custom_all_reduce_comm_(ernie_encoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(ernie_encoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
ErnieEncoder<T>::~ErnieEncoder()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ErnieEncoder<T>::setStream(cudaStream_t stream)
{
    attention_layer_->setStream(stream);
    ffn_layer_->setStream(stream);
    BaseLayer::setStream(stream);
}

template<typename T>
void ErnieEncoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        token_num_ = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
        padding_offset_ =
            (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * max_batch_size_ * max_seq_len_, false);
        trt_mha_padding_offset_ =
            (int*)allocator_->reMalloc(trt_mha_padding_offset_, sizeof(int) * (2 * max_batch_size_ + 1), false);

        attention_mask_ =
            (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);
        relative_attention_bias_ = (T*)allocator_->reMalloc(
            relative_attention_bias_, sizeof(T) * head_num_ * max_seq_len_ * max_seq_len_, false);

        ernie_encoder_emb_buf_ =
            (T*)allocator_->reMalloc(ernie_encoder_emb_buf_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        ernie_encoder_in_buffer_ = (T*)allocator_->reMalloc(
            ernie_encoder_in_buffer_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        attn_out_buf_ =
            (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        ernie_encoder_out_buffer_ = (T*)allocator_->reMalloc(
            ernie_encoder_out_buffer_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_  = nullptr;
            normed_attn_out_buf_ = nullptr;
        }
        else {
            normed_from_tensor_ = (T*)allocator_->reMalloc(
                normed_from_tensor_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
            normed_attn_out_buf_ = (T*)allocator_->reMalloc(
                normed_attn_out_buf_, sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        }
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void ErnieEncoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    token_num_      = (size_t*)allocator_->reMalloc(token_num_, sizeof(size_t) * 1, false);
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
    attn_out_buf_ = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * d_model_, false);
    ernie_encoder_out_buffer_ =
        (T*)allocator_->reMalloc(ernie_encoder_out_buffer_, sizeof(T) * batch_size * seq_len * d_model_, false);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_  = nullptr;
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
void ErnieEncoder<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&token_num_));
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&trt_mha_padding_offset_));

        allocator_->free((void**)(&attention_mask_));
        allocator_->free((void**)(&relative_attention_bias_));
        allocator_->free((void**)(&ernie_encoder_emb_buf_));
        allocator_->free((void**)(&ernie_encoder_in_buffer_));
        allocator_->free((void**)(&attn_out_buf_));
        allocator_->free((void**)(&ernie_encoder_out_buffer_));

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_  = nullptr;
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
bool ErnieEncoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ErnieEncoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ErnieEncoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ErnieEncoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
void ErnieEncoder<T>::forward(std::vector<Tensor>*       output_tensors,
                           const std::vector<Tensor>* input_tensors,
                           const ErnieEncoderWeight<T>*  ernie_encoder_weights)
{
    // input_tensors:
    //      word_ids [batch, seqlen, 1]
    //      pos_ids  [batch, seqlen, 1]
    //      sent_ids [batch, seqlen, 1]
    //      seq_len     [batch, 1, 1]
    // output tensors:
    //      attn_out [batch, seqlen, d_model]

    std::unordered_map<std::string, Tensor> input_tensors_map{{"word_ids",  input_tensors->at(0)},
                                                              {"pos_ids",   input_tensors->at(1)},
                                                              {"sent_ids",  input_tensors->at(2)},
                                                              {"seq_len",   input_tensors->at(3)}};

    std::unordered_map<std::string, Tensor> output_tensors_map{{"attn_out", output_tensors->at(0)}};
    forward(&output_tensors_map, &input_tensors_map, ernie_encoder_weights);
}

template<typename T>
void ErnieEncoder<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                           const std::unordered_map<std::string, Tensor>* input_tensors,
                           const ErnieEncoderWeight<T>*                      ernie_encoder_weights)
{
    // input_tensors:
    //      word_ids [batch, seqlen, 1]
    //      pos_ids  [batch, seqlen, 1]
    //      sent_ids [batch, seqlen, 1]
    //      seq_len     [batch, 1, 1]
    // output tensors:
    //      attn_out [batch, seqlen, d_model]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->at("word_ids").shape.size() == 2);

    const size_t request_batch_size = input_tensors->at("word_ids").shape[0];
    const size_t request_seq_len    = input_tensors->at("word_ids").shape[1];
    FT_CHECK(input_tensors->size() == 4);
    FT_CHECK(request_batch_size == input_tensors->at("pos_ids").shape[0]);
    FT_CHECK(input_tensors->at("pos_ids").shape.size() == 2);
    FT_CHECK(request_batch_size == input_tensors->at("sent_ids").shape[0]);
    FT_CHECK(input_tensors->at("sent_ids").shape.size() == 2);
    FT_CHECK(request_batch_size == input_tensors->at("seq_len").shape[0]);
    FT_CHECK(input_tensors->at("seq_len").shape.size() == 1);
    allocateBuffer(request_batch_size, request_seq_len);

    // Ernie Structure Difference
    PositionEmbeddingType position_embedding_type = ernie_encoder_weights->position_embedding_type;

    const bool use_inputs_embeds_buffer = false;

    if (attention_type_ == AttentionType::UNFUSED_MHA || attention_type_ == AttentionType::FUSED_MHA) {
        // prevent undefined behavior of the padding parts
        cudaMemset(output_tensors->at("attn_out").getPtr<T>(),
                   0,
                   sizeof(T) * request_batch_size * 1);
    }

    // parallal num = 1
    // local batch size = request batch size
    const size_t local_batch_size = getLocalBatchSize(request_batch_size, request_seq_len, pipeline_para_.world_size_); 
    const size_t iteration_num    = request_batch_size / local_batch_size;

    for (uint ite = 0; ite < iteration_num; ite++) {
        size_t id_offset      = ite * local_batch_size;
        size_t d_model_offset = id_offset * request_seq_len * d_model_;

        const int* sequence_lengths = input_tensors->at("seq_len").getPtr<int>() + id_offset;
        // preprocess (build embedding and layernorm)
        invokeEmbeddingLookupConcat(ernie_encoder_emb_buf_,
                                 head_num_ * size_per_head_,
                                 local_batch_size,
                                 request_seq_len,
                                 word_size_,
                                 pos_size_,
                                 sent_size_,
                                 ernie_encoder_weights->sent_embedding_table,
                                 ernie_encoder_weights->word_embedding_table,
                                 ernie_encoder_weights->pos_embedding_table,
                                 input_tensors->at("sent_ids").getPtrWithOffset<int>(id_offset * request_seq_len),
                                 input_tensors->at("word_ids").getPtrWithOffset<int>(id_offset * request_seq_len),
                                 input_tensors->at("pos_ids").getPtrWithOffset<int>(id_offset * request_seq_len),
                                 stream_);

        sync_check_cuda_error();
        size_t  h_token_num;
        T*      ernie_encoder_input_ptr;
        T*      ernie_encoder_output_ptr;
        Tensor* padding_offset_tensor_ptr;
        // preprocess (remove padding and build mask)
        switch (attention_type_) {
            case AttentionType::UNFUSED_MHA: {
                invokeBuildEncoderAttentionMask(
                    attention_mask_, sequence_lengths, local_batch_size, request_seq_len, stream_);

                sync_check_cuda_error();
                invokeGetPaddingOffset(&h_token_num,
                                       token_num_,
                                       padding_offset_,
                                       sequence_lengths,
                                       local_batch_size,
                                       request_seq_len,
                                       stream_);
                sync_check_cuda_error();

                invokeRemovePadding(ernie_encoder_in_buffer_,
                                    ernie_encoder_emb_buf_,
                                    padding_offset_,
                                    h_token_num,
                                    d_model_,
                                    stream_);
                sync_check_cuda_error();
                ernie_encoder_input_ptr  = ernie_encoder_in_buffer_;
                ernie_encoder_output_ptr = ernie_encoder_out_buffer_;

                padding_offset_tensor_ptr =
                    new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num}, padding_offset_);
                break;
            }
            case AttentionType::UNFUSED_PADDED_MHA: {
                invokeBuildEncoderAttentionMask(
                    attention_mask_, sequence_lengths, local_batch_size, request_seq_len, stream_);

                h_token_num = local_batch_size * request_seq_len;

                sync_check_cuda_error();
                h_token_num               = local_batch_size * request_seq_len;
                ernie_encoder_input_ptr      = ernie_encoder_emb_buf_;
                ernie_encoder_output_ptr     = output_tensors->at("attn_out").getPtr<T>() + d_model_offset;
                padding_offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
                break;
            }
            case AttentionType::FUSED_MHA: {
                FT_CHECK(false);  // not support FUSED_MHA now
                // invokeGetPaddingOffset(&h_token_num,
                //                        token_num_,
                //                        padding_offset_,
                //                        sequence_lengths,
                //                        local_batch_size,
                //                        request_seq_len,
                //                        stream_);

                // if (pipeline_para_.rank_ == 0) {
                //     invokeRemovePadding(
                //         ernie_encoder_in_buffer_, ernie_encoder_emb_buf_, padding_offset_, h_token_num, d_model_, stream_);
                //     sync_check_cuda_error();
                // }
                // sync_check_cuda_error();
                // ernie_encoder_input_ptr = ernie_encoder_in_buffer_;
                // ernie_encoder_output_ptr = ernie_encoder_out_buffer_;

                // invokeGetTrtPaddingOffset(trt_mha_padding_offset_, sequence_lengths, local_batch_size, stream_);

                // padding_offset_tensor_ptr = new Tensor(
                //     MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size + 1}, trt_mha_padding_offset_);
                // break;
            }
            case AttentionType::FUSED_PADDED_MHA: {
                FT_CHECK(false);  // not support FUSED_MHA now
                // h_token_num = local_batch_size * request_seq_len;
                // invokeGetTrtPaddingOffset(
                //     trt_mha_padding_offset_, sequence_lengths, local_batch_size, request_seq_len, stream_);
                // padding_offset_tensor_ptr = new Tensor(
                //     MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size * 2 + 1}, trt_mha_padding_offset_);
                // ernie_encoder_input_ptr = ernie_encoder_emb_buf_ + d_model_offset;
                // ernie_encoder_output_ptr = output_tensors->at("attn_out").getPtr<T>() + d_model_offset;
                // break;
            }
            default: {
                throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
            }
        }
        // TODO Pre Encoder LayerNorm

        invokeGeneralLayerNorm(ernie_encoder_input_ptr,
                               ernie_encoder_input_ptr,
                               ernie_encoder_weights->pre_transformer_layernorm_weights.gamma,
                               ernie_encoder_weights->pre_transformer_layernorm_weights.beta,
                               layernorm_eps_,
                               h_token_num,
                               d_model_,
                               stream_);
        sync_check_cuda_error();

        DataType data_type = getTensorType<T>();
        
        for (uint i = 0; i < num_layer_; i++) {
            T* from_tensor = (i == 0 ? ernie_encoder_input_ptr : ernie_encoder_output_ptr);
            T* out_tensor  = ernie_encoder_output_ptr;

            // attn
            {
                std::vector<Tensor> attn_input_tensors{
                    Tensor{MEMORY_GPU,
                           data_type,
                           std::vector<size_t>{h_token_num, d_model_},
                           from_tensor},
                    Tensor{MEMORY_GPU,
                           data_type,
                           std::vector<size_t>{local_batch_size, 1, request_seq_len, request_seq_len},
                           attention_mask_},
                    *padding_offset_tensor_ptr,
                    Tensor{MEMORY_GPU,
                           data_type,
                           std::vector<size_t>{1, head_num_, request_seq_len, request_seq_len},
                           ernie_encoder_weights->position_embedding_type == PositionEmbeddingType::relative ?
                               relative_attention_bias_ :
                               nullptr}};
                std::vector<Tensor> attn_output_tensors{
                    Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, d_model_}, attn_out_buf_}};

                attention_layer_->forward(&attn_output_tensors,
                                          &attn_input_tensors,
                                          &ernie_encoder_weights->ernie_encoder_layer_weights[i]->attention_weights);
            }
            // ln
            invokeGeneralAddBiasResidualPreLayerNorm(
                attn_out_buf_,
                attn_out_buf_,
                from_tensor,
                ernie_encoder_weights->ernie_encoder_layer_weights[i]->attn_layernorm_weights.gamma,
                ernie_encoder_weights->ernie_encoder_layer_weights[i]->attn_layernorm_weights.beta,
                ernie_encoder_weights->ernie_encoder_layer_weights[i]->attention_weights.attention_output_weight.bias,
                layernorm_eps_,
                h_token_num,
                d_model_,
                stream_);

            // FFN
            {
                std::vector<Tensor> ffn_input_tensors{
                    Tensor{MEMORY_GPU,
                           data_type,
                           std::vector<size_t>{h_token_num, d_model_},
                           layernorm_type_ == LayerNormType::pre_layernorm ? normed_attn_out_buf_ : attn_out_buf_}};
                std::vector<Tensor> ffn_output_tensors{
                    Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, d_model_}, out_tensor}};
                ffn_layer_->forward(&ffn_output_tensors,
                                    &ffn_input_tensors,
                                    &ernie_encoder_weights->ernie_encoder_layer_weights[i]->ffn_weights);
            }
            // ln
            invokeGeneralAddBiasResidualPreLayerNorm(
                out_tensor,
                out_tensor,
                attn_out_buf_,
                ernie_encoder_weights->ernie_encoder_layer_weights[i]->ffn_layernorm_weights.gamma,
                ernie_encoder_weights->ernie_encoder_layer_weights[i]->ffn_layernorm_weights.beta,
                ernie_encoder_weights->ernie_encoder_layer_weights[i]->ffn_weights.output_weight.bias,
                layernorm_eps_,
                h_token_num,
                d_model_,
                stream_);
            sync_check_cuda_error();
        }

        // exit(0);

        if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
            // post process (rebuild padding)
            switch (attention_type_) {
                case AttentionType::UNFUSED_MHA: {
                    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                        invokeRebuildPadding(output_tensors->at("attn_out").getPtr<T>() + d_model_offset,
                                             ernie_encoder_out_buffer_,
                                             padding_offset_,
                                             h_token_num,
                                             d_model_,
                                             stream_);
                    }
                    break;
                }
                case AttentionType::UNFUSED_PADDED_MHA: {
                    break;
                }
                case AttentionType::FUSED_MHA: {
                    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                        invokeRebuildPadding(output_tensors->at("attn_out").getPtr<T>() + d_model_offset,
                                             ernie_encoder_out_buffer_,
                                             padding_offset_,
                                             h_token_num,
                                             d_model_,
                                             stream_);
                    }
                    break;
                }
                case AttentionType::FUSED_PADDED_MHA: {
                    break;
                }
                default: {
                    throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type \n"));
                }
            }
        }

        delete padding_offset_tensor_ptr;
    }
    // todo postprocess
    

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class ErnieEncoder<float>;
template class ErnieEncoder<half>;
#ifdef ENABLE_BF16
template class ErnieEncoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
