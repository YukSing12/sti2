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

#include "src/fastertransformer/models/ernie_int8/ErnieINT8LayerWeight.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ErnieINT8LayerWeight<T>::ErnieINT8LayerWeight(const size_t layer_id,
                                              const size_t head_num,
                                              const size_t size_per_head,
                                              const size_t d_model,
                                              const size_t inter_size):
    layer_id_(layer_id), head_num_(head_num), size_per_head_(size_per_head), d_model_(d_model), inter_size_(inter_size)
{
    real_weights_num_ = 16;
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    setWeightPtr();
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void ErnieINT8LayerWeight<T>::initialize()
{
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");
    weights_size[0] = d_model_ * d_model_;
    weights_size[1] = d_model_;
    weights_size[2] = d_model_ * d_model_;
    weights_size[3] = d_model_;
    weights_size[4] = d_model_ * d_model_;
    weights_size[5] = d_model_;

    weights_size[6] = d_model_ * d_model_;
    weights_size[7] = d_model_;

    weights_size[8] = d_model_;
    weights_size[9] = d_model_;

    weights_size[10] = d_model_ * inter_size_;
    weights_size[11] = inter_size_;
    weights_size[12] = inter_size_ * d_model_;
    weights_size[13] = d_model_;

    weights_size[14] = d_model_;
    weights_size[15] = d_model_;

    scale_list_.size_ = ACTIVATION_AMAX_NUM + 9 * d_model_ + INT8O_GEMM_NUM + TRT_AMAX_NUM + SCALE_RESERVE_NUM;
    scale_list_.p3_offset_ = ACTIVATION_AMAX_NUM + 9 * d_model_;
    scale_list_.p4_offset_ = ACTIVATION_AMAX_NUM + 9 * d_model_ + INT8O_GEMM_NUM;
    deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
    scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);

    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");
}

template<typename T>
ErnieINT8LayerWeight<T>::~ErnieINT8LayerWeight()
{
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");
    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }
        deviceFree(scale_list_ptr[0]);
        free(scale_list_ptr[1]);

        attention_weights.query_weight.kernel = nullptr;
        attention_weights.key_weight.kernel = nullptr;
        attention_weights.value_weight.kernel = nullptr;
        attention_weights.attention_output_weight.kernel = nullptr;
        attn_layernorm_weights.gamma = nullptr;
        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight2.kernel = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_layernorm_weights.gamma = nullptr;
        attention_weights.query_weight.bias = nullptr;
        attention_weights.key_weight.bias = nullptr;
        attention_weights.value_weight.bias = nullptr;
        attention_weights.attention_output_weight.bias = nullptr;
        attn_layernorm_weights.beta = nullptr;
        ffn_weights.intermediate_weight.bias = nullptr;
        ffn_weights.intermediate_weight2.bias = nullptr;
        ffn_weights.output_weight.bias = nullptr;
        ffn_layernorm_weights.beta = nullptr;
        is_maintain_buffer = false;
    }

    if (is_maintain_sp_buffer == true) {
        for (int i = 0; i < 6; i++) {
            deviceFree(sp_weights_ptr[i]);
        }
        attention_weights.query_weight.sp_kernel = nullptr;
        attention_weights.key_weight.sp_kernel = nullptr;
        attention_weights.value_weight.sp_kernel = nullptr;
        attention_weights.attention_output_weight.sp_kernel = nullptr;
        ffn_weights.intermediate_weight.sp_kernel = nullptr;
        ffn_weights.output_weight.sp_kernel = nullptr;
        is_maintain_sp_buffer = false;
    }
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");
}

template<typename T>
ErnieINT8LayerWeight<T>::ErnieINT8LayerWeight(const ErnieINT8LayerWeight& other):
    layer_id_(other.layer_id_),
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    real_weights_num_(other.real_weights_num_)
{
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }

    scale_list_.size_ = other.scale_list_.size_;
    scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
    scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
    deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
    cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
    scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
    memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);

    setWeightPtr();
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");
}

template<typename T>
ErnieINT8LayerWeight<T>& ErnieINT8LayerWeight<T>::operator=(const ErnieINT8LayerWeight<T>& other)
{
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");

    layer_id_ = other.layer_id_;
    head_num_ = other.head_num_;
    size_per_head_ = other.size_per_head_;
    d_model_ = other.d_model_;
    inter_size_ = other.inter_size_;
    real_weights_num_ = other.real_weights_num_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }

    scale_list_.size_ = other.scale_list_.size_;
    scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
    scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
    deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
    cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
    scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
    memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);

    setWeightPtr();
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void ErnieINT8LayerWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");
    attention_weights.query_weight.kernel = weights_ptr[0];
    attention_weights.query_weight.bias = weights_ptr[1];
    attention_weights.key_weight.kernel = weights_ptr[2];
    attention_weights.key_weight.bias = weights_ptr[3];
    attention_weights.value_weight.kernel = weights_ptr[4];
    attention_weights.value_weight.bias = weights_ptr[5];
    attention_weights.attention_output_weight.kernel = weights_ptr[6];
    attention_weights.attention_output_weight.bias = weights_ptr[7];
    attn_layernorm_weights.gamma = weights_ptr[8];
    attn_layernorm_weights.beta = weights_ptr[9];
    ffn_weights.intermediate_weight.kernel = weights_ptr[10];
    ffn_weights.intermediate_weight.bias = weights_ptr[11];
    ffn_weights.output_weight.kernel = weights_ptr[12];
    ffn_weights.output_weight.bias = weights_ptr[13];
    ffn_layernorm_weights.gamma = weights_ptr[14];
    ffn_layernorm_weights.beta = weights_ptr[15];

    scale_list_.d_scale_list_ = scale_list_ptr[0];
    scale_list_.h_scale_list_ = scale_list_ptr[1];
    attention_weights.scale_list_ptr = &scale_list_;
    ffn_weights.scale_list_ptr = &scale_list_;

    is_maintain_buffer = true;
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void ErnieINT8LayerWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void ErnieINT8LayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " start");

    FT_CHECK(is_maintain_buffer == true);
    loadWeightFromBin<T>(weights_ptr[0],
                         {weights_size[0]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_query_fc.w_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[1],
                         {weights_size[1]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_query_fc.b_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {weights_size[2]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_key_fc.w_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {weights_size[3]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_key_fc.b_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[4],
                         {weights_size[4]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_value_fc.w_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5],
                         {weights_size[5]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_value_fc.b_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[6],
                         {weights_size[6]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_output_fc.w_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[7],
                         {weights_size[7]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_multi_head_att_output_fc.b_0"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[8],
                         {weights_size[8]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_post_att_layer_norm_scale"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[9],
                         {weights_size[9]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_post_att_layer_norm_bias" + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[10],
                         {weights_size[10]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_ffn_fc_0.w_0" + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[11],
                         {weights_size[11]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_ffn_fc_0.b_0" + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[12],
                         {weights_size[12]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_ffn_fc_1.w_0" + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[13],
                         {weights_size[13]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_ffn_fc_1.b_0" + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[14],
                         {weights_size[14]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_post_ffn_layer_norm_scale"
                             + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[15],
                         {weights_size[15]},
                         dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_post_ffn_layer_norm_bias" + ".bin",
                         model_file_type);

    loadWeightFromBin<float>(scale_list_ptr[0],
                             {scale_list_.size_},
                             dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_scale_list" + ".bin",
                             model_file_type);

    loadWeightFromBin<float>(scale_list_ptr[1],
                             {scale_list_.size_},
                             dir_path + "encoder_layer_" + std::to_string(layer_id_) + "_scale_list" + ".bin",
                             model_file_type);

    FT_LOG_DEBUG("ErnieINT8LayerWeight " + std::string(__func__) + " end");
}

template struct ErnieINT8LayerWeight<float>;
template struct ErnieINT8LayerWeight<half>;
#ifdef ENABLE_BF16
template struct ErnieINT8LayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
