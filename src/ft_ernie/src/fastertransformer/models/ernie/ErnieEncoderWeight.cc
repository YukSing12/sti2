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

#include "src/fastertransformer/models/ernie/ErnieEncoderWeight.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
ErnieEncoderWeight<T>::ErnieEncoderWeight(const size_t                head_num,
                                          const size_t                size_per_head,
                                          const size_t                d_model,
                                          const size_t                inter_size,
                                          const size_t                vocab_size,
                                          const size_t                pos_size,
                                          const size_t                sent_vocab_size,
                                          const size_t                num_layer,
                                          const PositionEmbeddingType pe_type):
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    pos_size_(pos_size),
    sent_vocab_size_(sent_vocab_size),
    num_layer_(num_layer),
    position_embedding_type(pe_type),
    real_weights_num_(23)
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");
    FT_LOG_DEBUG("ErnieEncoderWeight num_layer_ = " + std::to_string(num_layer_));
    initialize();
    mallocWeights();
    setWeightPtr();
    ernie_encoder_layer_weights.clear();
    ernie_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        ernie_encoder_layer_weights.push_back(new ErnieEncoderLayerWeight<T>(l,
                                                                            head_num_,
                                                                            size_per_head,
                                                                            d_model_,
                                                                            inter_size_));
    }
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void ErnieEncoderWeight<T>::initialize()
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");
    weights_size[0] = d_model_ * vocab_size_;
    weights_size[1] = d_model_ * pos_size_;
    weights_size[2] = d_model_ * sent_vocab_size_;
    weights_size[3] = d_model_;
    weights_size[4] = d_model_;
    weights_size[5] = d_model_ * d_model_;
    weights_size[6] = d_model_;
    weights_size[7] = d_model_;
    weights_size[8] = 1;
    // multi_field 0 ~ 7
    weights_size[9] = 11 * 20;
    weights_size[10] = 13 * 20;
    weights_size[11] = 11 * 20;
    weights_size[12] = 1432 * 20;
    weights_size[13] = 11 * 20;
    weights_size[14] = 11 * 20;
    weights_size[15] = 11 * 20;
    weights_size[16] = 11 * 20;
    // feature emb fc
    weights_size[17] = 160 * d_model_;
    weights_size[18] = d_model_;
    weights_size[19] = 384 * d_model_;
    weights_size[20] = 384;
    weights_size[21] = 384;
    weights_size[22] = 1;
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
ErnieEncoderWeight<T>::~ErnieEncoderWeight()
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");

    if (is_maintain_buffer == true) {
        for (int i = 0; i < real_weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_transformer_layernorm_weights.gamma = nullptr;
        pre_transformer_layernorm_weights.beta  = nullptr;
        pooled_fc.kernel                        = nullptr;
        pooled_fc.bias                          = nullptr;
        fea_emb_fc.kernel                       = nullptr;
        fea_emb_fc.bias                         = nullptr;
        fea_emb_fc2.kernel                      = nullptr;
        fea_emb_fc2.bias                        = nullptr;
        cls_out.kernel                          = nullptr;
        cls_out.bias                            = nullptr;
        cls_out_aside.kernel                    = nullptr;
        cls_out_aside.bias                      = nullptr;
        is_maintain_buffer                      = false;
    }
    for (int i = 0; i < num_layer_; i++) {
        delete ernie_encoder_layer_weights[i];
    }
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
ErnieEncoderWeight<T>::ErnieEncoderWeight(const ErnieEncoderWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    pos_size_(other.pos_size_),
    sent_vocab_size_(other.sent_vocab_size_),
    num_layer_(other.num_layer_),
    position_embedding_type(other.position_embedding_type),
    real_weights_num_(other.real_weights_num_)
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    ernie_encoder_layer_weights.clear();
    ernie_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        ernie_encoder_layer_weights.push_back(new ErnieEncoderLayerWeight<T>(*other.ernie_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
ErnieEncoderWeight<T>& ErnieEncoderWeight<T>::operator=(const ErnieEncoderWeight& other)
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");

    head_num_                  = other.head_num_;
    size_per_head_             = other.size_per_head_;
    d_model_                   = other.d_model_;
    inter_size_                = other.inter_size_;
    vocab_size_                = other.vocab_size_;
    num_layer_                 = other.num_layer_;
    position_embedding_type    = other.position_embedding_type;
    real_weights_num_          = other.real_weights_num_;
    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    ernie_encoder_layer_weights.clear();
    ernie_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        ernie_encoder_layer_weights.push_back(new ErnieEncoderLayerWeight<T>(*other.ernie_encoder_layer_weights[i]));
    }
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");

    return *this;
}

template<typename T>
void ErnieEncoderWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");
    word_embedding_table                    = weights_ptr[0];
    pos_embedding_table                     = weights_ptr[1];
    sent_embedding_table                    = weights_ptr[2];
    pre_transformer_layernorm_weights.gamma = weights_ptr[3];
    pre_transformer_layernorm_weights.beta  = weights_ptr[4];
    pooled_fc.kernel                        = weights_ptr[5];
    pooled_fc.bias                          = weights_ptr[6];
    cls_out.kernel                          = weights_ptr[7];
    cls_out.bias                            = weights_ptr[8];
    multi_field_1                           = weights_ptr[9];
    multi_field_3                           = weights_ptr[10];
    multi_field_6                           = weights_ptr[11];
    multi_field_0                           = weights_ptr[12];
    multi_field_5                           = weights_ptr[13];
    multi_field_7                           = weights_ptr[14];
    multi_field_4                           = weights_ptr[15];
    multi_field_2                           = weights_ptr[16];
    fea_emb_fc.kernel                       = weights_ptr[17];
    fea_emb_fc.bias                         = weights_ptr[18];
    fea_emb_fc2.kernel                      = weights_ptr[19];
    fea_emb_fc2.bias                        = weights_ptr[20];
    cls_out_aside.kernel                    = weights_ptr[21];
    cls_out_aside.bias                      = weights_ptr[22];
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void ErnieEncoderWeight<T>::mallocWeights()
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer = true;
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void ErnieEncoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");
    
    FtCudaDataType model_file_type = getModelFileType(dir_path + "/config.ini", "ernie");
    FT_CHECK(is_maintain_buffer == true);
    loadWeightFromBin<T>(
        weights_ptr[0], {(size_t)weights_size[0]}, dir_path + "/word_embedding.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[1], {(size_t)weights_size[1]}, dir_path + "/pos_embedding.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[2], {(size_t)weights_size[2]}, dir_path + "/sent_embedding.bin", model_file_type);

    loadWeightFromBin<T>(
        weights_ptr[3], {(size_t)weights_size[3]}, dir_path + "/pre_encoder_layer_norm_scale.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[4], {(size_t)weights_size[4]}, dir_path + "/pre_encoder_layer_norm_bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[5], {(size_t)weights_size[5]}, dir_path + "/pooled_fc.w_0.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[6], {(size_t)weights_size[6]}, dir_path + "/pooled_fc.b_0.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[7], {(size_t)weights_size[7]}, dir_path + "/cls_out_w.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[8], {(size_t)weights_size[8]}, dir_path + "/cls_out_b.bin", model_file_type);
    
    loadWeightFromBin<T>(
        weights_ptr[9], {(size_t)weights_size[9]}, dir_path + "/multi_field_1.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[10], {(size_t)weights_size[10]}, dir_path + "/multi_field_3.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[11], {(size_t)weights_size[11]}, dir_path + "/multi_field_6.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[12], {(size_t)weights_size[12]}, dir_path + "/multi_field_0.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[13], {(size_t)weights_size[13]}, dir_path + "/multi_field_5.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[14], {(size_t)weights_size[14]}, dir_path + "/multi_field_7.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[15], {(size_t)weights_size[15]}, dir_path + "/multi_field_4.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[16], {(size_t)weights_size[16]}, dir_path + "/multi_field_2.bin", model_file_type);
        
    loadWeightFromBin<T>(
        weights_ptr[17], {(size_t)weights_size[17]}, dir_path + "/feature_emb_fc_w.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[18], {(size_t)weights_size[18]}, dir_path + "/feature_emb_fc_b.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[19], {(size_t)weights_size[19]}, dir_path + "/feature_emb_fc_w2.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[20], {(size_t)weights_size[20]}, dir_path + "/feature_emb_fc_b2.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[21], {(size_t)weights_size[21]}, dir_path + "/cls_out_w_aside.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[22], {(size_t)weights_size[22]}, dir_path + "/cls_out_b_aside.bin", model_file_type);
    for (int l = 0; l < num_layer_; l++) {
        ernie_encoder_layer_weights[l]->loadModel(dir_path + "/",
                                                   model_file_type);
    }
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template<typename T>
void ErnieEncoderWeight<T>::resizeLayer(const int num_layer)
{
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " start");
    ernie_encoder_layer_weights.clear();
    num_layer_ = num_layer;
    ernie_encoder_layer_weights.reserve(num_layer_);
    for (int l = 0; l < num_layer_; l++) {
        ernie_encoder_layer_weights.push_back(new ErnieEncoderLayerWeight<T>());
    }
    FT_LOG_DEBUG("ErnieEncoderWeight " + std::string(__func__) + " end");
}

template struct ErnieEncoderWeight<float>;
template struct ErnieEncoderWeight<half>;
#ifdef ENABLE_BF16
template struct ErnieEncoderWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer