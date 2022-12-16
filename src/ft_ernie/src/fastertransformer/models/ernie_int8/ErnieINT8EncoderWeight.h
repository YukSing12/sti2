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

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/models/ernie_int8/ErnieINT8EncoderLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct ErnieINT8EncoderWeight {

    ErnieINT8EncoderWeight() = default;
    ErnieINT8EncoderWeight(const size_t                head_num,
                       const size_t                size_per_head,
                       const size_t                d_model,
                       const size_t                inter_size,
                       const size_t                vocab_size,
                       const size_t                pos_size,
                       const size_t                sent_vocab_size,
                       const size_t                num_layer,
                       const PositionEmbeddingType pe_type                   = PositionEmbeddingType::relative);
    ~ErnieINT8EncoderWeight();
    ErnieINT8EncoderWeight(const ErnieINT8EncoderWeight& other);
    ErnieINT8EncoderWeight& operator=(const ErnieINT8EncoderWeight& other);

    std::vector<ErnieINT8EncoderLayerWeight<T>*> ernie_encoder_layer_weights;
    LayerNormWeight<T>                    pre_transformer_layernorm_weights;
    T*                                    word_embedding_table                    = nullptr;
    T*                                    pos_embedding_table                     = nullptr;
    T*                                    sent_embedding_table                    = nullptr;
    PositionEmbeddingType                 position_embedding_type                 = PositionEmbeddingType::relative;

    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer);

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t vocab_size_;
    size_t pos_size_;
    size_t sent_vocab_size_;
    size_t num_layer_;
    // refer to num_buckt if using relative position embedding
    // refer to max_seq_len if using absolute position embedding

    bool is_maintain_buffer = false;

    int real_weights_num_;

    const static int weights_num_ = 5;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer