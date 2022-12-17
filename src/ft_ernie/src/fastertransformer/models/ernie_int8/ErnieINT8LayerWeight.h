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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/models/ernie/ErnieLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct ErnieINT8LayerWeight: ErnieLayerWeight<T>{

    ErnieINT8LayerWeight() = default;
    ErnieINT8LayerWeight(const size_t layer_id,
                            const size_t head_num,
                            const size_t size_per_head,
                            const size_t d_model,
                            const size_t inter_size);
    ~ErnieINT8LayerWeight();
    ErnieINT8LayerWeight(const ErnieINT8LayerWeight& other);
    ErnieINT8LayerWeight& operator=(const ErnieINT8LayerWeight& other);

    AttentionINT8Weight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    FfnINT8Weight<T>       ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;
    ScaleList              scale_list_;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t layer_id_;
    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;

    int real_weights_num_;

    bool is_maintain_buffer = false;

    // Assume bias added, and gated activation used
    const static int weights_num_ = 16;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
    float* scale_list_ptr[2];
    T*   sp_weights_ptr[6];
    bool is_maintain_sp_buffer = false;
};

}  // namespace fastertransformer
