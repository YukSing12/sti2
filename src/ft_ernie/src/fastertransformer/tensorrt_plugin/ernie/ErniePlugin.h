/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "ErniePluginGemm.h"
#include "src/fastertransformer/models/ernie/Ernie.h"
#include "src/fastertransformer/models/ernie/ErnieWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/logger.h"

#include <NvInfer.h>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>

// #define Ernie_PLUGIN_DEBUG
#ifdef Ernie_PLUGIN_DEBUG
#define WHERE_AM_I() printf("[%s]: this->%p\n", __func__, this);
#define PRINT_ENCODER(DATA_TYPE)                                                                                       \
    printf("[encoder::%s]Info:\n\tdatatype=%d\n", __func__, (int)DATA_TYPE);                                           \
    printf("\tmax_batch_size=%ld\n", m_.max_batch_size);                                                               \
    printf("\tmax_seq_len=%ld\n", m_.max_seq_len);                                                                     \
    printf("\tbeam_width=%ld\n", m_.beam_width);                                                                       \
    printf("\thead_num=%ld\n", m_.head_num);                                                                           \
    printf("\tsize_per_head=%ld\n", m_.size_per_head);                                                                 \
    printf("\td_model=%ld\n", m_.d_model);                                                                             \
    printf("\tinter_size=%ld\n", m_.inter_size);                                                                       \
    printf("\tnum_layer=%ld\n", m_.num_layer);                                                                         \
    printf("\tsm=%d\n", m_.sm);                                                                                        \
    printf("\tq_scaling=%f\n", m_.q_scaling);                                                                          \
    printf("\tuseFP16=%d\n", m_.useFP16);                                                                              \
    printf("\tvocab_size=%ld\n", m_.vocab_size);                                                                       \
    printf("\tpos_size=%ld\n", m_.pos_size);                                                                           \
    printf("\tsent_vocab_size=%ld\n", m_.sent_vocab_size);                                                             \
    printf("\tis_remove_padding=%d\n", (int)m_.is_remove_padding);                                                     \
    printf("\tis_free_buffer_after_forward=%d\n", (int)m_.is_free_buffer_after_forward);                               \
    printf("\tis_sparse=%d\n", (int)m_.is_sparse);                                                                     \
    printf("\tattention_type=%d\n", (int)m_.attention_type);                                                           \
    printf("\tactivation_type=%d\n", (int)m_.activation_type);                                                         \
    printf("\tlayernorm_type=%d\n", (int)m_.layernorm_type);                                                           \
    printf("\tbatch_size=%ld\n", m_.batch_size);                                                                       \
    printf("\tseq_len=%ld\n", m_.seq_len);                                                                             \
    printf("\tckpt_path=%s\n", m_.ckpt_path);

#define PRINT_DECODING(DATA_TYPE)                                                                                      \
    printf("[decoding::%s]Info:\n\tdatatype=%d\n", __func__, (int)DATA_TYPE);                                          \
    printf("\tmax_batch_size=%ld\n", m_.max_batch_size);                                                               \
    printf("\tmax_seq_len=%ld\n", m_.max_seq_len);                                                                     \
    printf("\tmem_max_seq_len=%ld\n", m_.mem_max_seq_len);                                                             \
    printf("\tbeam_width=%ld\n", m_.beam_width);                                                                       \
    printf("\thead_num=%ld\n", m_.head_num);                                                                           \
    printf("\tsize_per_head=%ld\n", m_.size_per_head);                                                                 \
    printf("\td_model=%ld\n", m_.d_model);                                                                             \
    printf("\tinter_size=%ld\n", m_.inter_size);                                                                       \
    printf("\tnum_layer=%ld\n", m_.num_layer);                                                                         \
    printf("\tvocab_size=%ld\n", m_.vocab_size);                                                                       \
    printf("\tpos_size=%ld\n", m_.pos_size);                                                                           \
    printf("\tsent_vocab_size=%ld\n", m_.sent_vocab_size);                                                             \
    printf("\tq_scaling=%f\n", m_.q_scaling);                                                                          \
    printf("\tstart_id=%d\n", m_.start_id);                                                                            \
    printf("\tend_id=%d\n", m_.end_id);                                                                                \
    printf("\tuseFP16=%d\n", m_.useFP16);                                                                              \
    printf("\tmem_d_model=%ld\n", m_.mem_d_model);                                                                     \
    printf("\tmem_hidden_units=%ld\n", m_.mem_hidden_units);                                                           \
    printf("\tis_free_buffer_after_forward=%d\n", m_.is_free_buffer_after_forward);                                    \
    printf("\tbatch_size=%ld\n", m_.batch_size);                                                                       \
    printf("\tseq_len=%ld\n", m_.seq_len);                                                                             \
    printf("\tckpt_path=%s\n", m_.ckpt_path);

#else
#define WHERE_AM_I()
#define PRINT_ENCODER(DATA_TYPE)
#define PRINT_DECODING(DATA_TYPE)
#endif  // DEBUG_ENABLE==1

namespace {
static const char* ENCODER_NAME{"ErniePlugin"};
static const char* ENCODER_VERSION{"1"};
}  // namespace

using namespace fastertransformer;

namespace nvinfer1 {
// class ErniePlugin ---------------------------------------------------------------------------
class ErniePlugin: public IPluginV2DynamicExt {
private:
    using IPluginV2::enqueue;
    using IPluginV2::getOutputDimensions;
    using IPluginV2::getWorkspaceSize;
    using IPluginV2Ext::configurePlugin;

    const std::string name_;
    std::string       namespace_;
    bool              is_own_weight = false;
    cublasHandle_t    cublasHandle_;
    cublasLtHandle_t  cublasltHandle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparseltHandle_;
#endif
    cublasAlgoMap*                      pCublasAlgoMap_             = nullptr;
    std::mutex*                         pCublasWrapperMutex_        = nullptr;
    ErnieWeight<half>*           pErnieWeightHalf_    = nullptr;
    ErnieWeight<float>*          pErnieWeightFloat_   = nullptr;
    Allocator<AllocatorType::CUDA>*     pAllocator_                 = nullptr;
    cublasMMWrapper*                    pCublasWrapper_             = nullptr;
    Ernie<half>*                 pErnieHalf_          = nullptr;
    Ernie<float>*                pErnieFloat_         = nullptr;
    struct {
        // constructor parameter
        size_t max_batch_size = 10;
        size_t max_seq_len    = 128;
        size_t beam_width     = 1;
        size_t head_num       = 12;
        size_t size_per_head  = 64;
        size_t d_model        = head_num * size_per_head;   // 768
        size_t inter_size     = d_model * 4;
        size_t num_layer      = 12;
        int    sm             = -1;  // assign later
        float  q_scaling      = 1.0f;
        bool   useFP16        = false;
        // internal parameter
        size_t                            vocab_size                   = 50000;
        size_t                            pos_size                     = 513;
        size_t                            sent_vocab_size              = 4;
        bool                              is_remove_padding            = true;
        bool                              is_free_buffer_after_forward = false;
        bool                              is_sparse                    = false;
        AttentionType                     attention_type               = AttentionType::FUSED_MHA;
        fastertransformer::ActivationType activation_type              = fastertransformer::ActivationType::Relu;
        LayerNormType                     layernorm_type               = LayerNormType::post_layernorm;
        // runtime parameter
        size_t batch_size     = 0;
        size_t seq_len        = 0;
        char   ckpt_path[256] = "";
    } m_;

public:
    ErniePlugin() = delete;
    ErniePlugin(const std::string& name,
                    size_t             max_batch_size,
                    size_t             max_seq_len,
                    size_t             beam_width,
                    size_t             head_num,
                    size_t             size_per_head,
                    size_t             d_model,
                    size_t             inter_size,
                    size_t             num_layer,
                    int                sm,
                    int                useFP16,
                    float              q_scaling,
                    const std::string& ckpt_path,
                    bool               own_weight);
    ErniePlugin(const std::string& name, const void* buffer, size_t length);
    ~ErniePlugin();

    virtual size_t       getSerializationSize() const noexcept override;
    virtual void         serialize(void* buffer) const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    int                  getNbOutputs() const noexcept override;
    DataType             getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    bool
    supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    DimsExprs    getOutputDimensions(int              index,
                                     const DimsExprs* pInputDim,
                                     int              nInputDim,
                                     IExprBuilder&    exprBuilder) noexcept override;
    virtual void configurePlugin(const DynamicPluginTensorDesc* in,
                                 int                            nbInput,
                                 const DynamicPluginTensorDesc* out,
                                 int                            nbOutput) noexcept override;
    size_t       getWorkspaceSize(const PluginTensorDesc* inputs,
                                  int32_t                 nbInputs,
                                  const PluginTensorDesc* outputs,
                                  int32_t                 nbOutputs) const noexcept override;
    int          enqueue(const PluginTensorDesc* inputDesc,
                         const PluginTensorDesc* outputDesc,
                         const void* const*      inputs,
                         void* const*            outputs,
                         void*                   workspace,
                         cudaStream_t            stream) noexcept override;
    void         setPluginNamespace(const char* szNamespace) noexcept override;
    const char*  getPluginNamespace() const noexcept override;
    const char*  getPluginType() const noexcept override;
    const char*  getPluginVersion() const noexcept override;
    int          initialize() noexcept override;
    void         terminate() noexcept override;
    void         destroy() noexcept override;
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) noexcept;
    void detachFromContext() noexcept;
};

// class ErniePluginCreator --------------------------------------------------------------------
class ErniePluginCreator: public IPluginCreator {
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    ErniePluginCreator();
    ~ErniePluginCreator();
    IPluginV2*  createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2*  deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void        setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
};

}  // namespace nvinfer1
