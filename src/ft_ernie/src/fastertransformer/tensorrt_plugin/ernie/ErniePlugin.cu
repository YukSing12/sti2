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

#include "ErniePlugin.h"

using namespace fastertransformer;

namespace nvinfer1 {

// class ErniePlugin ---------------------------------------------------------------------------
ErniePlugin::ErniePlugin(const std::string& name,
                                 size_t             max_batch_size,
                                 size_t             max_seq_len,
                                 size_t             beam_width,
                                 size_t             head_num,
                                 size_t             size_per_head,
                                 size_t             d_model,
                                 size_t             inter_size,
                                 size_t             num_layer,
                                 size_t             num_bucket,
                                 size_t             max_distance,
                                 int                sm,
                                 int                useFP16,
                                 float              q_scaling,
                                 const std::string& ckpt_path,
                                 bool               own_weight):
    name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();

    m_.max_batch_size = max_batch_size;
    m_.max_seq_len    = max_seq_len;
    m_.beam_width     = beam_width;
    m_.head_num       = head_num;
    m_.size_per_head  = size_per_head;
    m_.d_model        = d_model;
    m_.inter_size     = inter_size;
    m_.num_layer      = num_layer;
    m_.num_bucket     = num_bucket;
    m_.max_distance   = max_distance;
    m_.sm             = sm;
    m_.q_scaling      = q_scaling;
    m_.useFP16        = (bool)useFP16;
    m_.batch_size     = m_.max_batch_size;
    m_.seq_len        = m_.max_seq_len;
    strcpy(m_.ckpt_path, ckpt_path.c_str());

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtInit(&cusparseltHandle_);
#endif

    is_own_weight = own_weight;
    if (is_own_weight) {
        // Ernie EncoderWeight
        if (m_.useFP16) {
            m_.attention_type =
                getAttentionType<half>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len, false);
            pErnieEncoderWeightHalf_ = new ErnieEncoderWeight<half>(m_.head_num,
                                                              m_.size_per_head,
                                                              m_.d_model,
                                                              m_.inter_size,
                                                              m_.vocab_size,
                                                              m_.pos_size,
                                                              m_.sent_vocab_size,
                                                              m_.num_layer,
                                                              m_.num_bucket
            );
            pErnieEncoderWeightHalf_->loadModel(std::string(m_.ckpt_path));
        }
        else {
            m_.attention_type =
                getAttentionType<float>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len);
            pErnieEncoderWeightFloat_ = new ErnieEncoderWeight<float>(m_.head_num,
                                                                m_.size_per_head,
                                                                m_.d_model,
                                                                m_.inter_size,
                                                                m_.vocab_size,
                                                                m_.pos_size,
                                                                m_.sent_vocab_size,
                                                                m_.num_layer,
                                                                m_.num_bucket
            );
            pErnieEncoderWeightFloat_->loadModel(std::string(m_.ckpt_path));
        }
    }
    // Gemm file selection
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM") + std::to_string(m_.sm)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#ifdef Ernie_PLUGIN_DEBUG
        printf("Gemm file exist!\n");
#endif
    }
    else {
#ifdef Ernie_PLUGIN_DEBUG
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            (int)m_.max_batch_size,
            (int)m_.beam_width,                                                   // useless for encoder
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : (int)m_.seq_len,  // seq_len, in case of OOM
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.vocab_size,
            m_.useFP16 ? 1 : 0,  // is_fp16
            1,                   // tensor_para_size
            false                // always use fp32 compute type
        };
        ernie_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_      = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and ErnieEncoder
#ifdef SPARSITY_ENABLED
    pCublasWrapper_ = new cublasMMWrapper(
        cublasHandle_, cublasltHandle_, cusparseltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = true;
#else
    pCublasWrapper_ =
        new cublasMMWrapper(cublasHandle_, cublasltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = false;
#endif
    if (m_.useFP16) {
        pCublasWrapper_->setFP16GemmConfig();

        pErnieEncoderHalf_ = new ErnieEncoder<half>(m_.max_batch_size,
                                              m_.max_seq_len,
                                              m_.head_num,
                                              m_.size_per_head,
                                              m_.inter_size,
                                              m_.d_model,
                                              m_.num_layer,
                                              m_.num_bucket,
                                              m_.max_distance,
                                              m_.vocab_size,
                                              m_.pos_size,
                                              m_.sent_vocab_size,
                                              m_.sm,
                                              m_.q_scaling,
                                              0,  // stream placeholder
                                              pCublasWrapper_,
                                              pAllocator_,
                                              m_.is_free_buffer_after_forward,
                                              m_.attention_type,
                                              m_.is_sparse,
                                              m_.activation_type,
                                              m_.layernorm_type,
                                              NcclParam(0, 1),  // tensor_para
                                              NcclParam(0, 1)   // pipeline_para
        );
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pErnieEncoderFloat_ = new ErnieEncoder<float>(m_.max_batch_size,
                                                m_.max_seq_len,
                                                m_.head_num,
                                                m_.size_per_head,
                                                m_.inter_size,
                                                m_.d_model,
                                                m_.num_layer,
                                                m_.num_bucket,
                                                m_.max_distance,
                                                m_.vocab_size,
                                                m_.pos_size,
                                                m_.sent_vocab_size,
                                                m_.sm,
                                                m_.q_scaling,
                                                0,  // stream placeholder
                                                pCublasWrapper_,
                                                pAllocator_,
                                                m_.is_free_buffer_after_forward,
                                                m_.attention_type,
                                                m_.is_sparse,
                                                m_.activation_type,
                                                m_.layernorm_type,
                                                NcclParam(0, 1),  // tensor_para
                                                NcclParam(0, 1)   // pipeline_para
        );
    }
    PRINT_ENCODER(m_.useFP16)
}

ErniePlugin::ErniePlugin(const std::string& name, const void* buffer, size_t length): name_(name)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    memcpy(&m_, buffer, sizeof(m_));

    cublasCreate(&cublasHandle_);
    cublasLtCreate(&cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtInit(&cusparseltHandle_);
#endif

    is_own_weight = true;
    if (is_own_weight) {
        // Ernie EncoderWeight
        if (m_.useFP16) {
            m_.attention_type =
                getAttentionType<half>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len, false);
            pErnieEncoderWeightHalf_ = new ErnieEncoderWeight<half>(m_.head_num,
                                                              m_.size_per_head,
                                                              m_.d_model,
                                                              m_.inter_size,
                                                              m_.vocab_size,
                                                              m_.pos_size,
                                                              m_.sent_vocab_size,
                                                              m_.num_layer,
                                                              m_.num_bucket
            );
            pErnieEncoderWeightHalf_->loadModel(std::string(m_.ckpt_path));
        }
        else {
            m_.attention_type =
                getAttentionType<float>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len);
            pErnieEncoderWeightFloat_ = new ErnieEncoderWeight<float>(m_.head_num,
                                                                m_.size_per_head,
                                                                m_.d_model,
                                                                m_.inter_size,
                                                                m_.vocab_size,
                                                                m_.pos_size,
                                                                m_.sent_vocab_size,
                                                                m_.num_layer,
                                                                m_.num_bucket
            );
            pErnieEncoderWeightFloat_->loadModel(std::string(m_.ckpt_path));
        }
    }
    // Gemm file selection, in constructor, we use max_batch_szie and seq_len as
    // data size
    std::string gemmFileName = std::string(GEMM_CONFIG).substr(0, 11) + std::string("-SM") + std::to_string(m_.sm)
                               + std::string("-FP") + std::to_string(m_.useFP16 ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.batch_size) + std::string("-SL") + std::to_string(m_.seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
#ifdef Ernie_PLUGIN_DEBUG
        printf("Gemm file exist!\n");
#endif
    }
    else {
#ifdef Ernie_PLUGIN_DEBUG
        printf("Gemm file do not exist!\n");
#endif
        int argv[16] = {
            0,
            (int)m_.max_batch_size,
            (int)m_.beam_width,                                                   // useless for encoder
            (m_.batch_size == 128 && m_.seq_len == 384) ? 128 : (int)m_.seq_len,  // seq_len, in case of OOM
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.d_model,
            (int)m_.head_num,
            (int)m_.size_per_head,
            (int)m_.inter_size,
            (int)m_.vocab_size,
            m_.useFP16 ? 1 : 0,  // is_fp16
            1,                   // tensor_para_size
            false                // always use fp32 compute type
        };
        ernie_gemm(argv);
        rename(std::string(GEMM_CONFIG).c_str(), gemmFileName.c_str());
    }

    pCublasAlgoMap_      = new cublasAlgoMap(gemmFileName, "");
    pCublasWrapperMutex_ = new std::mutex();
    pAllocator_          = new Allocator<AllocatorType::CUDA>(getDevice());

    // cublas wrapper and ErnieEncoder
#ifdef SPARSITY_ENABLED
    pCublasWrapper_ = new cublasMMWrapper(
        cublasHandle_, cublasltHandle_, cusparseltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = true;
#else
    pCublasWrapper_ =
        new cublasMMWrapper(cublasHandle_, cublasltHandle_, 0, pCublasAlgoMap_, pCublasWrapperMutex_, pAllocator_);
    m_.is_sparse = false;
#endif
    if (m_.useFP16) {
        pCublasWrapper_->setFP16GemmConfig();

        pErnieEncoderHalf_ = new ErnieEncoder<half>(m_.max_batch_size,
                                              m_.max_seq_len,
                                              m_.head_num,
                                              m_.size_per_head,
                                              m_.inter_size,
                                              m_.d_model,
                                              m_.num_layer,
                                              m_.num_bucket,
                                              m_.max_distance,
                                              m_.vocab_size,
                                              m_.pos_size,
                                              m_.sent_vocab_size,
                                              m_.sm,
                                              m_.q_scaling,
                                              0,  // stream placeholder
                                              pCublasWrapper_,
                                              pAllocator_,
                                              m_.is_free_buffer_after_forward,
                                              m_.attention_type,
                                              m_.is_sparse,
                                              m_.activation_type,
                                              m_.layernorm_type,
                                              NcclParam(0, 1),  // tensor_para
                                              NcclParam(0, 1)   // pipeline_para
        );
    }
    else {
        pCublasWrapper_->setFP32GemmConfig();

        pErnieEncoderFloat_ = new ErnieEncoder<float>(m_.max_batch_size,
                                                m_.max_seq_len,
                                                m_.head_num,
                                                m_.size_per_head,
                                                m_.inter_size,
                                                m_.d_model,
                                                m_.num_layer,
                                                m_.num_bucket,
                                                m_.max_distance,
                                                m_.vocab_size,
                                                m_.pos_size,
                                                m_.sent_vocab_size,
                                                m_.sm,
                                                m_.q_scaling,
                                                0,  // stream placeholder
                                                pCublasWrapper_,
                                                pAllocator_,
                                                m_.is_free_buffer_after_forward,
                                                m_.attention_type,
                                                m_.is_sparse,
                                                m_.activation_type,
                                                m_.layernorm_type,
                                                NcclParam(0, 1),  // tensor_para
                                                NcclParam(0, 1)   // pipeline_para
        );
    }
    PRINT_ENCODER(m_.useFP16)
}

ErniePlugin::~ErniePlugin()
{
    WHERE_AM_I();
    if (is_own_weight && pErnieEncoderWeightHalf_ != nullptr) {
        delete pErnieEncoderWeightHalf_;
    }
    if (is_own_weight && pErnieEncoderWeightFloat_ != nullptr) {
        delete pErnieEncoderWeightFloat_;
    }
    if (pErnieEncoderHalf_ != nullptr) {
        delete pErnieEncoderHalf_;
    }
    if (pErnieEncoderFloat_ != nullptr) {
        delete pErnieEncoderFloat_;
    }
    delete pCublasAlgoMap_;
    delete pCublasWrapperMutex_;
    delete pCublasWrapper_;
    delete pAllocator_;
}

size_t ErniePlugin::getSerializationSize() const noexcept
{
    WHERE_AM_I();
    return sizeof(m_);
}

void ErniePlugin::serialize(void* buffer) const noexcept
{
    WHERE_AM_I();
    memcpy(buffer, &m_, sizeof(m_));
}

IPluginV2DynamicExt* ErniePlugin::clone() const noexcept
{
    WHERE_AM_I();
    auto p = new ErniePlugin(
        name_,
        m_.max_batch_size, 
        m_.max_seq_len, 
        m_.beam_width, 
        m_.head_num, 
        m_.size_per_head, 
        m_.d_model, 
        m_.inter_size, 
        m_.num_layer, 
        m_.num_bucket, 
        m_.max_distance, 
        m_.sm, 
        m_.useFP16, 
        m_.q_scaling,
        std::string(m_.ckpt_path), 
        false);
    p->setPluginNamespace(namespace_.c_str());
    p->pErnieEncoderWeightHalf_  = this->pErnieEncoderWeightHalf_;
    p->pErnieEncoderWeightFloat_ = this->pErnieEncoderWeightFloat_;
    return p;
}

int ErniePlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

DataType ErniePlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    WHERE_AM_I();
    return m_.useFP16 ? DataType::kHALF : DataType::kFLOAT;
}

bool ErniePlugin::supportsFormatCombination(int                     pos,
                                                const PluginTensorDesc* inOut,
                                                int                     nbInputs,
                                                int                     nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res = false;

    switch (pos) {
        case 0:
        case 1:
        case 2:
        case 3:
            res = (inOut[pos].type == DataType::kINT32)
                  && (inOut[pos].format == TensorFormat::kLINEAR);
            break;
        case 4:
            res = (inOut[pos].type == (m_.useFP16 ? DataType::kHALF : DataType::kFLOAT));
            break;
        default:  // should NOT be here!
            res = false;
    }
#ifdef Ernie_PLUGIN_DEBUG
    printf("Dim(");
    for (int i = 0; i < 3; i++) {
        printf("%d,", inOut[i].dims.nbDims);
    }
    printf("),");
    printf("pos=%d,res=%d,format(%d,%d,%d,%d),type(%d,%d,%d,%d),",
           pos,
           int(res),
           int(inOut[0].format),
           int(inOut[1].format),
           int(inOut[2].format),
           int(inOut[3].format),
           int(inOut[0].type),
           int(inOut[1].type),
           int(inOut[2].type),
           int(inOut[3].type));
    printf("kLINEAR=%d,float=%d,half=%d,int8=%d,int32=%d,bool=%d\n",
           int(TensorFormat::kLINEAR),
           int(DataType::kFLOAT),
           int(DataType::kHALF),
           int(DataType::kINT8),
           int(DataType::kINT32),
           int(DataType::kBOOL));
#endif
    return res;
}

DimsExprs ErniePlugin::getOutputDimensions(int              index,
                                               const DimsExprs* pInputDim,
                                               int              nInputDim,
                                               IExprBuilder&    exprBuilder) noexcept
{
    WHERE_AM_I();
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0]   = pInputDim[0].d[0];
    ret.d[1]   = pInputDim[0].d[1];
    ret.d[2]   = exprBuilder.constant(m_.d_model);
    return ret;
}

void ErniePlugin::configurePlugin(const DynamicPluginTensorDesc* in,
                                      int                            nbInput,
                                      const DynamicPluginTensorDesc* out,
                                      int                            nbOutput) noexcept
{
    WHERE_AM_I();
    PRINT_ENCODER(int(out[0].desc.type))
}

size_t ErniePlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
                                         int32_t                 nbInputs,
                                         const PluginTensorDesc* outputs,
                                         int32_t                 nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

void ErniePlugin::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* ErniePlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* ErniePlugin::getPluginType() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* ErniePlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

int ErniePlugin::initialize() noexcept
{
    WHERE_AM_I();
    return 0;
}

void ErniePlugin::terminate() noexcept
{
    WHERE_AM_I();
}

void ErniePlugin::destroy() noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    cublasDestroy(cublasHandle_);
    cublasLtDestroy(cublasltHandle_);
#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparseltHandle_);
#endif
    // delete this;
}

void ErniePlugin::attachToContext(cudnnContext* /*cudnn*/,
                                      cublasContext* /*cublas*/,
                                      IGpuAllocator* /*allocator*/) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
}

void ErniePlugin::detachFromContext() noexcept
{
    WHERE_AM_I();
}

int ErniePlugin::enqueue(const PluginTensorDesc* inputDesc,
                             const PluginTensorDesc* outputDesc,
                             const void* const*      inputs,
                             void* const*            outputs,
                             void*                   workspace,
                             cudaStream_t            stream) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    m_.batch_size = inputDesc[0].dims.d[0];
    m_.seq_len    = inputDesc[0].dims.d[1];
    PRINT_ENCODER(outputDesc[0].type)

    cublasSetStream(cublasHandle_, stream);
    pCublasWrapper_->setStream(stream);

    std::unordered_map<std::string, Tensor> inputTensor{
        {"word_ids", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size, m_.seq_len}, (int*)inputs[0]}},
        {"pos_ids", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size, m_.seq_len}, (int*)inputs[1]}},
        {"sent_ids", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size, m_.seq_len}, (int*)inputs[2]}},
        {"seq_len", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{m_.batch_size}, (int*)inputs[3]}}};
    if (m_.useFP16) {
        std::unordered_map<std::string, Tensor> outputTensor{
            {"attn_out",
             Tensor{MEMORY_GPU,
                    TYPE_FP16,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, (size_t)(m_.head_num * m_.size_per_head)},
                    (half*)outputs[0]}}};
        pErnieEncoderHalf_->setStream(stream);
        pErnieEncoderHalf_->forward(&outputTensor, &inputTensor, pErnieEncoderWeightHalf_);
    }
    else {
        std::unordered_map<std::string, Tensor> outputTensor{
            {"attn_out",
             Tensor{MEMORY_GPU,
                    TYPE_FP32,
                    std::vector<size_t>{m_.batch_size, m_.seq_len, (size_t)(m_.head_num * m_.size_per_head)},
                    (float*)outputs[0]}}};
        pErnieEncoderFloat_->setStream(stream);
        pErnieEncoderFloat_->forward(&outputTensor, &inputTensor, pErnieEncoderWeightFloat_);
    }
    return 0;
}

// class ErniePluginCreator --------------------------------------------------------------------
PluginFieldCollection    ErniePluginCreator::fc_{};
std::vector<PluginField> ErniePluginCreator::attr_;

ErniePluginCreator::ErniePluginCreator()
{
    WHERE_AM_I();
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
}

ErniePluginCreator::~ErniePluginCreator()
{
    WHERE_AM_I();
}

IPluginV2* ErniePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    WHERE_AM_I();
    int         max_batch_size = 10;
    int         max_seq_len    = 128;
    int         beam_width     = 1;
    int         head_num       = 12;
    int         size_per_head  = 768 / 12;
    int         d_model        = head_num * size_per_head;
    int         inter_size     = d_model * 4;
    int         num_layer      = 12;
    int         num_bucket     = 128;
    int         max_distance   = 128;
    int         sm             = -1;
    float       q_scaling      = 0.125f;
    int         useFP16        = 0;
    std::string ckpt_path      = std::string("/workspace/xys/sti2/model/bin");

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm = prop.major * 10 + prop.minor;

    std::map<std::string, int*> name2p{
        {"max_batch_size", &max_batch_size},
        {"max_seq_len", &max_seq_len},
        {"beam_width", &beam_width},
        {"head_num", &head_num},
        {"size_per_head", &size_per_head},
        {"d_model", &d_model},
        {"inter_size", &inter_size},
        {"num_layer", &num_layer},
        {"num_bucket", &num_bucket},
        {"max_distance", &max_distance},
        {"sm", &sm},
        {"useFP16", &useFP16},
    };
    for (int i = 0; i < fc->nbFields; i++) {
        if (name2p.find(fc->fields[i].name) != name2p.end()) {
            *name2p[fc->fields[i].name] = *(int*)fc->fields[i].data;
        }
        else if (!strcmp(fc->fields[i].name, "ckpt_path")) {
            ckpt_path = std::string((char*)fc->fields[i].data);
        }else if (!strcmp(fc->fields[i].name, "q_scaling")) {
            q_scaling = *(float*)fc->fields[i].data;
        }
    }
    return new ErniePlugin(name, max_batch_size, max_seq_len, beam_width, head_num, size_per_head, d_model, inter_size, num_layer, num_bucket, max_distance, sm, useFP16, q_scaling, ckpt_path, true);
}

IPluginV2*
ErniePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    WHERE_AM_I();
    return new ErniePlugin(name, serialData, serialLength);
}

void ErniePluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    WHERE_AM_I();
    namespace_ = szNamespace;
}

const char* ErniePluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return namespace_.c_str();
}

const char* ErniePluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return ENCODER_NAME;
}

const char* ErniePluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return ENCODER_VERSION;
}

const PluginFieldCollection* ErniePluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &fc_;
}

REGISTER_TENSORRT_PLUGIN(ErniePluginCreator);
}  // namespace nvinfer1
