#include "ErnieEngine.h"

using namespace fastertransformer;

ErnieEngine::ErnieEngine(const CublasDataType data_type, const std::string& ckpt_path): data_type_(data_type)
{
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    m_.sm = prop.major * 10 + prop.minor;

    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle_));
#endif
    cublasSetStream(cublas_handle_, stream_);

    cublas_algo_map_ = new cublasAlgoMap("gemm_config.in", "");
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());
    cublas_wrapper_mutex_ = new std::mutex();

#ifdef SPARSITY_ENABLED
    cublas_wrapper_ = new cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, cusparselt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);
#else
    cublas_wrapper_ =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);
#endif
    if (data_type_ == HALF_DATATYPE) {
        FT_LOG_INFO("FP16");
        cublas_wrapper_->setFP16GemmConfig();
        ernie_weights_half_ = new ErnieEncoderWeight<half>(m_.head_num,
                                                          m_.size_per_head,
                                                          m_.d_model,
                                                          m_.inter_size,
                                                          m_.vocab_size,
                                                          m_.pos_size,
                                                          m_.sent_vocab_size,
                                                          m_.num_layer);

        m_.attention_type =
            getAttentionType<half>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len, true);
        ernie_weights_half_->loadModel(ckpt_path);

        ernie_half_ = new ErnieEncoder<half>(m_.max_batch_size,
                                            m_.max_seq_len,
                                            m_.head_num,
                                            m_.size_per_head,
                                            m_.inter_size,
                                            m_.d_model,
                                            m_.num_layer,
                                            m_.vocab_size,
                                            m_.pos_size,
                                            m_.sent_vocab_size,
                                            m_.sm,
                                            m_.q_scaling,
                                            stream_,  // stream_ placeholder
                                            cublas_wrapper_,
                                            allocator_,
                                            m_.is_free_buffer_after_forward,
                                            m_.attention_type,
                                            m_.is_sparse,
                                            m_.activation_type,
                                            m_.layernorm_type,
                                            NcclParam(0, 1),  // tensor_para
                                            NcclParam(0, 1)   // pipeline_para
        );
    }
#ifdef ENABLE_BF16
    else if (data_type_ == BFLOAT16_DATATYPE) {
        FT_LOG_INFO("BF16");
        cublas_wrapper_->setBF16GemmConfig();
        ernie_weights_bfloat_ = new ErnieEncoderWeight<__nv_bfloat16>(m_.head_num,
                                                                     m_.size_per_head,
                                                                     m_.d_model,
                                                                     m_.inter_size,
                                                                     m_.vocab_size,
                                                                     m_.pos_size,
                                                                     m_.sent_vocab_size,
                                                                     m_.num_layer);

        m_.attention_type = getAttentionType<__nv_bfloat16>(
            m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len, true);
        ernie_weights_bfloat_->loadModel(ckpt_path);

        ernie_bfloat_ = new ErnieEncoder<__nv_bfloat16>(m_.max_batch_size,
                                                       m_.max_seq_len,
                                                       m_.head_num,
                                                       m_.size_per_head,
                                                       m_.inter_size,
                                                       m_.d_model,
                                                       m_.num_layer,
                                                       m_.vocab_size,
                                                       m_.pos_size,
                                                       m_.sent_vocab_size,
                                                       m_.sm,
                                                       m_.q_scaling,
                                                       stream_,  // stream_ placeholder
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       m_.is_free_buffer_after_forward,
                                                       m_.attention_type,
                                                       m_.is_sparse,
                                                       m_.activation_type,
                                                       m_.layernorm_type,
                                                       NcclParam(0, 1),  // tensor_para
                                                       NcclParam(0, 1)   // pipeline_para
        );
    }
#endif
    else if (data_type_ == FLOAT_DATATYPE) {
        FT_LOG_INFO("FP32");
        cublas_wrapper_->setFP32GemmConfig();
        ernie_weights_float_ = new ErnieEncoderWeight<float>(m_.head_num,
                                                            m_.size_per_head,
                                                            m_.d_model,
                                                            m_.inter_size,
                                                            m_.vocab_size,
                                                            m_.pos_size,
                                                            m_.sent_vocab_size,
                                                            m_.num_layer);

        m_.attention_type =
            getAttentionType<float>(m_.size_per_head, getSMVersion(), m_.is_remove_padding, m_.max_seq_len, true);
        ernie_weights_float_->loadModel(ckpt_path);

        ernie_float_ = new ErnieEncoder<float>(m_.max_batch_size,
                                              m_.max_seq_len,
                                              m_.head_num,
                                              m_.size_per_head,
                                              m_.inter_size,
                                              m_.d_model,
                                              m_.num_layer,
                                              m_.vocab_size,
                                              m_.pos_size,
                                              m_.sent_vocab_size,
                                              m_.sm,
                                              m_.q_scaling,
                                              stream_,  // stream_ placeholder
                                              cublas_wrapper_,
                                              allocator_,
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
        throw std::runtime_error(std::string("[FT][ERROR] data_type should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

ErnieEngine::~ErnieEngine()
{
#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle_);
#endif
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;
}

void ErnieEngine::Run(std::unordered_map<std::string, Tensor>* output_tensors, const std::unordered_map<std::string, Tensor>* input_tensors)
{
    if (data_type_ == HALF_DATATYPE) {
        ernie_half_->forward(output_tensors, input_tensors, ernie_weights_half_);
    }
#ifdef ENABLE_BF16
    else if (data_type_ == BFLOAT16_DATATYPE) {
        ernie_bfloat_->forward(output_tensors, input_tensors, ernie_weights_bfloat_);
    }
#endif
    else if (data_type_ == FLOAT_DATATYPE) {
        ernie_float_->forward(output_tensors, input_tensors, ernie_weights_float_);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] data_type should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

cudaStream_t ErnieEngine::GetStream()
{
    return stream_;
}