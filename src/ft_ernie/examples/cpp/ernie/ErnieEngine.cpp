#include "ErnieEngine.h"

using namespace fastertransformer;

template<typename T>
ErnieEngine<T>::ErnieEngine(const std::string& ckpt_path, const bool int8_mode, const bool useCudaGraph):
    int8_mode_(int8_mode), useCudaGraph_(useCudaGraph)
{
    m_.sm = getSMVersion();
    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle_));
#endif
    cublasSetStream(cublas_handle_, stream_);
    std::string init_name;
    if (int8_mode_) {
        init_name = std::string("gemmInt8_config");
    }
    else {
        init_name = std::string("gemm_config");
    }
    std::string gemmFileName = init_name + std::string("-SM") + std::to_string(m_.sm) + std::string("-FP")
                               + std::to_string(std::is_same<T, half>::value ? 16 : 32) + std::string("-BS")
                               + std::to_string(m_.max_batch_size) + std::string("-SL") + std::to_string(m_.max_seq_len)
                               + std::string("-BM") + std::to_string(m_.beam_width) + std::string("")
                               + std::string(".in");
    std::ifstream infile(gemmFileName);
    if (infile.good()) {
        printf("[INFO] Gemm file exist!\n");
    }
    else {
        printf("[Warning] Gemm file do not exist!\n");
        for (size_t b = 1; b <= m_.max_batch_size; b++) {
            for (size_t l = 16; l <= m_.max_seq_len; l++) {
                int argv[8] = {
                    0,
                    (int)b,
                    (int)l,  // seq_len, in case of OOM
                    (int)m_.head_num,
                    (int)m_.size_per_head,
                    std::is_same<T, half>::value ? 1 : 0,  // // 0 FP32, 1 FP16
                    (int)int8_mode_,                       // int8 mode
                    1,                                     // tensor_para_size
                };
                ernie_gemm(argv);
            }
        }
        rename(std::string("gemm_config.in").c_str(), gemmFileName.c_str());
    }
    cublas_algo_map_ = new cublasAlgoMap(gemmFileName, "");
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());
    cublas_wrapper_mutex_ = new std::mutex();
    bool use_ORDER_COL32_2R_4R4 = false;

#if (CUDART_VERSION >= 11000)
    if (m_.sm >= 80) {
        use_ORDER_COL32_2R_4R4 = true;
    }
#endif

#ifdef SPARSITY_ENABLED
    if (int8_mode_) {
        cublas_wrapper_int8_ = cublasINT8MMWrapper(cublaslt_handle_,
                                                   cusparselt_handle_,
                                                   stream_,
                                                   cublas_algo_map_,
                                                   cublas_wrapper_mutex_,
                                                   use_ORDER_COL32_2R_4R4);
    }
    else {
        cublas_wrapper_ = new cublasMMWrapper(cublas_handle_,
                                              cublaslt_handle_,
                                              cusparselt_handle_,
                                              stream_,
                                              cublas_algo_map_,
                                              cublas_wrapper_mutex_,
                                              allocator_);
    }
#else
    if (int8_mode_) {
        cublas_wrapper_int8_ = new cublasINT8MMWrapper(
            cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, use_ORDER_COL32_2R_4R4);
    }
    else {
        cublas_wrapper_ = new cublasMMWrapper(
            cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);
    }

#endif
    if (!int8_mode_) {
        if (std::is_same<T, half>::value) {
            cublas_wrapper_->setFP16GemmConfig();
        }
        else {
            cublas_wrapper_->setFP32GemmConfig();
        }
        ernie_ = new Ernie<T>(m_.max_batch_size,
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
        ernie_->setUseGraph(useCudaGraph_);
        ernie_->setHostMode(true);
        ernie_weights_ = new ErnieWeight<T>(m_.head_num,
                                            m_.size_per_head,
                                            m_.d_model,
                                            m_.inter_size,
                                            m_.vocab_size,
                                            m_.pos_size,
                                            m_.sent_vocab_size,
                                            m_.num_layer);

        ernie_weights_->loadModel(ckpt_path);
        m_.attention_type = getAttentionType<T>(m_.size_per_head, m_.sm, m_.is_remove_padding, m_.max_seq_len, true);
    }
    else {
        ernie_int8_ = new ErnieINT8<T>(m_.max_batch_size,
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
                                       int8_mode_,
                                       stream_,  // stream_ placeholder
                                       cublas_wrapper_int8_,
                                       allocator_,
                                       m_.is_free_buffer_after_forward,
                                       m_.attention_type,
                                       m_.is_sparse,
                                       m_.layernorm_type);
        ernie_int8_->setUseGraph(useCudaGraph_);
        ernie_int8_->setHostMode(true);
        ernie_weights_int8_ = new ErnieINT8Weight<T>(m_.head_num,
                                                     m_.size_per_head,
                                                     m_.d_model,
                                                     m_.inter_size,
                                                     m_.vocab_size,
                                                     m_.pos_size,
                                                     m_.sent_vocab_size,
                                                     m_.num_layer);

        ernie_weights_int8_->loadModel(ckpt_path);
        m_.attention_type =
            getAttentionTypeINT8<T>(m_.size_per_head, m_.sm, m_.is_remove_padding, m_.max_seq_len, int8_mode_);
    }
}

template<typename T>
ErnieEngine<T>::~ErnieEngine()
{
    if (ernie_weights_ != nullptr) {
        delete ernie_weights_;
    }
    if (ernie_weights_ != nullptr) {
        delete ernie_weights_;
    }
    if (ernie_weights_int8_ != nullptr) {
        delete ernie_weights_int8_;
    }
    if (ernie_int8_ != nullptr) {
        delete ernie_int8_;
    }

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle_);
#endif
    delete cublas_algo_map_;
    delete cublas_wrapper_;
    delete cublas_wrapper_mutex_;
    delete allocator_;
    delete cublas_wrapper_int8_;
}

template<typename T>
void ErnieEngine<T>::run(const int* h_word_ids_,
                         const int* h_pos_ids_,
                         const int* h_sent_ids_,
                         const int* h_seq_len_,
                         const int* h_multi_ids_,
                         const int request_batch_size,
                         const int request_seq_len)
{
    ernie_->forward(h_word_ids_,
                    h_pos_ids_,
                    h_sent_ids_,
                    h_seq_len_,
                    h_multi_ids_,
                    request_batch_size,
                    request_seq_len,
                    ernie_weights_);
}
template<typename T>
void ErnieEngine<T>::runInt8(const int* h_word_ids_,
                             const int* h_pos_ids_,
                             const int* h_sent_ids_,
                             const int* h_seq_len_,
                             const int* h_multi_ids_,
                             const int request_batch_size,
                             const int request_seq_len)
{
    ernie_int8_->forward(h_word_ids_,
                         h_pos_ids_,
                         h_sent_ids_,
                         h_seq_len_,
                         h_multi_ids_,
                         request_batch_size,
                         request_seq_len,
                         ernie_weights_int8_);
}

template<typename T>
void ErnieEngine<T>::copyToCpu(float* h_attn_out, const int request_batch_size)
{
    ernie_->copyToCpu(h_attn_out, request_batch_size);
}

template<typename T>
void ErnieEngine<T>::copyToCpuInt8(float* h_attn_out, const int request_batch_size)
{
    ernie_int8_->copyToCpu(h_attn_out, request_batch_size);
}

template class ErnieEngine<float>;
template class ErnieEngine<half>;