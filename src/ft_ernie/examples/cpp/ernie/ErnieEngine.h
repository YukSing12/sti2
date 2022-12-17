#pragma once
#include "ErnieGemm.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include "src/fastertransformer/models/ernie/Ernie.h"
#include "src/fastertransformer/models/ernie/ErnieWeight.h"
#include "src/fastertransformer/models/ernie_int8/ErnieINT8.h"
#include "src/fastertransformer/models/ernie_int8/ErnieINT8Weight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cublasINT8MMWrapper.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>

using namespace fastertransformer;

template<typename T>
class ErnieEngine {
private:
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle_;
#endif
    cublasAlgoMap* cublas_algo_map_ = nullptr;
    Allocator<AllocatorType::CUDA>* allocator_ = nullptr;
    std::mutex* cublas_wrapper_mutex_ = nullptr;
    cublasMMWrapper* cublas_wrapper_ = nullptr;
    cublasINT8MMWrapper* cublas_wrapper_int8_ = nullptr;

    ErnieWeight<T>* ernie_weights_ = nullptr;
    Ernie<T>* ernie_ = nullptr;

    ErnieINT8Weight<T>* ernie_weights_int8_ = nullptr;
    ErnieINT8<T>* ernie_int8_ = nullptr;

    bool int8_mode_ = false;
    bool useCudaGraph_ = false;
    struct {
        // constructor parameter
        size_t max_batch_size = 10;
        size_t max_seq_len = 128;
        size_t beam_width = 1;
        size_t head_num = 12;
        size_t size_per_head = 64;
        size_t d_model = head_num * size_per_head;  // 768
        size_t inter_size = d_model * 4;            // 3072
        size_t num_layer = 12;
        int sm = -1;  // assign later
        float q_scaling = 1.0f;
        // internal parameter
        size_t vocab_size = 50000;
        size_t pos_size = 513;
        size_t sent_vocab_size = 4;
        bool is_remove_padding = true;
        bool is_free_buffer_after_forward = false;
        bool is_sparse = false;
        AttentionType attention_type = AttentionType::FUSED_MHA;
        ActivationType activation_type = ActivationType::Relu;
        LayerNormType layernorm_type = LayerNormType::post_layernorm;
        // runtime parameter
        size_t batch_size = 0;
        size_t seq_len = 0;
    } m_;

public:
    ErnieEngine(const std::string& ckpt_path, const bool int8_mode, const bool useCudaGraph);
    ErnieEngine() = delete;
    ErnieEngine(const ErnieEngine&) = delete;
    ErnieEngine& operator=(const ErnieEngine&) = delete;

    ~ErnieEngine();

    void run(const int* h_word_ids_,
             const int* h_pos_ids_,
             const int* h_sent_ids_,
             const int* h_seq_len_,
             const int* h_multi_ids_,
             const int request_batch_size,
             const int request_seq_len);
    void copyToCpu(float* h_attn_out, const int request_batch_size);

    void runInt8(const int* h_word_ids_,
             const int* h_pos_ids_,
             const int* h_sent_ids_,
             const int* h_seq_len_,
             const int* h_multi_ids_,
             const int request_batch_size,
             const int request_seq_len);
    void copyToCpuInt8(float* h_attn_out, const int request_batch_size);

    cudaStream_t getStream()
    {
        return stream_;
    }
    size_t getMaxBatch() const
    {
        return m_.max_batch_size;
    }
    size_t getMaxSeqLen() const
    {
        return m_.max_seq_len;
    }
};