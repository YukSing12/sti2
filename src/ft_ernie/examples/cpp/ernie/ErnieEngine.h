#pragma once
#include "src/fastertransformer/models/ernie/ErnieEncoder.h"
#include "src/fastertransformer/models/ernie/ErnieEncoderWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <NvInfer.h>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <unordered_map>

using namespace fastertransformer;

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
    CublasDataType data_type_;

    ErnieEncoderWeight<half>* ernie_weights_half_ = nullptr;
    ErnieEncoderWeight<float>* ernie_weights_float_ = nullptr;
    ErnieEncoderWeight<__nv_bfloat16>* ernie_weights_bfloat_ = nullptr;

    ErnieEncoder<half>* ernie_half_ = nullptr;
    ErnieEncoder<float>* ernie_float_ = nullptr;
    ErnieEncoder<__nv_bfloat16>* ernie_bfloat_ = nullptr;
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
    explicit ErnieEngine(const CublasDataType data_type, const std::string& ckpt_path);
    ErnieEngine() = delete;
    ErnieEngine(const ErnieEngine&) = delete;
    ErnieEngine& operator=(const ErnieEngine&) = delete;

    ~ErnieEngine();

    void Run(std::unordered_map<std::string, Tensor>* output_tensors, const std::unordered_map<std::string, Tensor>* input_tensors);
    cudaStream_t GetStream();
};