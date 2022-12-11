#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/kernels/activation_kernels.h"

namespace fastertransformer {
template<typename T>
void invokeEmbeddingLookupConcat(T* from_tensor,
                                 const int ld,
                                 const int batch_size,
                                 const int seq_len,
                                 const int vocab_size,
                                 const int pos_size,
                                 const int sent_vocab_size,
                                 const T* sendEmb,
                                 const T* wordEmb,
                                 const T* posEmb,
                                 const int32_t* sendIds,
                                 const int32_t* wordIds,
                                 const int32_t* posIds,
                                 cudaStream_t stream);

template<typename T>
void invokeSlice(T* dst, const T* src, const int batch_size, const int seq_len, const int d_model, cudaStream_t stream);

template<typename T>
void invokeAddBiasTanh(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokePostEmbedding(const int* x,
                         const T* emb_0,
                         const T* emb_1,
                         const T* emb_2,
                         const T* emb_3,
                         const T* emb_4,
                         const T* emb_5,
                         const T* emb_6,
                         const T* emb_7,
                         T* output,
                         const int batch_size,
                         cudaStream_t stream);

template<typename T>
void invokeAddTwoAddBiasSigmoid(const T* input0,
                                const T* input1,
                                const T* bias0,
                                const T* bias1,
                                float* output,
                                const int batch_size,
                                cudaStream_t stream);
}  // namespace fastertransformer