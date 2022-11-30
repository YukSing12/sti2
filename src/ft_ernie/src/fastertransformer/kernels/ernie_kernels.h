#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"

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
void invokeBuildErnieAttentionMask(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len, cudaStream_t stream);
}  // namespace fastertransformer