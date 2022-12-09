#include "ernie_kernels.h"

namespace fastertransformer {
template<typename T, int TPB>
__global__ void embeddingLookupConcatKernel(const int ld,
                                            const int vocab_size,
                                            const int pos_size,
                                            const int sent_vocab_size,
                                            const T* sendEmb,
                                            const T* wordEmb,
                                            const T* posEmb,
                                            const int32_t* sendIds,
                                            const int32_t* wordIds,
                                            const int32_t* posIds,
                                            T* output)  // const half* gamma, const half* beta,
{
    __shared__ int wordId;
    __shared__ int sendId;
    __shared__ int posId;

    int32_t const seqPos = blockIdx.y + blockIdx.x * gridDim.y;
    if (threadIdx.x == 0) {
        wordId = (int)wordIds[seqPos];
        sendId = (int)sendIds[seqPos];
        posId = (int)posIds[seqPos];
    }
    __syncthreads();

    int32_t const poffset = posId * ld;
    int32_t const woffset = wordId * ld;
    int32_t const toffset = sendId * ld;
    int32_t const outOffset = seqPos * ld;

    if (wordId >= 0 && wordId < vocab_size && sendId >= 0 && sendId < sent_vocab_size && posId >= 0
        && posId < pos_size) {
        for (int it = threadIdx.x; it < ld; it += TPB) {
            T val = wordEmb[woffset + it] + sendEmb[toffset + it] + posEmb[poffset + it];
            output[outOffset + it] = val;
        }
    }
}

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
                                 cudaStream_t stream)
{
    dim3 grid(batch_size * seq_len);
    FT_CHECK(ld == 768);
    embeddingLookupConcatKernel<T, 768><<<grid, 768, 0, stream>>>(
        ld, vocab_size, pos_size, sent_vocab_size, sendEmb, wordEmb, posEmb, sendIds, wordIds, posIds, from_tensor);
}

template void invokeEmbeddingLookupConcat(float* from_tensor,
                                          const int ld,
                                          const int batch_size,
                                          const int seq_len,
                                          const int vocab_size,
                                          const int pos_size,
                                          const int sent_vocab_size,
                                          const float* sendEmb,
                                          const float* wordEmb,
                                          const float* posEmb,
                                          const int32_t* sendIds,
                                          const int32_t* wordIds,
                                          const int32_t* posIds,
                                          cudaStream_t stream);

template void invokeEmbeddingLookupConcat(half* from_tensor,
                                          const int ld,
                                          const int batch_size,
                                          const int seq_len,
                                          const int vocab_size,
                                          const int pos_size,
                                          const int sent_vocab_size,
                                          const half* sendEmb,
                                          const half* wordEmb,
                                          const half* posEmb,
                                          const int32_t* sendIds,
                                          const int32_t* wordIds,
                                          const int32_t* posIds,
                                          cudaStream_t stream);
}  // namespace fastertransformer