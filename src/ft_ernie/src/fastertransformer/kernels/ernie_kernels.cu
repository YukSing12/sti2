#include "ernie_kernels.h"

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
    cub::Sum pairSum;

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

template<typename T>
__global__ void
buildErnieAttentionMask(T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len)
{
    // sequence_lengths: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len, max_seq_len]
    attention_mask += blockIdx.x * max_seq_len * max_seq_len;
    const int length = sequence_lengths[blockIdx.x];
    for (int i = threadIdx.x; i < max_seq_len * max_seq_len; i += blockDim.x) {
        // int row_id = i / max_seq_len;
        int col_id = i % max_seq_len;
        // if (row_id < length && col_id < length) {
        // TODO (bhsueh) check this modification is ok or not on other rmodel
        if (col_id < length) {
            attention_mask[i] = (T)(1.0f);
        }
        else {
            attention_mask[i] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokeBuildErnieAttentionMask(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len, cudaStream_t stream)
{
    FT_CHECK(max_seq_len == 128);
    buildErnieAttentionMask<<<batch_size, 128, 0, stream>>>(attention_mask, sequence_lengths, max_seq_len);
}

template void invokeBuildErnieAttentionMask(float* attention_mask,
                                            const int* sequence_lengths,
                                            const int batch_size,
                                            const int max_seq_len,
                                            cudaStream_t stream);

template void invokeBuildErnieAttentionMask(half* attention_mask,
                                            const int* sequence_lengths,
                                            const int batch_size,
                                            const int max_seq_len,
                                            cudaStream_t stream);