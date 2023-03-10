#include "ernie_kernels.h"
#include "src/fastertransformer/kernels/bfloat16_fallback_kenrels.cuh"

namespace fastertransformer {
    
__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return (1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f);
#endif
}

__inline__ __device__ half tanh_opt(half x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    half r;
    asm("tanh.approx.f16 %0,%1; \n\t" : "=h"(__HALF_TO_US(r)) : "h"(__HALF_TO_US(x)));
    return r;
#else
    return __float2half(tanh_opt(__half2float(x)));
#endif
}

__inline__ __device__ half2 tanh_opt(half2 x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    half2 r;
    asm("tanh.approx.f16x2 %0,%1; \n\t" : "=r"(__HALF2_TO_UI(r)) : "r"(__HALF2_TO_UI(x)));
    return r;
#else
    half2 y;
    y.x = __float2half(tanh_opt(__half2float(x.x)));
    y.y = __float2half(tanh_opt(__half2float(x.y)));
    return y;
#endif
    
}

__inline__ __device__ __nv_bfloat16 tanh_opt(__nv_bfloat16 x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    __nv_bfloat16 r;
    asm("tanh.approx.bf16 %0,%1; \n\t" : "=h"(__BFLOAT16_TO_US(r)) : "h"(__BFLOAT16_TO_US(x)));
    return r;
#else
    return __float2bfloat16(tanh_opt(__bfloat162float(x)));
#endif
}

__inline__ __device__ __nv_bfloat162 tanh_opt(__nv_bfloat162 x)
{
#if (__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 11000)
    __nv_bfloat162 r;
    asm("tanh.approx.bf16x2 %0,%1; \n\t" : "=r"(__BFLOAT162_TO_UI(r)) : "r"(__BFLOAT162_TO_UI(x)));
    return r;
#else
    __nv_bfloat162 y;
    y.x = __float2bfloat16(tanh_opt(__bfloat162float(x.x)));
    y.y = __float2bfloat16(tanh_opt(__bfloat162float(x.y)));
    return y;
#endif
}

template<typename T, int TPB>
__global__ void embeddingLookupConcatKernel(const int ld,
                                            const int vocab_size,
                                            const int pos_size,
                                            const int sent_vocab_size,
                                            const T* __restrict sendEmb,
                                            const T* __restrict wordEmb,
                                            const T* __restrict posEmb,
                                            const int32_t* __restrict sendIds,
                                            const int32_t* __restrict wordIds,
                                            const int32_t* __restrict posIds,
                                            T* output)  // const half* gamma, const half* beta,
{
    __shared__ int wordId;
    __shared__ int sendId;
    __shared__ int posId;

    int32_t const seqPos = blockIdx.y + blockIdx.x * gridDim.y;
    if (threadIdx.x == 0) {
        wordId = (int)ldg(&wordIds[seqPos]);
        sendId = (int)ldg(&sendIds[seqPos]);
        posId = (int)ldg(&posIds[seqPos]);
    }
    __syncthreads();

    int32_t const poffset = posId * ld;
    int32_t const woffset = wordId * ld;
    int32_t const toffset = sendId * ld;
    int32_t const outOffset = seqPos * ld;

    if (wordId >= 0 && wordId < vocab_size && sendId >= 0 && sendId < sent_vocab_size && posId >= 0
        && posId < pos_size) {
        for (int it = threadIdx.x; it < ld; it += TPB) {
            T val = ldg(&wordEmb[woffset + it]) + ldg(&sendEmb[toffset + it]) + ldg(&posEmb[poffset + it]);
            output[outOffset + it] = val;
        }
    }
}
__global__ void ErniegetPaddingOffsetKernel(size_t*    valid_word_num,
                                            int*       tmp_mask_offset,
                                            const int* sequence_length,
                                            const int  batch_size,
                                            const int  max_seq_len)
{
    // do cumulated sum
    int total_seq_len = 0;
    int cum_offset    = 0;
    int index         = 0;
    for (int i = 0; i < batch_size; i++) {
        const int seq_len = sequence_length[i];
        for (int j = 0; j < seq_len; j++) {
                tmp_mask_offset[index] = cum_offset;
                index++;
            }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    valid_word_num[0] = (size_t)total_seq_len;
}

void invokeGetPaddingOffsetErnie(size_t*      d_token_num,
                            int*         tmp_mask_offset,
                            const int*   sequence_lengths,
                            const int    batch_size,
                            const int    max_seq_len,
                            cudaStream_t stream)
{
    ErniegetPaddingOffsetKernel<<<1, 1, 0, stream>>>(d_token_num, tmp_mask_offset, sequence_lengths, batch_size, max_seq_len);
    sync_check_cuda_error();
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
#ifdef ENABLE_BF16
template void invokeEmbeddingLookupConcat(__nv_bfloat16* from_tensor,
                                          const int ld,
                                          const int batch_size,
                                          const int seq_len,
                                          const int vocab_size,
                                          const int pos_size,
                                          const int sent_vocab_size,
                                          const __nv_bfloat16* sendEmb,
                                          const __nv_bfloat16* wordEmb,
                                          const __nv_bfloat16* posEmb,
                                          const int32_t* sendIds,
                                          const int32_t* wordIds,
                                          const int32_t* posIds,
                                          cudaStream_t stream);
#endif
template<typename T>
__global__ void slice(T* dst, const T* __restrict src, const int batch_size, const int seq_len, const int d_model)
{
    // TODO: vectorize
    const int src_idx = blockIdx.x * seq_len * d_model + 0 * d_model + threadIdx.x;
    const int dst_idx = blockIdx.x * d_model + threadIdx.x;
    dst[dst_idx] = ldg(&src[src_idx]);
}

template<typename T>
void invokeSlice(T* dst, const T* src, const int batch_size, const int seq_len, const int d_model, cudaStream_t stream)
{
    FT_CHECK(d_model == 768);
    slice<<<batch_size, d_model, 0, stream>>>(dst, src, batch_size, seq_len, d_model);
}

template void invokeSlice(
    float* dst, const float* src, const int batch_size, const int seq_len, const int d_model, cudaStream_t stream);

template void invokeSlice(
    half* dst, const half* src, const int batch_size, const int seq_len, const int d_model, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeSlice(__nv_bfloat16* dst,
                          const __nv_bfloat16* src,
                          const int batch_size,
                          const int seq_len,
                          const int d_model,
                          cudaStream_t stream);
#endif

template<typename T>
__global__ void addBiasTanh(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            T reg_bias = __ldg(&bias[id % n]);
            val = val + reg_bias;
        }
        out[id] = (T)(tanh_opt(val));
    }
}

template<>
__global__ void addBiasTanh(half* out, const half* __restrict bias, int m, int n)
{
    half2* out_ptr = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            half2 reg_bias = __ldg(&bias_ptr[id % n]);
            val = __hadd2(val, reg_bias);
        }
        out_ptr[id] = tanh_opt(val);
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void addBiasTanh(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162* out_ptr = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = out_ptr[id];
        if (bias != nullptr) {
            __nv_bfloat162 reg_bias = ldg(&bias_ptr[id % n]);
            val = bf16hadd2(val, reg_bias);
        }
        out_ptr[id] = tanh_opt(val);
    }
}
#endif

template<typename T>
void invokeAddBiasTanh(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = ceil(m * n / 1024.);
    }
    addBiasTanh<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasTanh(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasTanh(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasTanh(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

template<typename T, int TPB, int VPT, int LEN>
__global__ void postEmbedding(const int* x,
                              const T* __restrict emb_0,
                              const T* __restrict emb_1,
                              const T* __restrict emb_2,
                              const T* __restrict emb_3,
                              const T* __restrict emb_4,
                              const T* __restrict emb_5,
                              const T* __restrict emb_6,
                              const T* __restrict emb_7,
                              T* output)
{
    // shared memory
    __shared__ int local_x_0;
    __shared__ int local_x_1;
    __shared__ int local_x_2;
    __shared__ int local_x_3;
    __shared__ int local_x_4;
    __shared__ int local_x_5;
    __shared__ int local_x_6;
    __shared__ int local_x_7;
    if (threadIdx.x == 0) {
        local_x_0 = x[0 + blockIdx.x * 8] * LEN;
        local_x_1 = x[1 + blockIdx.x * 8] * LEN;
        local_x_2 = x[2 + blockIdx.x * 8] * LEN;
        local_x_3 = x[3 + blockIdx.x * 8] * LEN;
        local_x_4 = x[4 + blockIdx.x * 8] * LEN;
        local_x_5 = x[5 + blockIdx.x * 8] * LEN;
        local_x_6 = x[6 + blockIdx.x * 8] * LEN;
        local_x_7 = x[7 + blockIdx.x * 8] * LEN;
    }
    __syncthreads();

    if (threadIdx.x < LEN) {
        const int batch_idx = blockIdx.x * 8 * LEN;

        output[batch_idx + threadIdx.x + LEN * 0] = ldg(&emb_0[local_x_0 + threadIdx.x]);
        output[batch_idx + threadIdx.x + LEN * 1] = ldg(&emb_1[local_x_1 + threadIdx.x]);
        output[batch_idx + threadIdx.x + LEN * 2] = ldg(&emb_2[local_x_2 + threadIdx.x]);
        output[batch_idx + threadIdx.x + LEN * 3] = ldg(&emb_3[local_x_3 + threadIdx.x]);
        output[batch_idx + threadIdx.x + LEN * 4] = ldg(&emb_4[local_x_4 + threadIdx.x]);
        output[batch_idx + threadIdx.x + LEN * 5] = ldg(&emb_5[local_x_5 + threadIdx.x]);
        output[batch_idx + threadIdx.x + LEN * 6] = ldg(&emb_6[local_x_6 + threadIdx.x]);
        output[batch_idx + threadIdx.x + LEN * 7] = ldg(&emb_7[local_x_7 + threadIdx.x]);
    }
}

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
                         cudaStream_t stream)
{
    constexpr int VPT = 1;
    constexpr int TPB = 32;
    postEmbedding<T, TPB, VPT, 20>
        <<<batch_size, TPB, 0, stream>>>(x, emb_0, emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, output);
}

template void invokePostEmbedding(const int* x,
                                  const float* emb_0,
                                  const float* emb_1,
                                  const float* emb_2,
                                  const float* emb_3,
                                  const float* emb_4,
                                  const float* emb_5,
                                  const float* emb_6,
                                  const float* emb_7,
                                  float* output,
                                  const int batch_size,
                                  cudaStream_t stream);

template void invokePostEmbedding(const int* x,
                                  const half* emb_0,
                                  const half* emb_1,
                                  const half* emb_2,
                                  const half* emb_3,
                                  const half* emb_4,
                                  const half* emb_5,
                                  const half* emb_6,
                                  const half* emb_7,
                                  half* output,
                                  const int batch_size,
                                  cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokePostEmbedding(const int* x,
                                  const __nv_bfloat16* emb_0,
                                  const __nv_bfloat16* emb_1,
                                  const __nv_bfloat16* emb_2,
                                  const __nv_bfloat16* emb_3,
                                  const __nv_bfloat16* emb_4,
                                  const __nv_bfloat16* emb_5,
                                  const __nv_bfloat16* emb_6,
                                  const __nv_bfloat16* emb_7,
                                  __nv_bfloat16* output,
                                  const int batch_size,
                                  cudaStream_t stream);
#endif

__inline__ __device__ float sigmoid(float x)
{
    return 1.0f / ( 1.0f + expf( -x));
}

__inline__ __device__ float sigmoid(half x)
{
    return sigmoid(__half2float(x));
}

#ifdef ENABLE_BF16
__inline__ __device__ float sigmoid(__nv_bfloat16 x)
{
    return sigmoid(__bfloat162float(x));
}
#endif

template<typename T>
__global__ void addTwoAddBiasSigmoid(const T* __restrict input0,
                                     const T* __restrict input1,
                                     const T* __restrict bias0,
                                     const T* __restrict bias1,
                                     float* output,
                                     const int batch_size)
{
    if (threadIdx.x < batch_size) {
        const int idx = threadIdx.x;
        output[idx] = sigmoid(ldg(&input0[idx]) + ldg(&bias0[idx]) + ldg(&input1[idx]) + ldg(&bias1[idx]));
    }
}

template<typename T>
void invokeAddTwoAddBiasSigmoid(const T* input0,
                                const T* input1,
                                const T* bias0,
                                const T* bias1,
                                float* output,
                                const int batch_size,
                                cudaStream_t stream)
{
    FT_CHECK(batch_size <= 10);
    addTwoAddBiasSigmoid<<<1, batch_size, 0, stream>>>(input0, input1, bias0, bias1, output, batch_size);
}
template void invokeAddTwoAddBiasSigmoid(const float* input0,
                                         const float* input1,
                                         const float* bias0,
                                         const float* bias1,
                                         float* output,
                                         const int batch_size,
                                         cudaStream_t stream);

template void invokeAddTwoAddBiasSigmoid(const half* input0,
                                         const half* input1,
                                         const half* bias0,
                                         const half* bias1,
                                         float* output,
                                         const int batch_size,
                                         cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeAddTwoAddBiasSigmoid(const __nv_bfloat16* input0,
                                         const __nv_bfloat16* input1,
                                         const __nv_bfloat16* bias0,
                                         const __nv_bfloat16* bias1,
                                         float* output,
                                         const int batch_size,
                                         cudaStream_t stream);
#endif
}  // namespace fastertransformer