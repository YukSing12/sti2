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
    return __float2half(tanh_opt(__half2float(x)));
}

__inline__ __device__ half2 tanh_opt(half2 x)
{
    half2 y;
    y.x = __float2half(tanh_opt(__half2float(x.x)));
    y.y = __float2half(tanh_opt(__half2float(x.y)));
    return y;
}

__inline__ __device__ __nv_bfloat16 tanh_opt(__nv_bfloat16 x)
{
    return __float2bfloat16(tanh_opt(__bfloat162float(x)));
}

__inline__ __device__ __nv_bfloat162 tanh_opt(__nv_bfloat162 x)
{
    __nv_bfloat162 y;
    y.x = __float2bfloat16(tanh_opt(__bfloat162float(x.x)));
    y.y = __float2bfloat16(tanh_opt(__bfloat162float(x.y)));
    return y;
}

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
__global__ void slice(T* dst, const T* src, const int batch_size, const int seq_len, const int d_model)
{
    // TODO: vectorize
    const int src_idx = blockIdx.x * seq_len * d_model + 0 * d_model + threadIdx.x;
    const int dst_idx = blockIdx.x * d_model + threadIdx.x;
    dst[dst_idx] = src[src_idx];
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
                              const T* emb_0,
                              const T* emb_1,
                              const T* emb_2,
                              const T* emb_3,
                              const T* emb_4,
                              const T* emb_5,
                              const T* emb_6,
                              const T* emb_7,
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

        output[batch_idx + threadIdx.x + LEN * 0] = emb_0[local_x_0 + threadIdx.x];
        output[batch_idx + threadIdx.x + LEN * 1] = emb_1[local_x_1 + threadIdx.x];
        output[batch_idx + threadIdx.x + LEN * 2] = emb_2[local_x_2 + threadIdx.x];
        output[batch_idx + threadIdx.x + LEN * 3] = emb_3[local_x_3 + threadIdx.x];
        output[batch_idx + threadIdx.x + LEN * 4] = emb_4[local_x_4 + threadIdx.x];
        output[batch_idx + threadIdx.x + LEN * 5] = emb_5[local_x_5 + threadIdx.x];
        output[batch_idx + threadIdx.x + LEN * 6] = emb_6[local_x_6 + threadIdx.x];
        output[batch_idx + threadIdx.x + LEN * 7] = emb_7[local_x_7 + threadIdx.x];
    }
}

template<typename T, int LEN>
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
    FT_CHECK(LEN == 20);
    constexpr int VPT = 1;
    constexpr int TPB = 32;
    postEmbedding<T, TPB, VPT, LEN>
        <<<batch_size, TPB, 0, stream>>>(x, emb_0, emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7, output);
}

template<typename T, int LEN>
void invokePostEmbedding(const int* x,
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

template<typename T, int LEN>
void invokePostEmbedding(const int* x,
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
template<typename T, int LEN>
void invokePostEmbedding(const int* x,
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
}  // namespace fastertransformer