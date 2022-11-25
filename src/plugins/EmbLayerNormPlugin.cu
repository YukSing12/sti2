#include "EmbLayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    EmbLayerNormPluginCreator::fc_ {};
std::vector<PluginField> EmbLayerNormPluginCreator::attr_;

template <typename T, typename R, int TPB, int VPT>
__device__ void ln_vec(const int ld, kvp<R> threadData,  const float* gamma, const float* beta, T* output){

    // const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    const int idx = ld * blockIdx.x;

    using BlockReduce = cub::BlockReduce<kvp<R>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ R mu;     // mean
    __shared__ R rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu );
    }
    __syncthreads();

#pragma unroll
    for (int it = threadIdx.x; it < ld; it+=TPB)
    {
        const int offset = idx + it;
        const R val(output[offset]);
        const R g(gamma[it]);
        const R b(beta[it]);
        output[offset] = g * ( val - mu) * rsigma + b;
    }

}



template <typename T, int TPB, int VPT>
__global__ void embedding(const int ld,  const T* tokEmb, const T* wordEmb, const T* posEmb, const int32_t* tokIds, const int32_t* wordIds, const int32_t* posIds,
      const float* gamma, const float* beta, T* output)
{
    cub::Sum pairSum;

    __shared__ int wordId;
    __shared__ int tokenId;
    __shared__ int posId;

    float const rld = float(1) / float(ld);
    int32_t const seqPos = blockIdx.y + blockIdx.x * gridDim.y;
    if (threadIdx.x == 0)
    {
        wordId  = (int)wordIds[seqPos];
        tokenId = (int)tokIds[seqPos];
        posId   = (int)posIds[seqPos];
    }
    __syncthreads();


    int32_t const poffset = posId * ld;
    int32_t const woffset = wordId * ld;
    int32_t const toffset = tokenId * ld;
    int32_t const outOffset = blockIdx.x * ld;

    kvp<float> threadData(0, 0);
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block

    if (wordId >= 0 && wordId < 50000 && tokenId >= 0 && tokenId < 4 && posId >= 0 && posId < 513)
    {
        for (int it = threadIdx.x; it < ld; it += TPB )
        // for (int it = 0; it < VPT; it ++)
        {
            T val =  wordEmb[woffset + it] + tokEmb[toffset + it]  + posEmb[poffset + it];
            output[outOffset + it] = val;
            float const rldval = rld * (float)val; 
            threadData = pairSum(threadData, kvp<float>(rldval, rldval * (float)val));
        }
    }
    ln_vec<T,float,TPB,VPT>(768, threadData,  gamma, beta,  output);
}


int32_t EmbLayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = 1;
    for(int i = 0; i < inputDesc[3].dims.nbDims - 1; ++i)
        nBlock *= inputDesc[3].dims.d[i]; 
    if (inputDesc[0].type == DataType::kFLOAT)
    {

        // constexpr int VPT = 16 / sizeof(float);
        constexpr int VPT = 1;
        constexpr int TPB = 768 / VPT;
        embedding<float, TPB, VPT><<<nBlock, TPB, 0, stream>>>(768, (float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (float *)inputs[6], (float *)inputs[7], (float *)outputs[0]);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {

        constexpr int VPT = 1;
        constexpr int TPB = 768 / VPT;
        embedding<half, TPB, VPT><<<nBlock, TPB, 0, stream>>>(768, (half *)inputs[0], (half *)inputs[1], (half *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (float *)inputs[6], (float *)inputs[7], (half *)outputs[0]);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(EmbLayerNormPluginCreator);