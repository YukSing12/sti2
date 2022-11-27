#include "PreEmbeddingPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    PreEmbeddingPluginCreator::fc_ {};
std::vector<PluginField> PreEmbeddingPluginCreator::attr_;


template <typename T, int TPB, int VPT>
__global__ void embedding(const int ld, const T* tokEmb, const T* wordEmb, const T* posEmb, const int32_t* tokIds, const int32_t* wordIds, const int32_t* posIds,  
      T* output)//const half* gamma, const half* beta, 
{
    cub::Sum pairSum;

    __shared__ int wordId;
    __shared__ int tokenId;
    __shared__ int posId;

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
    int32_t const outOffset = seqPos * ld;


    if (wordId >= 0 && wordId < 50000 && tokenId >= 0 && tokenId < 4 && posId >= 0 && posId < 513)
    {
        for (int it = threadIdx.x; it < ld; it += TPB )
        { 
            T val =  wordEmb[woffset + it] + tokEmb[toffset + it]  + posEmb[poffset + it];
            output[outOffset + it] = val;
        }
    }
}


int32_t PreEmbeddingPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = 1;
    for(int i = 0; i < inputDesc[3].dims.nbDims - 1; ++i)
        nBlock *= inputDesc[3].dims.d[i]; 
    if (inputDesc[0].type == DataType::kHALF)
    {

        constexpr int VPT = 4;
        constexpr int TPB = 768 / VPT;
        embedding<half, TPB, VPT><<<nBlock, TPB, 0, stream>>>(768, (half *)inputs[0], (half *)inputs[1], (half *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (half *)outputs[0]);//(half *)inputs[6], (half *)inputs[7], 
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(PreEmbeddingPluginCreator);