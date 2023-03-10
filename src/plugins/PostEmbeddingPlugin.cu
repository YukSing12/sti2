#include "PostEmbeddingPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    PostEmbeddingPluginCreator::fc_ {};
std::vector<PluginField> PostEmbeddingPluginCreator::attr_;

template<typename T, int TPB, int VPT, int LEN>
__global__ void post_embedding(const int* x, const T* emb_0, const T* emb_1, const T* emb_2, const T* emb_3, const T* emb_4, const T* emb_5, const T* emb_6, const T* emb_7, T* output)
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
    if(threadIdx.x == 0)
    {
        local_x_0 = x[0+blockIdx.x*8]* LEN;
        local_x_1 = x[1+blockIdx.x*8]* LEN;
        local_x_2 = x[2+blockIdx.x*8]* LEN;
        local_x_3 = x[3+blockIdx.x*8]* LEN;
        local_x_4 = x[4+blockIdx.x*8]* LEN;
        local_x_5 = x[5+blockIdx.x*8]* LEN;
        local_x_6 = x[6+blockIdx.x*8]* LEN;
        local_x_7 = x[7+blockIdx.x*8]* LEN;
        
    }
    __syncthreads();

    if(threadIdx.x < LEN)
    {
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
int32_t PostEmbeddingPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = inputDesc[0].dims.d[0];    // batch
    // if (inputDesc[0].type == DataType::kFLOAT)
    // {
    //     constexpr int VPT = 1;
    //     constexpr int TPB = 32;
    //     post_embedding<float, TPB, VPT, 20><<<nBlock, TPB, 0, stream>>>((int *)inputs[0],
    //                                                                 (float *)inputs[1],
    //                                                                 (float *)inputs[2],
    //                                                                 (float *)inputs[3],
    //                                                                 (float *)inputs[4],
    //                                                                 (float *)inputs[5],
    //                                                                 (float *)inputs[6],
    //                                                                 (float *)inputs[7],
    //                                                                 (float *)inputs[8],
    //                                                                 (float *)outputs[0]);
    // }
    // else 
    if (inputDesc[1].type == DataType::kHALF)
    {
        constexpr int VPT = 1;
        constexpr int TPB = 32;
        post_embedding<half, TPB, VPT, 20><<<nBlock, TPB, 0, stream>>>((int *)inputs[0],
                                                                    (half *)inputs[1],
                                                                    (half *)inputs[2],
                                                                    (half *)inputs[3],
                                                                    (half *)inputs[4],
                                                                    (half *)inputs[5],
                                                                    (half *)inputs[6],
                                                                    (half *)inputs[7],
                                                                    (half *)inputs[8],
                                                                    (half *)outputs[0]);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(PostEmbeddingPluginCreator);