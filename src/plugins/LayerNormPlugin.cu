#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template <int TPB, int VPT>
__global__ void ln_vec(
    const int ld, const half* input, half* output, const half* beta, const half* gamma)
{
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    half in_local[VPT];
    half beta_local[VPT];
    half gamma_local[VPT];
    copy<sizeof(half) * VPT>(&input[idx], in_local);
    float local = 0.f;
    float local2 = 0.f;

    const float rld = float(1) / float(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = rld * __half2float(in_local[it]);
        local += tmp;
        local2 += tmp * __half2float(in_local[it]);
    }

    copy<sizeof(half) * VPT>(&gamma[threadIdx.x * VPT], gamma_local);
    copy<sizeof(half) * VPT>(&beta[threadIdx.x * VPT], beta_local);

    using BlockReduce = cub::BlockReduce<kvp<float>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu;     // mean
    __shared__ float rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<float>(local, local2), cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu );
    }
    __syncthreads();
    ///*
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] = gamma_local[it] * (in_local[it] - __float2half(mu)) * __float2half(rsigma) + beta_local[it];
    }
    /* */

    copy<sizeof(half) * VPT>(in_local, &output[idx]);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = 1;
    for(int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
        nBlock *= inputDesc[0].dims.d[i];
    if (inputDesc[0].type == DataType::kHALF)
    {
        constexpr int VPT = 4;
        constexpr int TPB = 768 / VPT;
        ln_vec<TPB, VPT><<<nBlock, TPB, 0, stream>>>(768, (half *)inputs[0], (half *)outputs[0], (half *)inputs[2], (half *)inputs[1]);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
