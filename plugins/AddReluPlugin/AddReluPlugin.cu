#include "plugins/AddReluPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    AddReluPluginCreator::fc_ {};
std::vector<PluginField> AddReluPluginCreator::attr_;

template <int TPB, int VPT>
__global__ void add_relu(
    const int ld, const float* input, float* output, const float* beta)
{

}

template <int TPB, int VPT>
__global__ void add_relu_half(
    const int ld, const half* input, half* output, const half* beta)
{

}

int32_t AddReluPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = 1;
    for(int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
        nBlock *= inputDesc[0].dims.d[i];
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        // constexpr int VPT = ?;
        // constexpr int TPB = ?;
        // add_relu<TPB, VPT><<<nBlock, TPB, 0, stream>>>(3072, (float *)inputs[0], (float *)outputs[0], (float *)inputs[1]);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        // constexpr int VPT = ?;
        // constexpr int TPB = ?;
        // add_relu_half<TPB, VPT><<<nBlock, TPB, 0, stream>>>(3072, (half *)inputs[0], (half *)outputs[0], (half *)inputs[1]);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddReluPluginCreator);