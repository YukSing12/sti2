#include <NvInfer.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include "common.cuh"
#include <string>
#include <vector>

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (((X) + (Y)-1) / (Y) * (Y))

#if NDEBUG
#define WHERE_AM_I()                                 \
        do                                               \
        {                                                \
            printf("[%s]: this=->%p\n", __func__, this); \
        } while (0);
#else
#define WHERE_AM_I()
#endif // #ifndef NDEBUG

namespace nvinfer1
{
    static const char *PLUGIN_NAME {"AddRelu"};
    static const char *PLUGIN_VERSION {"1"};

    class AddReluPlugin : public IPluginV2DynamicExt
    {
    private:
        std::string name_;
        std::string namespace_;

    public:
        AddReluPlugin(const std::string &name):
                name_(name)
        {
            WHERE_AM_I();
        }

        AddReluPlugin(const std::string &name, const void *data, size_t length):
                name_(name)
        {
            WHERE_AM_I();
        }

        AddReluPlugin() = delete;

        ~AddReluPlugin()
        {
            WHERE_AM_I();
        }

        size_t getSerializationSize() const noexcept override
                {
                        WHERE_AM_I();
                return 0;
                }

        void serialize(void *buffer) const noexcept override
                {
                        WHERE_AM_I();
                }

        IPluginV2DynamicExt *clone() const noexcept override
                {
                        WHERE_AM_I();
                return new AddReluPlugin(name_);
                }

        int getNbOutputs() const noexcept override
                {
                        WHERE_AM_I();
                return 1;
                }

        DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override
                {
                        WHERE_AM_I();
                return inputs[0];
                }
        using IPluginV2Ext::getOutputDimensions;

        bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
                {
                        WHERE_AM_I();
                if (inOut[pos].format != TensorFormat::kLINEAR)
                {
                    return false;
                }

                bool res = false;
                switch (pos)
                {
                    case 0: // input0
                        res = (inOut[0].type == DataType::kFLOAT) || (inOut[0].type == DataType::kHALF);
                    break;
                    case 1: // beta
                        res = (inOut[1].type == DataType::kFLOAT) || (inOut[1].type == DataType::kHALF);
                    break;
                    case 2: // output0
                        res = inOut[pos].type == inOut[0].type;
                    break;
                    default: // should NOT be here
                        break;
                }
                return res;
                }

        DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const noexcept override
                {
                        WHERE_AM_I();
                return inputTypes[0];
                }

        void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override
                {
                        WHERE_AM_I();
                }
        using IPluginV2Ext::configurePlugin;

        size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override
                {
                        WHERE_AM_I();
                return 0;
                }
        using IPluginV2Ext::getWorkspaceSize;

        void setPluginNamespace(const char *szNamespace) noexcept override
                {
                        WHERE_AM_I();
                namespace_ = szNamespace;
                }
        const char *getPluginNamespace() const noexcept override
                {
                        WHERE_AM_I();
                return namespace_.c_str();
                }
        const char *getPluginType() const noexcept override
                {
                        WHERE_AM_I();
                return PLUGIN_NAME;
                }
        const char *getPluginVersion() const noexcept override
                {
                        WHERE_AM_I();
                return PLUGIN_VERSION;
                }
        int initialize() noexcept override
                {
                        WHERE_AM_I();
                return 0;
                }
        void terminate() noexcept override
                {
                        WHERE_AM_I();
                return;
                }

        void destroy() noexcept override
                {
                        WHERE_AM_I();
                }

        int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
        using IPluginV2Ext::enqueue;

    }; // class AddReluPlugin

    class AddReluPluginCreator : public IPluginCreator
    {
    private:
        static PluginFieldCollection    fc_;
        static std::vector<PluginField> attr_;
        std::string                     namespace_;

    public:
        AddReluPluginCreator()
        {
            WHERE_AM_I();
            fc_.nbFields = attr_.size();
            fc_.fields   = attr_.data();
        }

        ~AddReluPluginCreator() {}

        IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
                {
                        WHERE_AM_I();
                return new AddReluPlugin(name);
                }

        IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
                {
                        WHERE_AM_I();
                return new AddReluPlugin(name, serialData, serialLength);
                }

        void setPluginNamespace(const char *szNamespace) noexcept override
                {
                        namespace_ = szNamespace;
                }

        const char *getPluginNamespace() const noexcept override
                {
                        return namespace_.c_str();
                }

        const char *getPluginName() const noexcept override
                {
                        return PLUGIN_NAME;
                }

        const char *getPluginVersion() const noexcept override
                {
                        return PLUGIN_VERSION;
                }

        const PluginFieldCollection *getFieldNames() noexcept override
                {
                        return &fc_;
                }
    }; // class AddReluPluginCreator

} // namespace nvinfer1
