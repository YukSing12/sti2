#include "cookbookHelper.hpp"
#include <iostream>
#include <fstream>

using namespace nvonnxparser;
using namespace nvinfer1;

static Logger     gLogger(ILogger::Severity::kERROR);

int main(int argc, char** argv)
{
    CHECK(cudaSetDevice(0));
    for(int i = 0; i < argc; ++i)
    {
        std::cout << argv[i] << std::endl;
    }
    if(argc != 3)
    {
        std::cout << "Usage: onnx2trt.exe <input_onnx_file> <output_trt_file>" << std::endl;
        return -1;
    }
    std::string output_trt_file = argv[2];
    
    IBuilder*               builder = createInferBuilder(gLogger);
    INetworkDefinition *    network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile *  profile = builder->createOptimizationProfile();
    IBuilderConfig *        config  = builder->createBuilderConfig();
    // config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

    IParser*  parser = createParser(*network, gLogger);
    parser->parseFromFile(argv[1], static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 128, 1}});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 128, 1}});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 128, 1}});

    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 128, 1}});
    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 128, 1}});
    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 128, 1}});

    profile->setDimensions(network->getInput(2)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 128, 1}});
    profile->setDimensions(network->getInput(2)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 128, 1}});
    profile->setDimensions(network->getInput(2)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 128, 1}});
    
    profile->setDimensions(network->getInput(3)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 128, 1}});
    profile->setDimensions(network->getInput(3)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 128, 1}});
    profile->setDimensions(network->getInput(3)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 128, 1}});

    profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    profile->setDimensions(network->getInput(5)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(5)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(5)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    profile->setDimensions(network->getInput(6)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(6)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(6)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    profile->setDimensions(network->getInput(7)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(7)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(7)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    profile->setDimensions(network->getInput(8)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(8)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(8)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    profile->setDimensions(network->getInput(9)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(9)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(9)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    profile->setDimensions(network->getInput(10)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(10)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(10)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    profile->setDimensions(network->getInput(11)->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
    profile->setDimensions(network->getInput(11)->getName(), OptProfileSelector::kOPT, Dims32 {3, {4, 1, 1}});
    profile->setDimensions(network->getInput(11)->getName(), OptProfileSelector::kMAX, Dims32 {3, {10, 1, 1}});

    config->addOptimizationProfile(profile);
    
    
    IHostMemory*  engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0)
    {
        std::cout << "Failed building serialized engine!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;
    IRuntime *runtime {createInferRuntime(gLogger)};
    ICudaEngine *engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    if (engine == nullptr)
    {
        std::cout << "Failed building engine!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded building engine!" << std::endl;

    std::ofstream engineFile(output_trt_file, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Failed opening file to write" << std::endl;
        return -1;
    }
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    if (engineFile.fail())
    {
        std::cout << "Failed saving .plan file!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded saving .plan file!" << std::endl;

    return 0;
}