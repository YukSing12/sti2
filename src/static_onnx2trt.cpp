#include <algorithm>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#elif defined(__linux) || defined(__unix)
#include <dirent.h>
#include <unistd.h>
#endif
#include "cookbookHelper.hpp"
using namespace nvonnxparser;
using namespace nvinfer1;

static Logger          gLogger(ILogger::Severity::kINFO);
bool                   postemb       = false;
bool                   dymshape      = false;
bool                   fp16          = false;
bool                   useNewFeature = false;

std::list<std::string> GetFileNameFromDir(const std::string& dir, const char* filter) {
    std::list<std::string> files;
#if defined(_WIN32) || defined(_WIN64)
    int64_t            hFile = 0;
    struct _finddata_t fileinfo;
    std::string        path;
    if ((hFile = _findfirst(path.assign(dir).append("/" + std::string(filter)).c_str(), &fileinfo)) != -1) {
        do {
            if (!(fileinfo.attrib & _A_SUBDIR)) {  // not directory
                std::string file_path = dir + "/" + fileinfo.name;
                files.push_back(file_path);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
#elif defined(__linux) || defined(__unix)
    DIR*           pDir = nullptr;
    struct dirent* pEntry;
    pDir = opendir(dir.c_str());
    if (pDir != nullptr) {
        while ((pEntry = readdir(pDir)) != nullptr) {
            if (strcmp(pEntry->d_name, ".") == 0 || strcmp(pEntry->d_name, "..") == 0 || strstr(pEntry->d_name, strstr(filter, "*") + 1) == nullptr || pEntry->d_type != DT_REG) {  // regular file
                continue;
            }
            std::string file_path = dir + "/" + pEntry->d_name;
            files.push_back(file_path);
        }
        closedir(pDir);
    }
#endif
    return files;
}

inline void loadLibrary(const std::string& path) {
    int32_t flags{ RTLD_LAZY };
    void*   handle = dlopen(path.c_str(), flags);
    if (handle == nullptr) {
        std::cout << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
    }
}
void printFlags(IBuilderConfig* config) {

    using namespace std;
    cout << "Show Used Flags Below" << endl;
    cout << "FP16:" << config->getFlag(BuilderFlag::kFP16) << endl;
    cout << "INT8 :" << config->getFlag(BuilderFlag::kINT8) << endl;
    cout << "TF32:" << config->getFlag(BuilderFlag::kTF32) << endl;
    cout << "DEBUG:" << config->getFlag(BuilderFlag::kDEBUG) << endl;
    cout << "GPU_FALLBACK:" << config->getFlag(BuilderFlag::kGPU_FALLBACK) << endl;
    cout << "REFIT:" << config->getFlag(BuilderFlag::kREFIT) << endl;
    cout << "DISABLE_TIMING_CACHE:" << config->getFlag(BuilderFlag::kDISABLE_TIMING_CACHE) << endl;
    cout << "SPARSE_WEIGHTS:" << config->getFlag(BuilderFlag::kSPARSE_WEIGHTS) << endl;
    cout << "SAFETY_SCOPE:" << config->getFlag(BuilderFlag::kSAFETY_SCOPE) << endl;
    cout << "OBEY_PRECISION_CONSTRAINTS:" << config->getFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS) << endl;
    cout << "PREFER_PRECISION_CONSTRAINTS:" << config->getFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS) << endl;
    cout << "DIRECT_IO:" << config->getFlag(BuilderFlag::kDIRECT_IO) << endl;
    cout << "REJECT_EMPTY_ALGORITHMS:" << config->getFlag(BuilderFlag::kREJECT_EMPTY_ALGORITHMS) << endl;
    cout << "FENABLE_TACTIC_HEURISTIC:" << config->getFlag(BuilderFlag::kENABLE_TACTIC_HEURISTIC) << endl;
}
void printHelp() {
    std::cout << "Usage: onnx2trt <input_onnx_file> <output_trt_file> <plugins> [options]\n" 
              << "\t<input_onnx_file>    \tPath of onnx file to load.\n"
              << "\t<output_trt_file>    \tPath of trt engine to save.\n"
              << "\t<plugins>            \tPath of plugins(*.so) to load.\n"
              << "Options:\n"
              << "\t--help,-h            \tPrint usage information and exit.\n"
              << "\t--dymshape           \tSet second dimension to dynamic shape.\n"
              << "\t--fp16               \tEnable fp16 precision.\n"
              << "\t--useNewFeature      \tEnable PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805 which may be unstable.\n"
              << "\t--postemb            \tMerge input4~input11 to one input.\n"
              << "Examples:\n"
              << "\t onnx2trt ./model/modified_model_dymshape_ln_postemb.onnx Ernie_fp16.plan ./so/plugins --postemb --fp16 --dymshape\n" 
              << std::endl;
}

int main(int argc, char** argv) {
    
    std::cout << "TensorRT: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "." << NV_TENSORRT_BUILD << std::endl;
    if (argc < 4) {
        printHelp();
        return -1;
    }

    for (int i = 4; i < argc; ++i) {
        //  std::cout << argv[i] << std::endl;
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h") ) {
            printHelp();
            return -1;
        }
        else if (!strcmp(argv[i], "--postemb")) {
            postemb = true;
        }
        else if (!strcmp(argv[i], "--fp16")) {
            fp16 = true;
        }
        else if (!strcmp(argv[i], "--dymshape")) {
            dymshape = true;
        }
        else if (!strcmp(argv[i], "--useNewFeature")) {
            useNewFeature = true;
        }else
        {
            printHelp();
            std::cout << "\n"
                      << "Unsupported options: "  << argv[i] << std::endl;
            return -1;
        }
    }
    
    CHECK(cudaSetDevice(0));
    std::string input_onnx_file = argv[1];
    std::string output_trt_file = argv[2];
    std::string plugins_path    = argv[3];

    IBuilder*             builder = createInferBuilder(gLogger);
    INetworkDefinition*   network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    IBuilderConfig*       config  = builder->createBuilderConfig();
    // config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);
    IParser* parser = createParser(*network, gLogger);

    try {
        auto so_files = GetFileNameFromDir(plugins_path, "*.so");
        std::cout << "Found " << so_files.size() << " plugin(s) in " << plugins_path << std::endl;
        for (const auto& so_file : so_files) {
            std::cout << "Loading supplied plugin library: " << so_file << std::endl;
            loadLibrary(so_file);
        }

        parser->parseFromFile(input_onnx_file.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        printHelp();
        return -1;
    }

    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    if (fp16) {
        // config->setFlag(BuilderFlag::kSTRICT_TYPES)
        std::cout << "Using fp16" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
        for (size_t i = 0; i < network->getNbLayers(); i++) {
            auto layer = network->getLayer(i);
            if (!strcmp(layer->getName(), "LayerNorm")) {
                layer->setPrecision(DataType::kFLOAT);
                layer->setOutputType(0, DataType::kFLOAT);
            }
        }
    }
    if (postemb) {
        std::cout << "Using postemb" << std::endl;
    }
    if (useNewFeature) {
        std::cout << "Using useNewFeature" << std::endl;
        config->setPreviewFeature(PreviewFeature::kFASTER_DYNAMIC_SHAPES_0805, true);
    }
    int min_dim2 = 128, opt_dim2 = 128, max_dim2 = 128;
    if (dymshape) {
        min_dim2 = 32;
        opt_dim2 = 96;
        std::cout << "Use dynamic shape with dim2" << std::endl;
    }

    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, min_dim2, 1 } });
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, opt_dim2, 1 } });
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, max_dim2, 1 } });

    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, min_dim2, 1 } });
    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, opt_dim2, 1 } });
    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, max_dim2, 1 } });

    profile->setDimensions(network->getInput(2)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, min_dim2, 1 } });
    profile->setDimensions(network->getInput(2)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, opt_dim2, 1 } });
    profile->setDimensions(network->getInput(2)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, max_dim2, 1 } });

    profile->setDimensions(network->getInput(3)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, min_dim2, 1 } });
    profile->setDimensions(network->getInput(3)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, opt_dim2, 1 } });
    profile->setDimensions(network->getInput(3)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, max_dim2, 1 } });
    if (postemb) {
        profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kMIN, Dims32{ 2, { 1, 8 } });
        profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kOPT, Dims32{ 2, { 4, 8 } });
        profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kMAX, Dims32{ 2, { 10, 8 } });
    }
    else {
        profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(4)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });

        profile->setDimensions(network->getInput(5)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(5)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(5)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });

        profile->setDimensions(network->getInput(6)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(6)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(6)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });

        profile->setDimensions(network->getInput(7)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(7)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(7)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });

        profile->setDimensions(network->getInput(8)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(8)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(8)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });

        profile->setDimensions(network->getInput(9)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(9)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(9)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });

        profile->setDimensions(network->getInput(10)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(10)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(10)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });

        profile->setDimensions(network->getInput(11)->getName(), OptProfileSelector::kMIN, Dims32{ 3, { 1, 1, 1 } });
        profile->setDimensions(network->getInput(11)->getName(), OptProfileSelector::kOPT, Dims32{ 3, { 4, 1, 1 } });
        profile->setDimensions(network->getInput(11)->getName(), OptProfileSelector::kMAX, Dims32{ 3, { 10, 1, 1 } });
    }
    config->addOptimizationProfile(profile);
    config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT));
    printFlags(config);

    std::cout << "------Start Build Engine-----" << std::endl;
    IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0) {
        std::cout << "Failed building serialized engine!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded building serialized engine!" << std::endl;
    IRuntime*    runtime{ createInferRuntime(gLogger) };
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    if (engine == nullptr) {
        std::cout << "Failed building engine!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded building engine!" << std::endl;

    std::ofstream engineFile(output_trt_file, std::ios::binary);
    if (!engineFile) {
        std::cout << "Failed opening file to write" << std::endl;
        return -1;
    }
    engineFile.write(static_cast<char*>(engineString->data()), engineString->size());
    if (engineFile.fail()) {
        std::cout << "Failed saving " << output_trt_file << ".plan file!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded saving " << output_trt_file << ".plan file!" << std::endl;

    return 0;
}