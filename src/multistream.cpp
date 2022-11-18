#include <sys/time.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <dlfcn.h>
#include <mutex>
#include <thread>
#include <queue>
#include "cookbookHelper.hpp"

static Logger gLogger(ILogger::Severity::kERROR);
static const int MAX_SEQ = 128;

inline void loadLibrary(const std::string &path)
{
    int32_t flags{RTLD_LAZY};
    void *handle = dlopen(path.c_str(), flags);
    if (handle == nullptr)
    {
        std::cout << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
    }
}

struct sample
{
    std::string qid;
    std::string label;
    std::vector<int> shape_info_0;
    int size0;
    std::vector<int> i0;
    std::vector<int> shape_info_1;
    int size1;
    std::vector<int> i1;
    std::vector<int> shape_info_2;
    int size2;
    std::vector<int> i2;
    std::vector<int> shape_info_3;
    int size3;
    std::vector<float> i3;
    std::vector<int> shape_info_4;
    int size4;
    std::vector<int> i4;
    std::vector<int> shape_info_5;
    int size5;
    std::vector<int> i5;
    std::vector<int> shape_info_6;
    int size6;
    std::vector<int> i6;
    std::vector<int> shape_info_7;
    int size7;
    std::vector<int> i7;
    std::vector<int> shape_info_8;
    int size8;
    std::vector<int> i8;
    std::vector<int> shape_info_9;
    int size9;
    std::vector<int> i9;
    std::vector<int> shape_info_10;
    int size10;
    std::vector<int> i10;
    std::vector<int> shape_info_11;
    int size11;
    std::vector<int> i11;
    std::vector<float> out_data;
    uint64_t timestamp;
};

void split_string(const std::string &str,
                  const std::string &delimiter,
                  std::vector<std::string> &fields)
{
    size_t pos = 0;
    size_t start = 0;
    size_t length = str.length();
    std::string token;
    while ((pos = str.find(delimiter, start)) != std::string::npos && start < length)
    {
        token = str.substr(start, pos - start);
        fields.push_back(token);
        start += delimiter.length() + token.length();
    }
    if (start <= length - 1)
    {
        token = str.substr(start);
        fields.push_back(token);
    }
}

void field2vec(const std::string &input_str,
               bool padding,
               std::vector<int> *shape_info,
               int *size,
               std::vector<int> *i32_vec,
               std::vector<float> *f_vec = nullptr)
{
    std::vector<std::string> i_f;
    split_string(input_str, ":", i_f);
    std::vector<std::string> i_v;
    split_string(i_f[1], " ", i_v);
    std::vector<std::string> s_f;
    split_string(i_f[0], " ", s_f);
    (*size)=0;
    for (auto &f : s_f)
    {
        shape_info->push_back(std::stoi(f));
    }
    int batch_size = shape_info->at(0);
    int seq_len = shape_info->at(1);
    if (i32_vec)
    {
        for (int i = 0; i < batch_size; ++i)
        {
            for (int j = 0; j < seq_len; ++j)
            {
                i32_vec->push_back(std::stoll(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++j)
            {
                i32_vec->push_back(0);
            }
        }
    }
    else
    {
        for (int i = 0; i < batch_size; ++i)
        {
            for (int j = 0; j < seq_len; ++j)
            {
                f_vec->push_back(std::stof(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++j)
            {
                f_vec->push_back(0);
            }
        }
    }
    if (padding)
    {
        (*shape_info)[1] = MAX_SEQ;
    }
    for(int i=0;i<3;i++)
    {
         (*size)*=(*shape_info)[i];
    }

}

void line2sample(const std::string &line, sample *sout)
{
    std::vector<std::string> fields;
    split_string(line, ";", fields);
    assert(fields.size() == 14);
    // parse qid
    std::vector<std::string> qid_f;
    split_string(fields[0], ":", qid_f);
    sout->qid = qid_f[1];
    // Parse label
    std::vector<std::string> label_f;
    split_string(fields[1], ":", label_f);
    sout->label = label_f[1];
    // Parse input field
    field2vec(fields[2], true, &(sout->shape_info_0), &(sout->size0), &(sout->i0));
    field2vec(fields[3], true, &(sout->shape_info_1), &(sout->size1), &(sout->i1));
    field2vec(fields[4], true, &(sout->shape_info_2), &(sout->size2), &(sout->i2));
    field2vec(fields[5], true, &(sout->shape_info_3), &(sout->size3), nullptr, &(sout->i3));
    field2vec(fields[6], false, &(sout->shape_info_4), &(sout->size4), &(sout->i4));
    field2vec(fields[7], false, &(sout->shape_info_5), &(sout->size5), &(sout->i5));
    field2vec(fields[8], false, &(sout->shape_info_6), &(sout->size6), &(sout->i6));
    field2vec(fields[9], false, &(sout->shape_info_7), &(sout->size7), &(sout->i7));
    field2vec(fields[10], false, &(sout->shape_info_8), &(sout->size8), &(sout->i8));
    field2vec(fields[11], false, &(sout->shape_info_9), &(sout->size9), &(sout->i9));
    field2vec(fields[12], false, &(sout->shape_info_10), &(sout->size10), &(sout->i10));
    field2vec(fields[13], false, &(sout->shape_info_11), &(sout->size11), &(sout->i11));
    (sout->out_data).resize(sout->shape_info_0[0]);
    return;
}

ICudaEngine *InitEngine(const std::string &engine_file)
{
    CHECK(cudaSetDevice(0));
    initLibNvInferPlugins(nullptr, "");
    // Load engine
    ICudaEngine *engine = nullptr;

    if (access(engine_file.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(engine_file, std::ios::binary);
        long int fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return nullptr;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return nullptr;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
        return engine;
    }
    else
    {
        std::cout << "Failed finding .plan file!" << std::endl;
        return nullptr;
    }
}
void copy_data(ICudaEngine *engine, cudaStream_t stream,sample &s, std::vector<void *> &vBufferH, std::vector<void *> &vBufferD)
{
    memcpy(vBufferH[0], s.i0.data(), s.size0*dataTypeToSize(engine->getBindingDataType(0)));
    memcpy(vBufferH[1], s.i1.data(), s.size1*dataTypeToSize(engine->getBindingDataType(1)));
    memcpy(vBufferH[2], s.i2.data(), s.size2*dataTypeToSize(engine->getBindingDataType(2)));
    memcpy(vBufferH[3], s.i3.data(), s.size3*dataTypeToSize(engine->getBindingDataType(3)));
    memcpy(vBufferH[4], s.i4.data(), s.size4*dataTypeToSize(engine->getBindingDataType(4)));
    memcpy(vBufferH[5], s.i5.data(), s.size5*dataTypeToSize(engine->getBindingDataType(5)));
    memcpy(vBufferH[6], s.i6.data(), s.size6*dataTypeToSize(engine->getBindingDataType(6)));
    memcpy(vBufferH[7], s.i7.data(), s.size7*dataTypeToSize(engine->getBindingDataType(7)));
    memcpy(vBufferH[8], s.i8.data(), s.size8*dataTypeToSize(engine->getBindingDataType(8)));
    memcpy(vBufferH[9], s.i9.data(), s.size9*dataTypeToSize(engine->getBindingDataType(9)));
    memcpy(vBufferH[10], s.i10.data(), s.size10*dataTypeToSize(engine->getBindingDataType(10)));
    memcpy(vBufferH[11], s.i11.data(), s.size11*dataTypeToSize(engine->getBindingDataType(11)));
    CHECK(cudaMemcpyAsync(vBufferD[0], vBufferH[0], s.size0*dataTypeToSize(engine->getBindingDataType(0)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[1], vBufferH[1], s.size1*dataTypeToSize(engine->getBindingDataType(1)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[2], vBufferH[2], s.size2*dataTypeToSize(engine->getBindingDataType(2)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[3], vBufferH[3], s.size3*dataTypeToSize(engine->getBindingDataType(3)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[4], vBufferH[4], s.size4*dataTypeToSize(engine->getBindingDataType(4)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[5], vBufferH[5], s.size5*dataTypeToSize(engine->getBindingDataType(5)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[6], vBufferH[6], s.size6*dataTypeToSize(engine->getBindingDataType(6)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[7], vBufferH[7], s.size7*dataTypeToSize(engine->getBindingDataType(7)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[8], vBufferH[8], s.size8*dataTypeToSize(engine->getBindingDataType(8)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[9], vBufferH[9], s.size9*dataTypeToSize(engine->getBindingDataType(9)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[10], vBufferH[10], s.size10*dataTypeToSize(engine->getBindingDataType(10)), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[11], vBufferH[11], s.size11*dataTypeToSize(engine->getBindingDataType(11)), cudaMemcpyHostToDevice, stream));
}

void run(ICudaEngine *engine, IExecutionContext *context, cudaStream_t stream, std::map<int, cudaGraphExec_t> &MapOfGraphs,
         sample &s, std::vector<void *> &vBufferH, std::vector<void *> &vBufferD)
{
    int batch_size = s.shape_info_0[0];
    context->setBindingDimensions(0, Dims32{3, {batch_size, s.shape_info_0[1], s.shape_info_0[2]}});
    context->setBindingDimensions(1, Dims32{3, {batch_size, s.shape_info_1[1], s.shape_info_1[2]}});
    context->setBindingDimensions(2, Dims32{3, {batch_size, s.shape_info_2[1], s.shape_info_2[2]}});
    context->setBindingDimensions(3, Dims32{3, {batch_size, s.shape_info_3[1], s.shape_info_3[2]}});
    context->setBindingDimensions(4, Dims32{3, {batch_size, s.shape_info_4[1], s.shape_info_4[2]}});
    context->setBindingDimensions(5, Dims32{3, {batch_size, s.shape_info_5[1], s.shape_info_5[2]}});
    context->setBindingDimensions(6, Dims32{3, {batch_size, s.shape_info_6[1], s.shape_info_6[2]}});
    context->setBindingDimensions(7, Dims32{3, {batch_size, s.shape_info_7[1], s.shape_info_7[2]}});
    context->setBindingDimensions(8, Dims32{3, {batch_size, s.shape_info_8[1], s.shape_info_8[2]}});
    context->setBindingDimensions(9, Dims32{3, {batch_size, s.shape_info_9[1], s.shape_info_9[2]}});
    context->setBindingDimensions(10, Dims32{3, {batch_size, s.shape_info_10[1], s.shape_info_10[2]}});
    context->setBindingDimensions(11, Dims32{3, {batch_size, s.shape_info_11[1], s.shape_info_11[2]}});
    auto it = MapOfGraphs.find(batch_size);
    if (it == MapOfGraphs.end())
    {
       // new batch, so need to capture new graph
        cudaGraph_t graph;
        cudaGraphExec_t graphExec = nullptr;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        // Inference
        context->enqueueV2(vBufferD.data(), stream, nullptr);
        CHECK(cudaMemcpyAsync(vBufferH[12], vBufferD[12],batch_size*dataTypeToSize(engine->getBindingDataType(12)), cudaMemcpyDeviceToHost, stream));
        
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        // add this graph to the container of saved graphs
        MapOfGraphs[batch_size] = graphExec;
        }
    else
    {
        // recognized parameters, so can launch previously captured graph
        cudaGraphLaunch(it->second, stream);
    }

    memcpy(s.out_data.data(),  vBufferH[12], batch_size*dataTypeToSize(engine->getBindingDataType(12)));
    struct timeval tv;
    gettimeofday(&tv, NULL);
    s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    return;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage: main.exe <engine_file> <input_data_file> <output_data_file> [plugin_path1] [plugin_path2] ..." << std::endl;
        return -1;
    }
    for (size_t i = 4; i < argc; ++i)
    {
        std::cout << "Loading supplied plugin library: " << argv[i] << std::endl;
        loadLibrary(argv[i]);
    }

    // init
    std::string engine_file = argv[1];
    auto engine = InitEngine(engine_file);
    IExecutionContext *context = engine->createExecutionContext();

    // stream
    cudaStream_t stream1;
    CHECK(cudaStreamCreate(&stream1));
    cudaStream_t stream2;
    CHECK(cudaStreamCreate(&stream2));
    // graph
    std::map<int, cudaGraphExec_t> MapOfGraphs;

    // allocate memory
    int nBinding = engine->getNbBindings();
    std::vector<int> vBindingSize(nBinding, 0);
    std::vector<void *> vBufferH1{nBinding, nullptr};
    std::vector<void *> vBufferD1{nBinding, nullptr};
    std::vector<void *> vBufferH2{nBinding, nullptr};
    std::vector<void *> vBufferD2{nBinding, nullptr};
    // tmp_0 ~ tmp_3
    for (size_t i = 0; i < 4; i++)
    {
        // CHECK(cudaMallocHost(&vBufferH[i], 10 * 128 * 1 * sizeof(float)));
        CHECK(cudaMalloc(&vBufferD1[i], 10 * 128 * 1 * sizeof(float)));
        cudaHostAlloc(&vBufferH1[i], 10 * 128 * 1 * sizeof(float),cudaHostAllocMapped);
        CHECK(cudaMalloc(&vBufferD2[i], 10 * 128 * 1 * sizeof(float)));
        cudaHostAlloc(&vBufferH2[i], 10 * 128 * 1 * sizeof(float),cudaHostAllocMapped);
    }
    // tmp_6 ~ tmp_13
    for (size_t i = 4; i < 12; i++)
    {
        // CHECK(cudaMallocHost(&vBufferH[i], 10 * 1 * 1 * sizeof(float)));
        CHECK(cudaMalloc(&vBufferD1[i], 10 * 1 * 1 * sizeof(float)));
        cudaHostAlloc(&vBufferH1[i], 10 * 1 * 1 * sizeof(float),cudaHostAllocMapped);
        CHECK(cudaMalloc(&vBufferD2[i], 10 * 1 * 1 * sizeof(float)));
        cudaHostAlloc(&vBufferH2[i], 10 * 1 * 1 * sizeof(float),cudaHostAllocMapped);
    }
    // output
    for (size_t i = 12; i < 13; i++)
    {
        // CHECK(cudaMallocHost(&vBufferH[i], 10 * 1 * 1 * sizeof(float)));
        cudaHostAlloc(&vBufferH1[i], 10 * 1 * 1 * sizeof(float),cudaHostAllocMapped);
        CHECK(cudaMalloc(&vBufferD1[i], 10 * 1 * 1 * sizeof(float)));
         cudaHostAlloc(&vBufferH2[i], 10 * 1 * 1 * sizeof(float),cudaHostAllocMapped);
        CHECK(cudaMalloc(&vBufferD2[i], 10 * 1 * 1 * sizeof(float)));
    }

    // preprocess
    std::string aline;
    std::ifstream ifs;
    ifs.open(argv[2], std::ios::in);
    std::ofstream ofs;
    ofs.open(argv[3], std::ios::out);
    std::vector<sample> sample_vec;
    while (std::getline(ifs, aline))
    {
        sample s;
        line2sample(aline, &s);
        sample_vec.push_back(s);
    }

    // inference
    copy_data(engine,stream1,sample_vec[0],vBufferH1, vBufferD1);
    for (size_t index=0;index<sample_vec.size();index++)
    {
        if(index%2==0)
        {   
            if(index!=sample_vec.size()-1)
            {
                copy_data(engine,stream2,sample_vec[index+1],vBufferH2, vBufferD2);
            } 
            run(engine, context, stream1, MapOfGraphs, sample_vec[index], vBufferH1, vBufferD1);
            cudaStreamSynchronize(stream1);
        }
        else
        {
            if(index!=sample_vec.size()-1)
            {
                copy_data(engine,stream1,sample_vec[index+1],vBufferH1, vBufferD1);
            } 
            run(engine, context, stream2, MapOfGraphs, sample_vec[index], vBufferH2, vBufferD2);
            cudaStreamSynchronize(stream2);
        }
       
    }

    // postprocess
    for (auto &s : sample_vec)
    {
        std::ostringstream oss;
        oss << s.qid << "\t";
        oss << s.label << "\t";
        for (int i = 0; i < s.out_data.size(); ++i)
        {
            oss << s.out_data[i];
            if (i == s.out_data.size() - 1)
            {
                oss << "\t";
            }
            else
            {
                oss << ",";
            }
        }
        oss << s.timestamp << "\n";
        ofs.write(oss.str().c_str(), oss.str().length());
    }
    ofs.close();
    ifs.close();

    // Release pinned memory
    for (int i = 0; i < 13; ++i)
    {
        CHECK(cudaFreeHost(vBufferH1[i]));
        CHECK(cudaFree(vBufferD1[i]));
        CHECK(cudaFreeHost(vBufferH2[i]));
        CHECK(cudaFree(vBufferD2[i]));
    }

    return 0;
}
