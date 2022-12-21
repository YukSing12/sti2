#include "ErnieEngine.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/debug_utils.h"
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <vector>
#include <cuda_fp16.h>

using namespace fastertransformer;

static int MAX_SEQ = 128;

struct sample {
    std::string qid;
    std::string label;
    int size0;
    std::vector<int> shape_info_0;
    std::vector<int> i0;
    int size1;
    std::vector<int> shape_info_1;
    std::vector<int> i1;
    int size2;
    std::vector<int> shape_info_2;
    std::vector<int> i2;
    int size3;
    std::vector<int> shape_info_3;
    std::vector<int> i3;
    int size4;
    std::vector<int> shape_info_4;
    std::vector<int> i4;
    std::vector<float> out_data;
    size_t batch_size;
    uint64_t timestamp;
};

void split_string(const std::string& str, const std::string& delimiter, std::vector<std::string>& fields)
{
    size_t pos = 0;
    size_t start = 0;
    size_t length = str.length();
    std::string token;
    while ((pos = str.find(delimiter, start)) != std::string::npos && start < length) {
        token = str.substr(start, pos - start);
        fields.push_back(token);
        start += delimiter.length() + token.length();
    }
    if (start <= length - 1) {
        token = str.substr(start);
        fields.push_back(token);
    }
}

void field2vec(const std::string& input_str,
               bool padding,
               int& size_i,
               std::vector<int>* shape_info,
               std::vector<int>* i32_vec,
               std::vector<float>* f_vec = nullptr)
{
    std::vector<std::string> i_f;
    split_string(input_str, ":", i_f);
    std::vector<std::string> i_v;
    split_string(i_f[1], " ", i_v);
    std::vector<std::string> s_f;
    split_string(i_f[0], " ", s_f);
    for (auto& f : s_f) {
        shape_info->push_back(std::stoi(f));
    }
    int batch_size = shape_info->at(0);
    int seq_len = shape_info->at(1);

    if (i32_vec) {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                i32_vec->push_back(std::stoll(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++j) {
                i32_vec->push_back(0);
            }
        }
    }
    else {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                f_vec->push_back(std::stof(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++j) {
                f_vec->push_back(0);
            }
        }
    }

    int size = 1;
    for (int i = 0; i < 2; i++) {
        size *= (*shape_info)[i];
    }
    size_i = size;
}

void line2sample(const std::string& line, sample* sout)
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
    int _tmp_size;

    std::vector<float> mask_f;
    // Parse input field
    field2vec(fields[2], false, sout->size0, &(sout->shape_info_0), &(sout->i0));
    field2vec(fields[3], false, sout->size1, &(sout->shape_info_1), &(sout->i1));
    field2vec(fields[4], false, sout->size2, &(sout->shape_info_2), &(sout->i2));
    field2vec(fields[5], false, sout->size3, &(sout->shape_info_3), nullptr, &mask_f);
    // get seq_len on cpu
    sout->i3.resize(sout->shape_info_3[0]);
    for (size_t i = 0; i < sout->shape_info_3[0]; i++)  // batch
    {
        sout->i3[i] = sout->shape_info_3[1];
        for (size_t j = 0; j < sout->shape_info_3[1]; j++)  // seq_len
        {
            if (mask_f[i * sout->shape_info_3[1] + j] == 0.0f) {
                sout->i3[i] = j;
                break;
            }
        }
    }
    sout->shape_info_3[1] = 1;
    sout->size3 = sout->shape_info_3[0];
    std::vector<std::vector<int>> shape_info(8);
    std::vector<std::vector<int>> f_vec(8);
    field2vec(fields[6], false, _tmp_size, &shape_info[0], &f_vec[0]);
    field2vec(fields[7], false, _tmp_size, &shape_info[1], &f_vec[1]);
    field2vec(fields[8], false, _tmp_size, &shape_info[2], &f_vec[2]);
    field2vec(fields[9], false, _tmp_size, &shape_info[3], &f_vec[3]);
    field2vec(fields[10], false, _tmp_size, &shape_info[4], &f_vec[4]);
    field2vec(fields[11], false, _tmp_size, &shape_info[5], &f_vec[5]);
    field2vec(fields[12], false, _tmp_size, &shape_info[6], &f_vec[6]);
    field2vec(fields[13], false, _tmp_size, &shape_info[7], &f_vec[7]);
    for (size_t j = 0; j < shape_info[0][0]; j++) {
        for (size_t i = 0; i < f_vec.size(); i++) {
            (sout->i4).push_back(f_vec[i][j]);
        }
    }
    (sout->shape_info_4).resize(2);
    (sout->shape_info_4)[0] = shape_info[0][0];
    (sout->shape_info_4)[1] = 8;
    (sout->size4) = shape_info[0][0] * 8;
    (sout->batch_size) = (sout->shape_info_0)[0];
    (sout->out_data).resize(sout->batch_size);
    return;
}

int* HostFill(int size, int value)
{
    int* arr = new int[size];
    std::fill(arr, arr + size, value);
    return arr;
}

void printHelp()
{
    std::cout << "Usage: ernie_infer <data_type> <ckpt_path> <input_data_file> <output_data_file> [options]\n"
              << "\t<data_type>          \tPrecision of inference. (0: FP32 1: FP16 2: BF16)\n"
              << "\t<ckpt_path>          \tPath of model weights.\n"
              << "\t<input_data_file>    \tPath of input data file to load.\n"
              << "\t<output_data_file>   \tPath of output data file to save.\n"
              << "Options:\n"
              << "\t--help,-h            \tPrint usage information and exit.\n"
              << "\t--int8               \tEnable int8 precision (default = disabled).\n"
              << "\t--useCudaGraph       \tUse CUDA graph to capture engine execution and then launch inference (default = disabled).\n"
              << "\t--warmUp             \tRun for B * L times to warmup before measuring performance"
              << "Examples:\n"
              << "\t ernie_infer 1 model/bin data/label.test.txt label.res.txt\n"
              << std::endl;
}

template<typename T>
void ernieInference(const std::string& ckpt_path, std::vector<sample>& sample_vec, const bool useCudaGraph, const bool warmUp);
template<typename T>
void ernieInt8Inference(const std::string& ckpt_path, std::vector<sample>& sample_vec, const bool useCudaGraph, const bool warmUp);

int main(int argc, char** argv)
{
    // argc=6;
    // argv[1]="1";
    // argv[2]="/workspace/cgc/sti2/model/bin";
    // argv[3]="/workspace/cgc/sti2/data/label.test.txt";
    // argv[4]="/workspace/cgc/sti2/label.res.txt";
    // argv[5]="--int8";
    printf("[INFO] Device: %s \n", getDeviceName().c_str());
    if (argc < 5) {
        printHelp();
        return -1;
    }
    bool int8_mode = false;
    bool useCudaGraph = false;
    bool warmUp = false;
    for (int i = 5; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printHelp();
            return -1;
        }
        else if (!strcmp(argv[i], "--int8")) {
            printf("[INFO] Enable INT8 mode\n");
            int8_mode = true;
        }
        else if (!strcmp(argv[i], "--useCudaGraph")) {
            printf("[INFO] Enable Cuda Graph\n");
            useCudaGraph = true;
        }
        else if (!strcmp(argv[i], "--warmUp")) {
            printf("[INFO] Enable warmUp\n");
            warmUp = true;
        }
        else {
            printHelp();
            std::cout << "\n"
                      << "[Error] Unsupported options: " << argv[i] << std::endl;
            return -1;
        }
    }

    const CublasDataType data_type = static_cast<CublasDataType>(atoi(argv[1]));  // 0 FP32, 1 FP16, 2 BF 16

    // preprocess
    std::string aline;
    std::ifstream ifs;
    ifs.open(argv[3], std::ios::in);
    std::ofstream ofs;
    ofs.open(argv[4], std::ios::out);
    std::vector<sample> sample_vec;
    while (std::getline(ifs, aline)) {
        sample s;
        line2sample(aline, &s);
        sample_vec.push_back(s);
    }

    // inference
    if (data_type == FLOAT_DATATYPE) {
        printf("[INFO] Input data type is FP32.\n");
        if (int8_mode) {
            ernieInt8Inference<float>(argv[2], sample_vec, useCudaGraph, warmUp);
        }
        else {
            ernieInference<float>(argv[2], sample_vec, useCudaGraph, warmUp);
        }
    }
    else if (data_type == HALF_DATATYPE) {
        printf("[INFO] Input data type is FP16.\n");
        if (int8_mode) {
            ernieInt8Inference<half>(argv[2], sample_vec, useCudaGraph, warmUp);
        }
        else {
            ernieInference<half>(argv[2], sample_vec, useCudaGraph, warmUp);
        }
    }else{
        throw std::runtime_error(std::string("[FT][ERROR] Input data type:" + std::string(argv[1]) + " is not supported.\n "));
    }

    // postprocess
    for (auto& s : sample_vec) {
        std::ostringstream oss;
        oss << s.qid << "\t";
        oss << s.label << "\t";
        for (int i = 0; i < s.out_data.size(); ++i) {
            oss << s.out_data[i];
            if (i == s.out_data.size() - 1) {
                oss << "\t";
            }
            else {
                oss << ",";
            }
        }
        oss << s.timestamp << "\n";
        ofs.write(oss.str().c_str(), oss.str().length());
    }
    ofs.close();
    ifs.close();
}

template<typename T>
void ernieInference(const std::string& ckpt_path, std::vector<sample>& sample_vec, const bool useCudaGraph, const bool warmUp)
{
    // Init
    auto engine = new ErnieEngine<T>(ckpt_path, false, useCudaGraph);
    auto stream = engine->getStream();
    auto max_batch_size = engine->getMaxBatch();
    auto max_seq_len = engine->getMaxSeqLen();
    
    // Allocate memory
    int* h_word_ids;
    int* h_pos_ids;
    int* h_sent_ids;
    int* h_seq_len;
    int* h_multi_ids;
    cudaHostAlloc(&h_word_ids, max_batch_size * max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_pos_ids, max_batch_size * max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_sent_ids, max_batch_size * max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_seq_len, max_batch_size * 1 * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_multi_ids, max_batch_size * 8 * sizeof(int), cudaHostAllocWriteCombined);

    if (warmUp) {
        // Warmup
        unsigned int seed = 0;
        for (size_t i = 1; i <= 10; i++) {
            for (size_t j = 1; j <= 128; j++) {
                auto temp1 = HostFill(i * j, 1);
                memcpy(h_word_ids, temp1, i * j * sizeof(int));
                auto temp2 = HostFill(i * j, 2);
                memcpy(h_pos_ids, temp2, i * j * sizeof(int));
                auto temp3 = HostFill(i * j, 3);
                memcpy(h_sent_ids, temp3, i * j * sizeof(int));
                auto temp4 = HostFill(i, (int)j);
                memcpy(h_seq_len, temp4, i * sizeof(int));
                auto temp5 = HostFill(i * 8, 1);
                memcpy(h_multi_ids, temp5, i * 8 * sizeof(int));
                engine->run(h_word_ids, h_pos_ids, h_sent_ids, h_seq_len, h_multi_ids, i, j);
            }
        }
    }

    // Inference
    for (auto& s : sample_vec) {
        memcpy(h_word_ids, s.i0.data(), s.batch_size * s.shape_info_0[1] * sizeof(int));
        memcpy(h_pos_ids, s.i1.data(), s.batch_size * s.shape_info_0[1] * sizeof(int));
        memcpy(h_sent_ids, s.i2.data(), s.batch_size * s.shape_info_0[1] * sizeof(int));
        memcpy(h_seq_len, s.i3.data(), s.batch_size * 1 * sizeof(int));
        memcpy(h_multi_ids, s.i4.data(), s.batch_size * 8 * sizeof(int));
        // Host ptr must fixed when using cuda graph
        engine->run(h_word_ids, h_pos_ids, h_sent_ids, h_seq_len, h_multi_ids, s.batch_size, s.shape_info_0[1]);
        engine->copyToCpu(s.out_data.data(), s.batch_size);

        struct timeval tv;
        gettimeofday(&tv, NULL);
        s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    }
}

template<typename T>
void ernieInt8Inference(const std::string& ckpt_path, std::vector<sample>& sample_vec, const bool useCudaGraph, const bool warmUp)
{
    // Init
    auto ernie_int8_ = new ErnieEngine<T>(ckpt_path, true, useCudaGraph);
    auto stream = ernie_int8_->getStream();
    auto max_batch_size = ernie_int8_->getMaxBatch();
    auto max_seq_len = ernie_int8_->getMaxSeqLen();
    
    // Allocate memory
    int* h_word_ids;
    int* h_pos_ids;
    int* h_sent_ids;
    int* h_seq_len;
    int* h_multi_ids;
    cudaHostAlloc(&h_word_ids, max_batch_size * max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_pos_ids, max_batch_size * max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_sent_ids, max_batch_size * max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_seq_len, max_batch_size * 1 * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&h_multi_ids, max_batch_size * 8 * sizeof(int), cudaHostAllocWriteCombined);

    if (warmUp) {
        // Warmup
        unsigned int seed = 0;
        for (size_t i = 1; i <= 10; i++) {
            for (size_t j = 1; j <= 128; j++) {
                auto temp1 = HostFill(i * j, 1);
                memcpy(h_word_ids, temp1, i * j * sizeof(int));
                auto temp2 = HostFill(i * j, 2);
                memcpy(h_pos_ids, temp2, i * j * sizeof(int));
                auto temp3 = HostFill(i * j, 3);
                memcpy(h_sent_ids, temp3, i * j * sizeof(int));
                auto temp4 = HostFill(i, (int)j);
                memcpy(h_seq_len, temp4, i * sizeof(int));
                auto temp5 = HostFill(i * 8, 1);
                memcpy(h_multi_ids, temp5, i * 8 * sizeof(int));
                ernie_int8_->runInt8(h_word_ids, h_pos_ids, h_sent_ids, h_seq_len, h_multi_ids, i, j);
            }
        }
    }

    // Inference
    for (auto& s : sample_vec) {
        memcpy(h_word_ids, s.i0.data(), s.batch_size * s.shape_info_0[1] * sizeof(int));
        memcpy(h_pos_ids, s.i1.data(), s.batch_size * s.shape_info_0[1] * sizeof(int));
        memcpy(h_sent_ids, s.i2.data(), s.batch_size * s.shape_info_0[1] * sizeof(int));
        memcpy(h_seq_len, s.i3.data(), s.batch_size * 1 * sizeof(int));
        memcpy(h_multi_ids, s.i4.data(), s.batch_size * 8 * sizeof(int));
        // Host ptr must fixed when using cuda graph
        ernie_int8_->runInt8(h_word_ids, h_pos_ids, h_sent_ids, h_seq_len, h_multi_ids, s.batch_size, s.shape_info_0[1]);
        ernie_int8_->copyToCpuInt8(s.out_data.data(), s.batch_size);

        struct timeval tv;
        gettimeofday(&tv, NULL);
        s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    }
}