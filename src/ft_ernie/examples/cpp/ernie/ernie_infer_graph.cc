#include "src/fastertransformer/models/ernie/Ernie.h"
#include "src/fastertransformer/models/ernie/ErnieWeight.h"
#include "ErnieGemm.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/logger.h"
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
    size_t batchsize;
    uint64_t timestamp;
};

struct ErnieStruct {
    // constructor parameter
    size_t max_batch_size = 10;
    size_t max_seq_len = 128;
    size_t beam_width = 1;
    size_t head_num = 12;
    size_t size_per_head = 64;
    size_t d_model = head_num * size_per_head;  // 768
    size_t inter_size = d_model * 4;            // 3072
    size_t num_layer = 12;
    int sm = -1;  // assign later
    float q_scaling = 1.0f;
    // internal parameter
    size_t vocab_size = 50000;
    size_t pos_size = 513;
    size_t sent_vocab_size = 4;
    bool is_remove_padding = true;
    bool is_free_buffer_after_forward = false;
    bool is_sparse = false;
    AttentionType attention_type = AttentionType::FUSED_MHA;
    fastertransformer::ActivationType activation_type = fastertransformer::ActivationType::Relu;
    LayerNormType layernorm_type = LayerNormType::post_layernorm;
    // runtime parameter
    size_t batch_size = 0;
    size_t seq_len = 0;
    bool search_gemm = false;
} g_ernie_struct;

int* HostFill(int size, int value)
{
    int* arr = new int[size];
    std::fill(arr, arr + size, value);
    return arr;
}

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

void field2vec(const std::string& input_str, bool padding, int& size_i, std::vector<int>* shape_info, std::vector<int>* i32_vec, std::vector<float>* f_vec = nullptr) {
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
    int seq_len    = shape_info->at(1);
   
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
    (sout->batchsize) = (sout->shape_info_0)[0];
    (sout->out_data).resize(sout->batchsize);
    return;
}

template<typename T>
int ernieInference(ErnieStruct ernie_struct, const std::string& ckpt_path, std::vector<sample>& sample_vec);

void printHelp()
{
    std::cout << "Usage: ernie_infer <data_type> <ckpt_path> <input_data_file> <output_data_file> [options]\n"
              << "\t<data_type>          \tPrecision of inference. (0: FP32 1: FP16 2: BF16)\n"
              << "\t<ckpt_path>          \tPath of model weights.\n"
              << "\t<input_data_file>    \tPath of input data file to load.\n"
              << "\t<output_data_file>   \tPath of output data file to save.\n"
              << "Options:\n"
              << "\t--help,-h            \tPrint usage information and exit.\n"
              << "Examples:\n"
              << "\t ernie_infer 1 model/bin data/label.test.txt label.res.txt\n"
              << std::endl;
}

int main(int argc, char** argv)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());
    if (argc < 5) {
        printHelp();
        return -1;
    }
    for (int i = 5; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printHelp();
            return -1;
        }
        else if (!strcmp(argv[i], "--searchGEMM")) 
        {
            g_ernie_struct.search_gemm = true;
        }
        else {
            printHelp();
            std::cout << "\n"
                      << "Unsupported options: " << argv[i] << std::endl;
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
        FT_LOG_INFO("FP32");
        ernieInference<float>(g_ernie_struct, argv[2], sample_vec);
    }
    else if (data_type == HALF_DATATYPE) {
        FT_LOG_INFO("FP16");
        ernieInference<half>(g_ernie_struct, argv[2], sample_vec);
    }
#ifdef ENABLE_BF16
    else if (data_type == BFLOAT16_DATATYPE) {
        FT_LOG_INFO("BF16");
        ernieInference<__nv_bfloat16>(g_ernie_struct, argv[2], sample_vec);
    }
#endif
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
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
            } else {
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
int ernieInference(ErnieStruct ernie_struct, const std::string& ckpt_path, std::vector<sample>& sample_vec)
{
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    ernie_struct.sm = prop.major * 10 + prop.minor;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
#endif
    cublasSetStream(cublas_handle, stream);

    // Gemm file selection
    std::string gemmFileName;
    if(ernie_struct.search_gemm)
    {
        gemmFileName = std::string("gemm_config.in").substr(0, 11) + std::string("-SM") + std::to_string(ernie_struct.sm)
                               + std::string("-FP") + std::to_string(std::is_same<T, half>::value ? 16 : 32) + std::string("-BS")
                               + std::to_string(ernie_struct.max_batch_size) + std::string("-SL") + std::to_string(ernie_struct.max_seq_len)
                               + std::string("-BM") + std::to_string(ernie_struct.beam_width) + std::string(".in");
        std::ifstream infile(gemmFileName);
        if (infile.good()) {
            printf("Gemm file exist!\n");
        }
        else {
            printf("Gemm file do not exist!\n");
            for (size_t b = 1; b <= ernie_struct.max_batch_size; b++) {
                for (size_t l = 16; l <= ernie_struct.max_seq_len; l++) {
                    int argv[8] = {
                        0,
                        (int)b,
                        (ernie_struct.batch_size == 128 && ernie_struct.seq_len == 384) ?
                            128 :
                            (int)l,  // seq_len, in case of OOM
                        (int)ernie_struct.head_num,
                        (int)ernie_struct.size_per_head,
                        std::is_same<T, half>::value ? 1 : 0,  // // 0 FP32, 1 FP16
                        0,                                     // int8 mode
                        1,                                     // tensor_para_size
                    };
                    ernie_gemm(argv);
                }
            }
            rename(std::string("gemm_config.in").c_str(), gemmFileName.c_str());
        }
    }else
    {
        gemmFileName = std::string("gemm_config.in").substr(0, 11) + std::string("-SM") + std::to_string(ernie_struct.sm)
                               + std::string("-FP") + std::to_string(std::is_same<T, half>::value ? 16 : 32) + std::string("-BS")
                               + std::to_string(ernie_struct.max_batch_size) + std::string("-SL") + std::to_string(ernie_struct.max_seq_len)
                               + std::string("-BM") + std::to_string(ernie_struct.beam_width) + std::string(".in");
    }

    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap(gemmFileName, "");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();
#ifdef SPARSITY_ENABLED
    cublasMMWrapper cublas_wrapper = cublasMMWrapper(
        cublas_handle, cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#else
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#endif
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    ErnieWeight<T> ernie_weights(ernie_struct.head_num,
                                        ernie_struct.size_per_head,
                                        ernie_struct.d_model,
                                        ernie_struct.inter_size,
                                        ernie_struct.vocab_size,
                                        ernie_struct.pos_size,
                                        ernie_struct.sent_vocab_size,
                                        ernie_struct.num_layer);

    ernie_struct.attention_type = getAttentionType<T>(
        ernie_struct.size_per_head, getSMVersion(), ernie_struct.is_remove_padding, ernie_struct.max_seq_len, true);
    ernie_weights.loadModel(ckpt_path);

    Ernie<T> ernie = Ernie<T>(ernie_struct.max_batch_size,
                                            ernie_struct.max_seq_len,
                                            ernie_struct.head_num,
                                            ernie_struct.size_per_head,
                                            ernie_struct.inter_size,
                                            ernie_struct.d_model,
                                            ernie_struct.num_layer,
                                            ernie_struct.vocab_size,
                                            ernie_struct.pos_size,
                                            ernie_struct.sent_vocab_size,
                                            ernie_struct.sm,
                                            ernie_struct.q_scaling,
                                            stream,  // stream placeholder
                                            &cublas_wrapper,
                                            &allocator,
                                            ernie_struct.is_free_buffer_after_forward,
                                            ernie_struct.attention_type,
                                            ernie_struct.is_sparse,
                                            ernie_struct.activation_type,
                                            ernie_struct.layernorm_type,
                                            NcclParam(0, 1),  // tensor_para
                                            NcclParam(0, 1)   // pipeline_para
    );
    ernie.setUseGraph(true);
    ernie.setHostMode(true);
    // five inputs
    int* word_ids;
    int* pos_ids;
    int* sent_ids;
    int* seq_len;
    int* multi_ids;
    cudaHostAlloc(&word_ids, ernie_struct.max_batch_size * ernie_struct.max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&pos_ids, ernie_struct.max_batch_size * ernie_struct.max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&sent_ids, ernie_struct.max_batch_size * ernie_struct.max_seq_len * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&seq_len, ernie_struct.max_batch_size * 1 * sizeof(int), cudaHostAllocWriteCombined);
    cudaHostAlloc(&multi_ids, ernie_struct.max_batch_size * 8 * sizeof(int), cudaHostAllocWriteCombined);

    // one output
    float* attn_out;
    deviceMalloc(&attn_out, ernie_struct.max_batch_size * 1, false);

    // //warmup
    // unsigned int seed               = 0;
    for (size_t i = 0; i < 10; i++)
    {
       for(size_t j=1;j<129;j++)
       {
            auto temp1=HostFill((i+1) * j, 1);
            memcpy(word_ids,temp1,(i+1) * j*sizeof(int));
            auto temp2=HostFill((i+1) * j, 2);
            memcpy(pos_ids,temp2,(i+1) * j*sizeof(int));
            auto temp3=HostFill((i+1) * j, 3);
            memcpy(sent_ids,temp3,(i+1) * j*sizeof(int));
            auto temp4=HostFill((i+1), (int)j);
            memcpy(seq_len,temp4,(i+1) * sizeof(int));
            auto temp5=HostFill((i+1) * 8, 1);
            memcpy(multi_ids,temp5,(i+1) * 8*sizeof(int));
            ernie.forward(word_ids, pos_ids, sent_ids, seq_len, multi_ids, i+1, j, &ernie_weights);
       }   
    }
    

    // inference
    for (auto& s : sample_vec) {

        memcpy(word_ids, s.i0.data(), s.size0 * sizeof(int));
        memcpy(pos_ids, s.i1.data(), s.size1 * sizeof(int));
        memcpy(sent_ids, s.i2.data(), s.size2 * sizeof(int));
        memcpy(seq_len, s.i3.data(), s.size3 * sizeof(int));
        memcpy(multi_ids, s.i4.data(), s.size4 * sizeof(int));

        ernie.forward(word_ids, pos_ids, sent_ids, seq_len, multi_ids, s.batchsize, s.shape_info_0[1], &ernie_weights);
        ernie.copyToCpu(s.out_data.data(), s.batchsize);

        struct timeval tv;
        gettimeofday(&tv, NULL);
        s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    }

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif

    cudaFreeHost(word_ids);
    cudaFreeHost(pos_ids);
    cudaFreeHost(sent_ids);
    cudaFreeHost(seq_len);
    cudaFreeHost(multi_ids);

    deviceFree(attn_out);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}