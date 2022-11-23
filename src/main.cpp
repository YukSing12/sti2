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


#define DYMSHAPE
#define POSTEMB

static Logger gLogger(ILogger::Severity::kERROR);
static const int MAX_SEQ = 128;

std::list<std::string> GetFileNameFromDir(const std::string &dir,
                                          const char *filter) {
  std::list<std::string> files;
#if defined(_WIN32) || defined(_WIN64)
  int64_t hFile = 0;
  struct _finddata_t fileinfo;
  std::string path;
  if ((hFile = _findfirst(
           path.assign(dir).append("/" + std::string(filter)).c_str(),
           &fileinfo)) != -1) {
    do {
      if (!(fileinfo.attrib & _A_SUBDIR)) { // not directory
        std::string file_path = dir + "/" + fileinfo.name;
        files.push_back(file_path);
      }
    } while (_findnext(hFile, &fileinfo) == 0);
    _findclose(hFile);
  }
#elif defined(__linux) || defined(__unix)
  DIR *pDir = nullptr;
  struct dirent *pEntry;
  pDir = opendir(dir.c_str());
  if (pDir != nullptr) {
    while ((pEntry = readdir(pDir)) != nullptr) {
      if (strcmp(pEntry->d_name, ".") == 0 ||
          strcmp(pEntry->d_name, "..") == 0 ||
          strstr(pEntry->d_name, strstr(filter, "*") + 1) == nullptr ||
          pEntry->d_type != DT_REG) { // regular file
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

inline void loadLibrary(const std::string& path)
{
    int32_t flags{RTLD_LAZY};
    void* handle = dlopen(path.c_str(), flags);
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
  std::vector<int> i0;
  std::vector<int> shape_info_1;
  std::vector<int> i1;
  std::vector<int> shape_info_2;
  std::vector<int> i2;
  std::vector<int> shape_info_3;
  std::vector<float> i3;
  std::vector<int> shape_info_4;
#ifdef POSTEMB
  std::vector<float> i4;
#elif
  std::vector<int> i4;
#endif
  std::vector<int> shape_info_5;
  std::vector<int> i5;
  std::vector<int> shape_info_6;
  std::vector<int> i6;
  std::vector<int> shape_info_7;
  std::vector<int> i7;
  std::vector<int> shape_info_8;
  std::vector<int> i8;
  std::vector<int> shape_info_9;
  std::vector<int> i9;
  std::vector<int> shape_info_10;
  std::vector<int> i10;
  std::vector<int> shape_info_11;
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
               std::vector<int> *i32_vec,
               std::vector<float> *f_vec = nullptr)
{
    std::vector<std::string> i_f;
    split_string(input_str, ":", i_f);
    std::vector<std::string> i_v;
    split_string(i_f[1], " ", i_v);
    std::vector<std::string> s_f;
    split_string(i_f[0], " ", s_f);
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
#ifdef DYMSHAPE
        if((*shape_info)[1]<32)
        {
            (*shape_info)[1]=32;
        }
#elif
      (*shape_info)[1] = MAX_SEQ;
#endif
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
  std::vector<std::vector<int>> shape_info(8);
  std::vector<std::vector<int>> f_vec(8);
  // Parse input field
  field2vec(fields[2], true, &(sout->shape_info_0), &(sout->i0));
  field2vec(fields[3], true, &(sout->shape_info_1), &(sout->i1));
  field2vec(fields[4], true, &(sout->shape_info_2), &(sout->i2));
  field2vec(fields[5], true, &(sout->shape_info_3), nullptr, &(sout->i3));
  field2vec(fields[6], false, &shape_info[0], &f_vec[0]);
  field2vec(fields[7], false, &shape_info[1], &f_vec[1]);
  field2vec(fields[8], false, &shape_info[2], &f_vec[2]);
  field2vec(fields[9], false, &shape_info[3], &f_vec[3]);
  field2vec(fields[10], false, &shape_info[4], &f_vec[4]);
  field2vec(fields[11], false, &shape_info[5], &f_vec[5]);
  field2vec(fields[12], false, &shape_info[6], &f_vec[6]);
  field2vec(fields[13], false, &shape_info[7], &f_vec[7]);
  for (size_t j = 0; j < shape_info[0][0]; j++) 
  {
    for (size_t i = 0; i < f_vec.size(); i++) 
    {
      (sout->i4).push_back((float)f_vec[i][j]);
    }
  }
  (sout->shape_info_4).resize(2);
  (sout->shape_info_4)[0]=shape_info[0][0];
  (sout->shape_info_4)[1]=8;
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

void run(ICudaEngine *engine, IExecutionContext *context, cudaStream_t stream, sample &s, std::vector<void *> &vBufferH, std::vector<void *> &vBufferD, const int &nBinding)
{
    context->setBindingDimensions(0, Dims32{3, {s.shape_info_0[0], s.shape_info_0[1], s.shape_info_0[2]}});
    context->setBindingDimensions(1, Dims32{3, {s.shape_info_1[0], s.shape_info_1[1], s.shape_info_1[2]}});
    context->setBindingDimensions(2, Dims32{3, {s.shape_info_2[0], s.shape_info_2[1], s.shape_info_2[2]}});
    context->setBindingDimensions(3, Dims32{3, {s.shape_info_3[0], s.shape_info_3[1], s.shape_info_3[2]}});
    context->setBindingDimensions(4, Dims32{2, {s.shape_info_4[0], 8}});

  std::vector<int> vBindingSize(nBinding, 0);
  int size = 1;
  for (int i = 0; i < nBinding; ++i) 
  {
    Dims32 dim = context->getBindingDimensions(i);
    size = 1;
    for (int j = 0; j < dim.nbDims; ++j) 
    {
      size *= dim.d[j];
    }
    vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    // std::cout << "vBindingSize[" << i << "] = " << vBindingSize[i] << std::endl;
  }

    CHECK(cudaMemcpyAsync(vBufferD[0], s.i0.data(), vBindingSize[0], cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[1], s.i1.data(), vBindingSize[1], cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[2], s.i2.data(), vBindingSize[2], cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[3], s.i3.data(), vBindingSize[3], cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(vBufferD[4], s.i4.data(), vBindingSize[4], cudaMemcpyHostToDevice, stream));
    // Inference
    context->enqueueV2(vBufferD.data(), stream, nullptr);

    // Get output from device to host
    s.out_data.resize(size);
    CHECK(cudaMemcpyAsync(s.out_data.data(), vBufferD[5], vBindingSize[5],cudaMemcpyDeviceToHost, stream));

  struct timeval tv;
  gettimeofday(&tv, NULL);
  s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
  return;
}

int main(int argc, char *argv[]) 
{
  if (argc < 4) 
  {
    std::cout << "Usage: main.exe <engine_file> <input_data_file> <output_data_file> [plugins_path]" << std::endl;
    return -1;
  }else if(argc == 5) 
  {
    auto so_files = GetFileNameFromDir(argv[4], "*.so");
    std::cout << "Found " << so_files.size() << " plugin(s) in " << argv[4] << std::endl;
    for(const auto &so_file : so_files) 
    {
      std::cout << "Loading supplied plugin library: " << so_file << std::endl;
      loadLibrary(so_file);
    }
  }

  // init
  std::string engine_file = argv[1];
  auto engine = InitEngine(engine_file);
  IExecutionContext *context = engine->createExecutionContext();
  int nBinding = engine->getNbBindings();
  std::vector<int> vBindingSize(nBinding, 0);

  // stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  // graph
  cudaGraph_t     graph;
  cudaGraphExec_t graphExec = nullptr;

  // allocate memory
  // TODO(pinned memory)
  std::vector<void *> vBufferH{nBinding, nullptr};
  std::vector<void *> vBufferD{nBinding, nullptr};
  // tmp_0 ~ tmp_3
  for (size_t i = 0; i < 4; i++) {
    vBufferH[i] = (void *)new char[10 * 128 * 1 * sizeof(float)];
    CHECK(cudaMalloc(&vBufferD[i], 10 * 128 * 1 * sizeof(float)));
  }
  // tmp_6 ~ tmp_13
  vBufferH[4] = (void *)new char[10 * 8  * sizeof(float)];
  CHECK(cudaMalloc(&vBufferD[4], 10 * 8  * sizeof(float)));

  // output
  vBufferH[5] = (void *)new char[10 * 1 * 1 * sizeof(float)];
  CHECK(cudaMalloc(&vBufferD[5], 10 * 1 * 1 * sizeof(float)));

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
  for (auto &s : sample_vec) 
  {
    run(engine, context, stream, s, vBufferH, vBufferD, nBinding);
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

  // Release memory
  for (int i = 0; i < nBinding; ++i) 
  {
    delete[] vBufferH[i];
    CHECK(cudaFree(vBufferD[i]));
  }

  return 0;
}
