/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cookbookHelper.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <sys/time.h>

using namespace nvinfer1;

const std::string trtFile {"../Ernie.plan"};

static Logger     gLogger(ILogger::Severity::kERROR);

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c) {
	v.clear();
    std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

void SplitShape(const std::string& s, std::vector<int>& v, const std::string& c) {
	v.clear();
    std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(stoi(s.substr(pos1, pos2 - pos1)));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(stoi(s.substr(pos1)));
}

template<typename T>
void SplitValue(const std::string& s, T* v, const std::string& c, const size_t& bs, const size_t& seq_len, const size_t& pad_len) {
	std::stringstream ss(s);
	for (size_t i = 0; i < bs; i++)
	{
		for (size_t j = 0; j < seq_len; j++)
		{
			ss >> *(v + i * pad_len + j);
		}
		// padding
		for (size_t j = seq_len; j < pad_len; j++)
		{
			*(v + i * pad_len + j) = 0;
		}
	}
	return;
}


void run(const std::string& testFile, const std::string& resFile)
{
    // Load engine
    ICudaEngine *engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        std::cout << "Failed finding .plan file!" << std::endl;
        return;
    }

    // Load dataset
    std::ifstream infile(testFile);
    if(!infile.is_open())
    {
        std::cout << "Failed loading data!" << std::endl;
        return;
    }
    std::string tmp_s;
	std::string qid;
	std::string label;
    std::vector<std::string> data;
    std::vector<int> shape;
	const int dim = 3;

    // Record result
    std::stringstream ss;

    IExecutionContext *context = engine->createExecutionContext();
    int nBinding = engine->getNbBindings();
    std::vector<int> vBindingSize(nBinding, 0);

    // Allocate memory
    std::vector<void *> vBufferH {nBinding, nullptr};
    std::vector<void *> vBufferD {nBinding, nullptr};
    // tmp_0 ~ tmp_3
    for (size_t i = 0; i < 4; i++)
    {
        vBufferH[i] = (void *)new char[10 * 128 * 1 * sizeof(float)];
        CHECK(cudaMalloc(&vBufferD[i], 10 * 128 * 1 * sizeof(float)));
    }
    // tmp_6 ~ tmp_13
    for (size_t i = 4; i < 12; i++)
    {
        vBufferH[i] = (void *)new char[10 * 1 * 1 * sizeof(float)];
        CHECK(cudaMalloc(&vBufferD[i], 10 * 1 * 1 * sizeof(float)));
    }
    // output
    for (size_t i = 12; i < 13; i++)
    {
        vBufferH[i] = (void *)new char[10 * 1 * 1 * sizeof(float)];
        CHECK(cudaMalloc(&vBufferD[i], 10 * 1 * 1 * sizeof(float)));
    }
    
    

    // Start loading data and inference
    while(getline(infile, tmp_s))
    {
        SplitString(tmp_s, data, ";");
        qid = data[0].substr(data[0].find(":")+1);
        label = data[1].substr(data[1].find(":")+1);

        // 12 inputs
        int nInput   = data.size() - 2;
        for(size_t i = 0; i < nInput; ++i)
        {
            size_t pos = data[i + 2].find(":");
            SplitShape(data[i + 2].substr(0, pos), shape, " ");
            // Dynamicly binding intput shape
            if(i == 3)
            {
                context->setBindingDimensions(i, Dims32 {3, {shape[0], 128, shape[2]}});
                float *pData = (float *)vBufferH[i];
                SplitValue<float>(data[i + 2].substr(pos + 1, data[i + 2].size()), pData, " ", shape[0], shape[1], 128);
            }else if(i < 4)
            {
                context->setBindingDimensions(i, Dims32 {3, {shape[0], 128, shape[2]}});
                int *pData = (int *)vBufferH[i];
                SplitValue<int>(data[i + 2].substr(pos + 1, data[i + 2].size()), pData, " ", shape[0], shape[1], 128);
            }else
            {
                context->setBindingDimensions(i, Dims32 {3, {shape[0], shape[1], shape[2]}});
                int *pData = (int *)vBufferH[i];
                SplitValue<int>(data[i + 2].substr(pos + 1, data[i + 2].size()), pData, " ", shape[0], shape[1], 1);
            }

            Dims32 in_dim  = context->getBindingDimensions(i);
            int    size = 1;
            for (size_t j = 0; j < in_dim.nbDims; ++j)
            {
                size *= in_dim.d[j];
            }
            vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));

            // Copy data from host to device
            CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice));
        }
        // Check binding shape
        // std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
        // for (int i = 0; i < nBinding; ++i)
        // {
        //     std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        //     std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        //     std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        //     std::cout << engine->getBindingName(i) << std::endl;
        // }

        // Binding output shape
        Dims32 out_dim  = context->getBindingDimensions(12);
        int    size = 1;
        for (size_t i = 0; i < out_dim.nbDims; ++i)
        {
            size *= out_dim.d[i];
        }
        vBindingSize[12] = size * dataTypeToSize(engine->getBindingDataType(12));

        // Allocate output memory
        vBufferH[12] = (void *)new char[vBindingSize[12]];
        CHECK(cudaMalloc(&vBufferD[12], vBindingSize[12])); 

        // Inference
        context->executeV2(vBufferD.data());
        
        // Get output from device to host
        CHECK(cudaMemcpy(vBufferH[12], vBufferD[12], vBindingSize[12], cudaMemcpyDeviceToHost));

        struct timeval tv;
        gettimeofday(&tv, NULL);
        // Write output data
        float *pData = (float *)vBufferH[12];
        ss << qid << "\t" << label << "\t";
        for (size_t i = 0; i < shape[0]; i++)
        {
            ss << *(pData + i) << "\t";
        }
        ss << (tv.tv_sec * 1000000 + tv.tv_usec) << "\n";

    }
    // Release memory
    for (int i = 0; i < nBinding; ++i)
    {
        delete[] vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }

    // Save result into file
    std::ofstream outfile(resFile);
    outfile << ss.str();
    outfile.close();
    return;
}

int main(int argc, char** argv)
{
    CHECK(cudaSetDevice(0));
    if(argc < 2)
    {
        std::cout << "Usage: main.exe [label/perf]" << std::endl;
        return -1;
    }
    const std::string testFile {"../data/" + std::string(argv[1]) + ".test.txt"};
    const std::string resFile {"../" + std::string(argv[1]) + ".res.txt"};
    run(testFile, resFile);
    return 0;
}