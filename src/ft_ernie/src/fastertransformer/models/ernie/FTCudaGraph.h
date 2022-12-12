/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "src/fastertransformer/utils/cuda_utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <vector>

namespace fastertransformer {

class FTCudaGraph {
public:
    explicit FTCudaGraph() = default;

    FTCudaGraph(const FTCudaGraph&) = delete;

    FTCudaGraph& operator=(const FTCudaGraph&) = delete;

    FTCudaGraph(FTCudaGraph&&) = delete;

    FTCudaGraph& operator=(FTCudaGraph&&) = delete;

    ~FTCudaGraph()
    {
        if (instantiated_) {
            check_cuda_error(cudaGraphDestroy(graph_));
            check_cuda_error(cudaGraphExecDestroy(graph_exec_));
            instantiated_ = false;
        }
    }

    void beginCapture(cudaStream_t stream)
    {
        check_cuda_error(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    }

    void launch(cudaStream_t stream)
    {
        if (instantiated_) {
            check_cuda_error(cudaGraphLaunch(graph_exec_, stream));
        }
        else {
            FT_LOG_ERROR("Launching an invalid or uninstantiated graph\n");
        }
    }

    void endCapture(cudaStream_t stream)
    {
        if (instantiated_) {
            check_cuda_error(cudaGraphDestroy(graph_));
        }
        check_cuda_error(cudaStreamEndCapture(stream, &graph_));

        bool need_instantiation;

        if (instantiated_) {
            cudaGraphExecUpdateResult updateResult;
            cudaGraphNode_t errorNode;
            // First we try to update the graph as this is much cheaper than re-instantiation
            cudaGraphExecUpdate(graph_exec_, graph_, &errorNode, &updateResult);
            if (graph_exec_ == nullptr || updateResult != cudaGraphExecUpdateSuccess) {
                // The update is unsuccessful, need to re-instantiate
                cudaGetLastError();  // <- Clear the error state
                if (graph_exec_ != nullptr) {
                    check_cuda_error(cudaGraphExecDestroy(graph_exec_));
                }
                need_instantiation = true;
            }
            else {
                // The update is successful, no need to re-instantiate
                need_instantiation = false;
            }
        }
        else {
            need_instantiation = true;
        }

        if (need_instantiation) {
            check_cuda_error(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
        }
        instantiated_ = true;
    }

    static std::string AppendShape2Key(std::vector<size_t> shape, std::string key = "")
    {
        std::ostringstream oss;
        oss << key;
        for (size_t i = 0; i < shape.size(); ++i)
            oss << "," << shape[i];
        return oss.str();
    }

private:
    cudaGraph_t graph_{};
    cudaGraphExec_t graph_exec_{};
    bool instantiated_;
};

}  // namespace fastertransformer
