#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuda import cudart  # 使用 cuda runtime API
import numpy as np
import os
import tensorrt as trt
import sys
from tqdm import tqdm
import time
from utils.loadLabelsandData import loadLabelsAndData


def run(mode):
    trtFile = "Ernie_fp16.plan"
    testFile = f"data/{mode}.test.txt"
    SAVE_PATH = f"{mode}.res.txt"
    # 指定 Logger，可用等级：VERBOSE，INFO，WARNING，ERRROR，INTERNAL_ERROR
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trtFile):  # 如果有 .plan 文件则直接读取
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:  # 没有 .plan 文件，从头开始创建
        print("Failed finding serialized engine!")
        exit(-1)

    engine = trt.Runtime(logger).deserialize_cuda_engine(
        engineString)  # 使用 Runtime 来创建 engine
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")
    context = engine.create_execution_context()  # 创建 context（相当于 GPU 进程）
    datas = loadLabelsAndData(testFile)
    nInput = np.sum([engine.binding_is_input(i)
                    for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    io_flags = []
    for i in range(nInput):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(name)
        dtype = engine.get_binding_dtype(name)
        flags_tuple = (name, shape, trt.nptype(dtype))
        print("Bind[%2d]:i[%2d]->" % (i, i), dtype, shape, shape, name)
        io_flags.append(flags_tuple)
    for i in range(nInput, nInput + nOutput):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(name)
        dtype = engine.get_binding_dtype(name)
        flags_tuple = (name, shape, trt.nptype(dtype))

        io_flags.append(flags_tuple)
        print("Bind[%2d]:o[%2d]->" %
              (i, i - nInput), dtype, shape, shape, name)
    # Malloc cuda memory
    bufferD = []
    for i in range(nInput + nOutput):
        shape = io_flags[i][1]
        shape[0] = 10
        input_size = np.zeros(shape, dtype=io_flags[i][2])
        bufferD.append(cudart.cudaMalloc(input_size.nbytes)[1])
    result = []
    # infr
    for _, data in enumerate(tqdm(datas)):
        # Host-> Device
        for index, value in enumerate(data["tensors"]):
            value = value.astype(io_flags[index][2])
            context.set_binding_shape(index, value.shape)
            cudart.cudaMemcpy(bufferD[index], value.ctypes.data,
                              value.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        res = np.zeros((data["batch_size"], 1), dtype=io_flags[-1][-1])
        # for i in range(nInput+nOutput):
        #     context.set_tensor_address(io_flags[i][0], int(bufferD[i]))
        # infr
        context.execute_v2(bufferD)
        # Device->Host
        cudart.cudaMemcpy(
            res.ctypes.data, bufferD[-1], res.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        timestamp = time.time_ns()
        result.append([res, timestamp / 1000.0])
    for b in bufferD:  # 释放 Device 端内存
        cudart.cudaFree(b)
    print(f"infr done,writing results in {SAVE_PATH}")
    with open(SAVE_PATH, "w+") as fp:
        for index, data in enumerate(datas):
            line = ("%d\t%s\t" + "%f," * (len(result[index][0]) - 1) + "%f\t" + "%.3f\n") % (
                data["qid"], data["label"], *(result[index][0]), result[index][1])
            fp.write(line)
    print("write done")
