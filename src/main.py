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

# yapf:disable

trtFile = "../Ernie.plan"
testFile = "../data/label.test.txt"

def run():
    logger = trt.Logger(trt.Logger.ERROR)                                       # 指定 Logger，可用等级：VERBOSE，INFO，WARNING，ERRROR，INTERNAL_ERROR
    if os.path.isfile(trtFile):                                                 # 如果有 .plan 文件则直接读取
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:                                                                       # 没有 .plan 文件，从头开始创建
        print("Failed finding serialized engine!")
        exit(-1)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)          # 使用 Runtime 来创建 engine
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    context = engine.create_execution_context()                                 # 创建 context（相当于 GPU 进程）

    with open(testFile, "r") as fid:
        lines = fid.readlines()
        for line in lines:
            data = line.split(";")
            print(data)
    exit(0)
    context.set_input_shape("inputT0", [3, 4, 5])                                       # Dynamic Shape 模式需要绑定真实数据形状
    nInput = [engine.get_tensor_mode(engine.get_tensor_name(i)) for i in range(engine.num_bindings)].count(trt.TensorIOMode.INPUT)
    nOutput = [engine.get_tensor_mode(engine.get_tensor_name(i)) for i in range(engine.num_bindings)].count(trt.TensorIOMode.OUTPUT)
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_tensor_dtype(engine.get_tensor_name(i)), engine.get_tensor_shape(engine.get_tensor_name(i)), context.get_tensor_shape(engine.get_tensor_name(i)), engine.get_tensor_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_tensor_dtype(engine.get_tensor_name(i)), engine.get_tensor_shape(engine.get_tensor_name(i)), context.get_tensor_shape(engine.get_tensor_name(i)), engine.get_tensor_name(i))

    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)              # 准备数据和 Host/Device 端内存
    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_tensor_shape(engine.get_tensor_name(i)), dtype=trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(i)))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):                                                     # 首先将 Host 数据拷贝到 Device 端
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.set_tensor_address("inputT0", int(bufferD[0]))
    context.set_tensor_address("outputT0", int(bufferD[1]))

    context.execute_async_v3(0)                                                 # 运行推理计算

    for i in range(nInput, nInput + nOutput):                                   # 将结果从 Device 端拷回 Host 端
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput + nOutput):
        print(engine.get_tensor_name(i))
        print(bufferH[i])

    for b in bufferD:                                                           # 释放 Device 端内存
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run()                                                                       # 创建 TensorRT 引擎并推理
    run()                                                                       # 读取 TensorRT 引擎并推理