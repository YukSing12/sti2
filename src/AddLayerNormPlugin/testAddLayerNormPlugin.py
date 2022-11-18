#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import os
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt

soFilePath      = './AddLayerNormPlugin.so'
nBS             = 1
nSL             = 4
nEmbedding      = 16
epsilon         = 1e-5
npDataType      = np.float32
np.random.seed(97)

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def layerNormCPU(bufferH):
    _x1 = bufferH[0]
    _x2 = bufferH[1]
    _x3 = bufferH[2]
    _x2 = _x2 + _x3
    _x = _x1 + _x2 
    nEmbed = bufferH[0].shape[2]
    _0  = np.mean(_x,2)[:,:,np.newaxis]
    _1  = _x - _0
    _2  = _1 * _1
    _3  = np.mean(_2,2)[:,:,np.newaxis]
    _4  = np.array(epsilon,dtype=np.float32)
    _5  = _4.reshape(1,1,1)
    _6  = _3 + _5
    _7  = np.sqrt(_6)
    _8  = 1 / _7                # 1/sqrt(...)
    _9  = _1 * _8
    return _9

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'AddLayerNorm':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = 1 << int(trt.BuilderFlag.FP16) if int(npDataType == np.float16) else 0

    inputTensorList = []
    trtDataType = trt.float16 if int(npDataType == np.float16) else trt.float32
    inputTensorList.append( network.add_input('inputT1', trtDataType, [-1,-1,nEmbedding]) )
    inputTensorList.append( network.add_input('inputT2', trtDataType, [-1,-1,nEmbedding]) )
    inputTensorList.append( network.add_input('inputT3', trtDataType, [-1,-1,nEmbedding]) )


    profile = builder.create_optimization_profile()
    profile.set_shape('inputT1',[1,4,nEmbedding],[4,64,nEmbedding],[16,256,nEmbedding])
    profile.set_shape('inputT2',[1,4,nEmbedding],[4,64,nEmbedding],[16,256,nEmbedding])
    profile.set_shape('inputT3',[1,4,nEmbedding],[4,64,nEmbedding],[16,256,nEmbedding])

    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    pluginLayer.get_output(0).dtype = trtDataType
    network.mark_output(pluginLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS,nSL,nEmbedding])
    context.set_binding_shape(1,[nBS,nSL,nEmbedding])
    context.set_binding_shape(2,[nBS,nSL,nEmbedding])

    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(npDataType).reshape(nBS,nSL,nEmbedding) * 2 - 1)
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(npDataType).reshape(nBS,nSL,nEmbedding) * 2 - 1)
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(npDataType).reshape(nBS,nSL,nEmbedding) * 2 - 1)

    bufferH.append(np.empty(context.get_binding_shape(0),dtype=trt.nptype(engine.get_binding_dtype(0))))
    

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("check result:")
    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:3])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()