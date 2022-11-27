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
import onnx
import onnx_graphsurgeon as gs
import numpy as np

soFilePath      = './EmbLayerNormPlugin.so'
nBS             = 10
# nH              = 50
nW              = 128
tokenW          = 4
wordW           = 50000
posW            = 513

nEmbedding      = 768
epsilon         = 1e-5
idDataType      = np.int32
npDataType      = np.float16
cDataType      = np.float16

np.random.seed(97)
globalGamma     = cDataType(np.random.randn(nEmbedding))
globalBeta     = cDataType(np.random.randn(nEmbedding))

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def layerNormCPU(bufferH):
    _tokEmb   = bufferH[0]
    _wordEmb  = bufferH[1]
    _posEmb   = bufferH[2]
    _tokenIds = bufferH[3]
    _wordIds  = bufferH[4]
    _posIds   = bufferH[5]

    _wordIds  = np.squeeze(_wordIds, 2)
    _tokenIds = np.squeeze(_tokenIds, 2)
    _posIds   = np.squeeze(_posIds, 2)
    # tokargs   = np.arange(1, _tokEmb)
    # wordargs  = np.arange(1, _wordEmb)
    # posargs   = np.arange(1, _posEmb)

    # _x = _tokEmb[_tokenIds, tokargs] + _wordEmb[_wordIds, wordargs] + _posEmb[_posIds, posargs]
    _x = _tokEmb.take(_tokenIds, 0) + _wordEmb.take(_wordIds, 0) + _posEmb.take(_posIds, 0)
    # nEmbed = bufferH[0].shape[3]
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
    return _9 * globalGamma + globalBeta

def getLayerNormOnnx():
    onnx_file = "temp.onnx"
    shape1 = ('B', nW, 1)
    shape_tok  = (tokenW, nEmbedding)
    shape_word = (wordW, nEmbedding)
    shape_pos  = (posW, nEmbedding)

    shape3 = ('B', nW, nEmbedding)

    tokenIds = gs.Variable(name="tokenIds", dtype=idDataType, shape=shape1)
    wordIds  = gs.Variable(name="wordIds", dtype=idDataType, shape=shape1)
    posIds   = gs.Variable(name="posIds", dtype=idDataType, shape=shape1)
    tokEmb   = gs.Variable(name="tokEmb", dtype=npDataType, shape=shape_tok)
    wordEmb  = gs.Variable(name="wordEmb", dtype=npDataType, shape=shape_word)
    posEmb   = gs.Variable(name="posEmb", dtype=npDataType, shape=shape_pos)

    gamma = gs.Constant(name="gamma", values=globalGamma)
    beta = gs.Constant(name="beta", values=globalBeta)
    y = gs.Variable(name="y", dtype=npDataType, shape=shape3)
    layernorm = gs.Node(op="EmbLayerNorm", 
                        name="EmbLayerNorm_1", 
                        inputs=[tokEmb, wordEmb, posEmb, tokenIds, wordIds, posIds, gamma, beta], 
                        outputs=[y], 
                        attrs={"epsilon":epsilon})
    graph = gs.Graph(nodes=[layernorm], inputs=[tokEmb, wordEmb, posEmb, tokenIds, wordIds, posIds,  gamma, beta], outputs=[y])
    onnx.save(gs.export_onnx(graph), onnx_file)
    return onnx_file

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = 1 << int(trt.BuilderFlag.FP16) if int(npDataType == np.float16) else 0
    profile = builder.create_optimization_profile()
    
    parser = trt.OnnxParser(network, logger)
    onnxFile = getLayerNormOnnx()
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing onnx file!")
    
    inputTensor0 = network.get_input(0)  # x
    inputTensor1 = network.get_input(1)  # x
    inputTensor2 = network.get_input(2)  # x
    inputTensor3 = network.get_input(3)  # x
    inputTensor4 = network.get_input(4)  # x
    inputTensor5 = network.get_input(5)  # x
    


    print("inputTensor0.name:{}".format(inputTensor0.name))
    print("inputTensor1.name:{}".format(inputTensor1.name))
    print("inputTensor2.name:{}".format(inputTensor2.name))
    print("inputTensor3.name:{}".format(inputTensor3.name))
    print("inputTensor4.name:{}".format(inputTensor4.name))
    print("inputTensor5.name:{}".format(inputTensor5.name))

    profile.set_shape(inputTensor0.name, (tokenW, nEmbedding), (tokenW, nEmbedding), (tokenW, nEmbedding))  
    profile.set_shape(inputTensor1.name, (wordW, nEmbedding), (wordW, nEmbedding), (wordW, nEmbedding))  
    profile.set_shape(inputTensor2.name, (posW, nEmbedding), (posW, nEmbedding), (posW, nEmbedding))  
    # profile.set_shape(inputTensor3.name, (1, nW, nEmbedding), (4, nW, nEmbedding), (10, nW, nEmbedding))  
    # profile.set_shape(inputTensor4.name, (1, nW, nEmbedding), (4, nW, nEmbedding), (10, nW, nEmbedding))  
    # profile.set_shape(inputTensor5.name, (1, nW, nEmbedding), (4, nW, nEmbedding), (10, nW, nEmbedding))  
    profile.set_shape(inputTensor3.name, (1, nW, 1), (4, nW, 1), (10, nW, 1))  
    profile.set_shape(inputTensor4.name, (1, nW, 1), (4, nW, 1), (10, nW, 1))  
    profile.set_shape(inputTensor5.name, (1, nW, 1), (4, nW, 1), (10, nW, 1))  
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[tokenW,nEmbedding])
    context.set_binding_shape(1,[wordW,nEmbedding])
    context.set_binding_shape(2,[posW,nEmbedding])
    # context.set_binding_shape(3,[nBS,nW,nEmbedding])
    # context.set_binding_shape(4,[nBS,nW,nEmbedding])
    # context.set_binding_shape(5,[nBS,nW,nEmbedding])
    context.set_binding_shape(3,[nBS,nW,1])
    context.set_binding_shape(4,[nBS,nW,1])
    context.set_binding_shape(5,[nBS,nW,1])

    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(tokenW,nEmbedding).astype(npDataType).reshape(tokenW,nEmbedding))
    bufferH.append( np.random.rand(wordW,nEmbedding).astype(npDataType).reshape(wordW,nEmbedding))
    bufferH.append( np.random.rand(posW,nEmbedding).astype(npDataType).reshape(posW,nEmbedding))  
    bufferH.append( np.random.randint(tokenW,size=(nBS,nW,1)).astype(idDataType).reshape(nBS,nW,1))
    bufferH.append( np.random.randint(wordW,size=(nBS,nW,1)).astype(idDataType).reshape(nBS,nW,1))
    bufferH.append( np.random.randint(posW,size=(nBS,nW,1)).astype(idDataType).reshape(nBS,nW,1))
    # bufferH.append( np.random.rand(nBS,nW,nEmbedding).astype(npDataType).reshape(nBS,nW,nEmbedding) * 2 - 1)
    # bufferH.append( np.random.rand(nBS,nW,nEmbedding).astype(npDataType).reshape(nBS,nW,nEmbedding) * 2 - 1)
    # bufferH.append( np.random.rand(nBS,nW,nEmbedding).astype(npDataType).reshape(nBS,nW,nEmbedding) * 2 - 1)

    bufferH.append(np.empty(context.get_binding_shape(6),dtype=trt.nptype(engine.get_binding_dtype(6))))

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
    temp2 = layerNormCPU(bufferH[:6])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()
