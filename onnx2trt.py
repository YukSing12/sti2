import os
import sys
import numpy as np
import argparse
import tensorrt as trt
import ctypes

from glob import glob
from time import time
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser('Export Ernie TensorRT', add_help=False)
    parser.add_argument('--onnx', type=str, help='Path of onnx file to load')
    parser.add_argument('--trt', default='', type=str, help='Path of trt engine to save')
    parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 mode or not, default is TF32 if it is supported')
    parser.add_argument('--log_level', default=1, type=int, help='Logger level. (0:VERBOSE, 1:INFO, 2:WARNING, 3:ERROR, 4:INTERNAL_ERROR)')

    parser.add_argument('--ln', action='store_true', default=True, help='Replace ops with LayernormPlugin or not')
    args = parser.parse_args()
    return args

args = get_args()
if args.onnx:
    onnxFile = args.onnx
else:
    onnxFile = './model/modified_model.onnx'

if args.trt == '':
    trtFile = './Ernie.plan'
else:
    trtFile = args.trt

if args.fp16:
    trtFile = trtFile.replace('.plan','_fp16.plan')

if args.ln:
    onnxFile = onnxFile.replace(".onnx", "_ln.onnx")
    
timeCacheFile = "./Ernie.cache"
soFileList = glob("./so/*.so")
useTimeCache = False

log_level = {0:trt.Logger.VERBOSE,
             1:trt.Logger.INFO,
             2:trt.Logger.WARNING,
             3:trt.Logger.ERROR,
             4:trt.Logger.INTERNAL_ERROR}

logger = trt.Logger(log_level[args.log_level])                                       # Logger Level: VERBOSE,INFO,WARNING,ERROR,INTERNAL_ERROR
if os.path.isfile(trtFile):                                                
    with open(trtFile, 'rb') as f:
        engineString = f.read()
    if engineString == None:
        print("Failed getting serialized engine!")
        exit(1)
    print("Succeeded getting serialized engine in {}!".format(trtFile))
else:                                                                     
    timeCache = b""
    if useTimeCache and os.path.isfile(timeCacheFile):
        with open(timeCacheFile, 'rb') as f:
            timeCache = f.read()
        if timeCache == None:
            print("Failed getting serialized timing cache!")
            exit(1)
        print("Succeeded getting serialized timing cache!")
    if len(soFileList) > 0:
        print("Find Plugin %s!"%soFileList)
    else:
        print("No Plugin!")
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)

    builder = trt.Builder(logger)                                         
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 24 << 30                   
    if useTimeCache:
        cache = config.create_timing_cache(timeCache)
        config.set_timing_cache(cache, False)
    # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT) | 1 << int(trt.TacticSource.CUDNN))
    if args.fp16:
        config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
        config.flags = 1 << int(trt.BuilderFlag.FP16)
    else:
        config.flags = 0
    # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, calibrationCount, (1, 1, imageHeight, imageWidth), cacheFile)
    
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file in {}!".format(onnxFile))
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing onnx file!")

    if args.fp16:
        # pass
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # if 'LayerNorm' in layer.name or 'SoftmaxPlugin' in layer.name:
            if 'LayerNorm' in layer.name:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)

    inputTensor = network.get_input(0)  # tmp_0
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 128, 1), (4, 128, 1), (10, 128, 1))             

    inputTensor = network.get_input(1)  # tmp_1
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 128, 1), (4, 128, 1), (10, 128, 1))        

    inputTensor = network.get_input(2)  # tmp_2
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 128, 1), (4, 128, 1), (10, 128, 1))        

    inputTensor = network.get_input(3)  # tmp_3
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 128, 1), (4, 128, 1), (10, 128, 1))        
    
    inputTensor = network.get_input(4)  # tmp_6
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))     

    inputTensor = network.get_input(5)  # tmp_7
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))

    inputTensor = network.get_input(6)  # tmp_8
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))

    inputTensor = network.get_input(7)  # tmp_9
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))

    inputTensor = network.get_input(8)  # tmp_10
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))

    inputTensor = network.get_input(9)  # tmp_11
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))

    inputTensor = network.get_input(10)  # tmp_12
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))

    inputTensor = network.get_input(11)  # tmp_13
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 1, 1), (4, 1, 1), (10, 1, 1))    
    config.add_optimization_profile(profile)

    t0 = time()
    engineString = builder.build_serialized_network(network, config)     
    t1 = time()
    print("%s timing cache, %f mins"%("With" if useTimeCache/60 else "Without",(t1-t0)))

    if useTimeCache and not os.path.isfile(timeCacheFile):
        timeCache = config.get_timing_cache()
        timeCacheString = timeCache.serialize()
        with open(timeCacheFile, 'wb') as f:
            f.write(timeCacheString)
            print("Succeeded saving .cache file!")

    if engineString == None:
        print("Failed getting serialized engine!")
        exit(1)
    print("Succeeded getting serialized engine!")
    with open(trtFile, 'wb') as f:                                         
        f.write(engineString)
        print("Succeeded saving .plan file in {}!".format(trtFile))