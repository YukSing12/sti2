import os
import tensorrt as trt
import ctypes
from glob import glob
from time import time
import typer
from typer import Option
from typing import List


def onnx2trt(
    onnx_file: str = Option(
        "./model/modified_model.onnx", help="Path of onnx _file to load"
    ),
    trt_file: str = Option("./Ernie.plan", help="Path of trt engine to save"),
    fp16: bool = Option(
        False, help="Enable FP16 mode or not, default is TF32 if it is supported"
    ),
    log_level: int = Option(
        1,
        help="Logger level. (0:VERBOSE, 1:INFO, 2:WARNING, 3:ERROR, 4:INTERNAL_ERROR)",
    ),
    plugins: List[str] = Option(None, help="Replace ops with these plugins."),
):
    if fp16:
        trt_file = trt_file.replace(".plan", "_fp16.plan")

    if "ln" in plugins:
        onnx_file = onnx_file.replace(".onnx", "_ln.onnx")

    timeCache_file = "./Ernie.cache"
    so_fileList = glob("./so/plugins/*.so")
    useTimeCache = False

    log_level = {
        0: trt.Logger.VERBOSE,
        1: trt.Logger.INFO,
        2: trt.Logger.WARNING,
        3: trt.Logger.ERROR,
        4: trt.Logger.INTERNAL_ERROR,
    }[log_level]

    logger = trt.Logger(
        log_level
    )  # Logger Level: VERBOSE,INFO,WARNING,ERROR,INTERNAL_ERROR
    if os.path.isfile(trt_file):
        with open(trt_file, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            exit(1)
        print("Succeeded getting serialized engine in {}!".format(trt_file))
    else:
        timeCache = b""
        if useTimeCache and os.path.isfile(timeCache_file):
            with open(timeCache_file, "rb") as f:
                timeCache = f.read()
            if timeCache == None:
                print("Failed getting serialized timing cache!")
                exit(1)
            print("Succeeded getting serialized timing cache!")
        if len(so_fileList) > 0:
            print("Find Plugin %s!" % so_fileList)
        else:
            print("No Plugin!")
        for so_file in so_fileList:
            ctypes.cdll.LoadLibrary(so_file)

        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 24 << 30
        if useTimeCache:
            cache = config.create_timing_cache(timeCache)
            config.set_timing_cache(cache, False)
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT) | 1 << int(trt.TacticSource.CUDNN))
        if fp16:
            config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
            config.flags = 1 << int(trt.BuilderFlag.FP16)
        else:
            config.flags = 0
        # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, calibrationCount, (1, 1, imageHeight, imageWidth), cache_file)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnx_file):
            print("Failed finding onnx file in {}!".format(onnx_file))
            exit()
        print("Succeeded finding onnx file!")
        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing onnx file!")

        if fp16:
            # pass
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                # if 'LayerNorm' in layer.name or 'SoftmaxPlugin' in layer.name:
                if "LayerNorm" in layer.name:
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
        print(
            "%s timing cache, %f mins"
            % ("With" if useTimeCache / 60 else "Without", (t1 - t0))
        )

        if useTimeCache and not os.path.isfile(timeCache_file):
            timeCache = config.get_timing_cache()
            timeCacheString = timeCache.serialize()
            with open(timeCache_file, "wb") as f:
                f.write(timeCacheString)
                print("Succeeded saving .cache file!")

        if engineString == None:
            print("Failed getting serialized engine!")
            exit(1)
        print("Succeeded getting serialized engine!")
        with open(trt_file, "wb") as f:
            f.write(engineString)
            print("Succeeded saving .plan file in {}!".format(trt_file))


if __name__ == "__main__":
    typer.run(onnx2trt)
