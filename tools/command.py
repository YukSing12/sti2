#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/18 21:48
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import sys

import typer
import subprocess as sp
import multiprocessing as mp
import os
import rich
from typing import Iterable, Union, Optional

app = typer.Typer()


def create_input_shapes(shapes: Iterable[Optional[Iterable[Union[int, str]]]]):
    input_shapes = []
    for i, shape in enumerate(shapes):
        if shape is not None:
            shape = "x".join(map(str, shape))
            input_shapes.append(f"read_file_0.tmp_{i}:{shape}")
    return ",".join(input_shapes)


@app.command()
def build(working_path: str = ".", enable_plugins: bool = True, rebuild: bool = False):
    sys.path.append(working_path)
    sys.path.append(f"{working_path}/src/python")
    if rebuild:
        clean_command = [
            f"cd {working_path}",
            "rm *.plan",
            "rm -rf build",
            "mkdir build",
        ]
        sp.run("\n".join(clean_command), shell=True)

    cmake_command = [f"cd {working_path}"]
    if not os.path.exists("build"):
        cmake_command.append("mkdir build")
    cmake_command.extend(["cd build", "cmake ..", f"make install -j{mp.cpu_count()}"])
    # TODO: run cmake when cmake changes and rm build if error
    rich.print(cmake_command)
    sp.run("\n".join(cmake_command), shell=True)

    # TODO: import module in other method
    # TODO: now use cl because typer Option class cannot directly use in python

    sp.run(
        "python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx --ln",
        shell=True,
    )

    # from tools.onnx2trt import onnx2trt
    # onnx2trt(plugins=["ln"])
    plugins = ["ln"]
    onnx2trt_command = [
        "python tools/onnx2trt.py",
        *[f"--plugins {plugin}" for plugin in plugins],
    ]
    rich.print(onnx2trt_command)
    sp.run(" ".join(onnx2trt_command), shell=True)


@app.command()
def run(working_path: str = None, perf_mode=0):
    # TODO: auto compilation
    if perf_mode == 1:
        sp.run(
            "./bin/main ./Ernie.plan ./data/perf.test.txt ./perf.res.txt ./so/plugins/",
            shell=True,
        )
    else:
        sp.run(
            "./bin/main ./Ernie.plan ./data/label.test.txt ./label.res.txt ./so/plugins/",
            shell=True,
        )


@app.command()
def test(working_path: str = None, trtexec: str = "trtexec"):
    # TODO: auto compilation
    typer.echo("Evaluate performance of model")
    trtexec_command = [f"{trtexec}"]
    min_shapes = [
        *[[1, 128, 1] for _ in range(4)],
        None,
        None,
        *[[1, 1, 1] for _ in range(8)],
    ]
    opt_shapes = [
        *[[4, 128, 1] for _ in range(4)],
        None,
        None,
        *[[4, 1, 1] for _ in range(8)],
    ]
    max_shapes = [
        *[[10, 128, 1] for _ in range(4)],
        None,
        None,
        *[[10, 1, 1] for _ in range(8)],
    ]
    trtexec_command.extend(
        [
            "--loadEngine=Ernie.plan",
            "--avgRuns=1000",
            "--profilingVerbosity=detailed",
            f"--minShapes={create_input_shapes(min_shapes)}",
            f"--optShapes={create_input_shapes(opt_shapes)}",
            f"--maxShapes={create_input_shapes(max_shapes)}",
            # f"--plugins=./so/plugins/libAddReluPlugin.so",  # TODO:more plugins so
            f"--plugins=./so/plugins/libLayerNormPlugin.so",  # TODO:more plugins so
            # f"--plugins=./so/plugins/libMaskedSoftmaxPlugin.so",  # TODO:more plugins so
            # f"--plugins=./so/plugins.so",  # TODO:plugins.so
        ]
    )
    print(" ".join(trtexec_command))
    sp.run(" ".join(trtexec_command), shell=True)

    # TODO: use python module
    if os.path.exists("bin/main"):
        sp.run(
            "\n".join(
                [
                    "./bin/main ./Ernie.plan ./data/label.test.txt ./label.res.txt ./so/plugins",
                    "./bin/main ./Ernie.plan ./data/perf.test.txt ./perf.res.txt ./so/plugins",
                    "python src/python/utils/local_evaluate.py ./label.res.txt",
                    "python src/python/utils/local_evaluate.py ./perf.res.txt",
                ]
            ),
            shell=True,
        )


def auto_test(config: str = "tests/auto_test.yaml"):
    pass


if __name__ == "__main__":
    app()
