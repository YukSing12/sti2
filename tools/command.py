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
from typing import Iterable, Union, Optional, List
from typer import Option

app = typer.Typer()


def create_input_shapes(shapes: Iterable[Optional[Iterable[Union[int, str]]]]):
    input_shapes = []
    for i, shape in enumerate(shapes):
        if shape is not None:
            shape = "x".join(map(str, shape))
            input_shapes.append(f"read_file_0.tmp_{i}:{shape}")
    return ",".join(input_shapes)


def enable_working_path(working_path: str):
    sys.path.append(working_path)
    sys.path.append(f"{working_path}/src/python")
    os.chdir(working_path)


@app.command()
def build(
    enable_plugins: bool = True,
    rebuild: bool = False,
    plugins: List[str] = Option(None, help="Replace ops with these plugins."),
    working_path: str = ".",
):
    enable_working_path(working_path)
    if rebuild:
        clean_command = [
            "rm *.plan",
            "rm so/plugins.so",
            "rm -rf build",
            "mkdir build",
        ]
        rich.print(clean_command)
        sp.run("\n".join(clean_command), shell=True)

    cmake_command = []
    if not os.path.exists("build"):
        cmake_command.append("mkdir build")
    cmake_command.extend(["cd build", "cmake ..", f"make install -j{mp.cpu_count()}"])
    # TODO: run cmake when cmake changes and rm build if error
    rich.print(cmake_command)
    sp.run("\n".join(cmake_command), shell=True)

    # TODO: import module in other method
    # TODO: now use cl because typer Option class cannot directly use in python

    # for plugin in plugins:
    #     sp.run(
    #         f"python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx --{plugin}",
    #         shell=True,
    #     )

    sp.run(
        f"python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx --ln",
        shell=True,
    )
    sp.run(
        f"python src/python/modify_ERNIE.py --src model/model.onnx --dst model/modified_model.onnx",
        shell=True,
    )
    from src.python.onnx_opt.fuser import fuse

    fuse()

    from tools.onnx2trt import onnx2trt

    onnx2trt("./model/modified_model.onnx", "./Ernie.plan", False, 1, plugins)


@app.command()
def run(
    perf_mode=0,
    working_path: str = ".",
):
    enable_working_path(working_path)
    exec_file = "./bin/main"
    plan_file = "./Ernie.plan"
    # TODO: auto compilation
    if perf_mode == 1:
        sp.run(
            f"{exec_file} {plan_file} ./data/perf.test.txt ./perf.res.txt ./so",
            shell=True,
        )
    else:
        sp.run(
            f"{exec_file} {plan_file} ./data/label.test.txt ./label.res.txt ./so",
            shell=True,
        )


@app.command()
def test(trtexec: str = "trtexec", working_path: str = "."):
    enable_working_path(working_path)
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
            f"--plugins=./so/plugins.so",  # TODO:more plugins so
            # f"--plugins=./so/plugins/libMaskedSoftmaxPlugin.so",  # TODO:more plugins so
        ]
    )
    rich.print(trtexec_command)
    rich.print(" ".join(trtexec_command))
    sp.run(" ".join(trtexec_command), shell=True)

    # TODO: use python module
    run(perf_mode=0)
    run(perf_mode=1)
    evaluate_command = [
        "python src/python/utils/local_evaluate.py ./label.res.txt",
        "python src/python/utils/local_evaluate.py ./perf.res.txt",
    ]
    if os.path.exists("bin/main"):
        sp.run(
            "\n".join(
                [
                    "./bin/main ./Ernie.plan ./data/label.test.txt ./label.res.txt ./so",
                    "./bin/main ./Ernie.plan ./data/perf.test.txt ./perf.res.txt ./so",
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
