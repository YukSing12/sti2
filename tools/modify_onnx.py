import sys
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import numpy as np
import argparse
import onnxsim
import typer
from typer import Option
from typing import List

app = typer.Typer()


@app.command()
def modified_onnx(
    src_path: str = Option(..., "--src", help="Path of onnx file to load"),
    dst_path: str = Option(..., "--dst", help="Path of onnx file to save"),
    onnxsim: bool = Option(False, help="pre simplify onnx by onnxsim library"),
    debug: bool = Option(False, "-D", help="Enable debug mode"),
    dymshape: bool = Option(False, help="modify dim2 dynamic shape"),
    plugins: List[str] = Option(None, help="Replace ops with these plugins."),
):
    ENABLE_LAYERNORM_PLUGIN = "ln" in plugins
    ENABLE_ADDLAYERNORM_PLUGIN = "aln" in plugins
    ENABLE_SLICERESHAPE_PLUGIN = "slreshape" in plugins
    ENABLE_FUSING_ADDRELU = "addrelu" in plugins
    ENABLE_POSTEMBEDDING_PLUGIN = "postemb" in plugins
    DEBUG = debug
    SIM = onnxsim
    DYNAMIC = dymshape

    src_onnx_path = src_path
    dst_onnx_path = dst_path

    print("Load onnx model from {}".format(src_onnx_path))
    print(src_onnx_path)
    graph = gs.import_onnx(onnx.load(src_onnx_path))
    print("Nodes:{}".format(len(graph.nodes)))
    graph.fold_constants().cleanup()

    if DYNAMIC:
        for i in range(4):
            graph.inputs[i].shape = [-1, -1, 1]
        dst_onnx_path = dst_onnx_path.replace(".onnx", "_dymshape.onnx")

    nodes = graph.nodes
    nodes_dict = {}
    for node in nodes:
        name = node.name
        nodes_dict.update({name: node})

    passes = []
    onnx_opt_plugins = []
    sys.path.append("src/python")

    if ENABLE_LAYERNORM_PLUGIN:
        from onnx_opt.passes import LayernormPass

        passes.append(LayernormPass())

    if ENABLE_ADDLAYERNORM_PLUGIN:
        from onnx_opt.passes import TowOpPass

        if not ENABLE_LAYERNORM_PLUGIN:
            passes.append(LayernormPass())
        passes.append(TowOpPass((["Add"], ["Layernorm"])))

    if ENABLE_SLICERESHAPE_PLUGIN:
        from onnx_opt.passes import SliceReshapePass

        passes.append(SliceReshapePass())

    if ENABLE_FUSING_ADDRELU:
        from onnx_opt.passes import TowOpPass

        passes.append(TowOpPass((["Add"], ["Relu"])))

    if ENABLE_POSTEMBEDDING_PLUGIN:
        from onnx_opt.plugins import PostEmbeddingPlugin

        onnx_opt_plugins.append(PostEmbeddingPlugin())

    if "ms" in plugins:
        from onnx_opt.passes import MaskedSoftmaxPass

        passes.append(MaskedSoftmaxPass())

    for p in plugins:
        dst_onnx_path = dst_onnx_path.replace(".onnx", f"_{p}.onnx")

    from onnx_opt.passes import _deprecated_nodes_dict
    from onnx_opt.fuser import Fuser

    _deprecated_nodes_dict.update(nodes_dict)

    with Fuser(graph, passes=passes, plugins=onnx_opt_plugins) as fuser:
        fuser.fuse()
    if ENABLE_POSTEMBEDDING_PLUGIN:
        graph.inputs = graph.inputs[:5]
        graph.inputs[4].shape = (-1, 8)
        graph.inputs[4].dtype = np.int32
        graph.inputs[4].name = "read_file_0.tmp_6-13"

    if DEBUG:
        graph.cleanup().toposort()
        dst_onnx_path = "./model/debug.onnx"
        # graph.toposort()
    else:
        graph.cleanup().toposort()

    print("Nodes:{}".format(len(graph.nodes)))
    if SIM:
        onnx_model, check = onnxsim.simplify(gs.export_onnx(graph))
    else:
        onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, dst_onnx_path)
    # onnx.save(onnx.shape_inference.infer_shapes(onnx_model), dst_onnx_path)
    print("Save modified onnx model to {}".format(dst_onnx_path))


if __name__ == "__main__":
    app()
