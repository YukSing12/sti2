import sys
import onnx
import onnx_graphsurgeon as gs
import argparse
import onnxsim
import numpy as np

def get_args():
    parser = argparse.ArgumentParser("Export ERNIE TensorRT", add_help=False)
    parser.add_argument(
        "--src", required=True, type=str, help="Path of onnx file to load"
    )
    parser.add_argument(
        "--dst", required=True, type=str, help="Path of onnx file to save"
    )
    parser.add_argument(
        '--dymshape', 
        action='store_true', 
        default=False, 
        help='modify dim2 dynamic shape'
    )
    parser.add_argument(
        "--onnxsim",
        action="store_true",
        default=False,
        help="pre simplify onnx by onnxsim library",
    )
    parser.add_argument(
        "--ln",
        action="store_true",
        default=False,
        help="Replace ops with LayernormPlugin or not",
    )
    parser.add_argument(
        "--aln",
        action="store_true",
        default=False,
        help="Replace ops with LayernormPlugin or not",
    )
    parser.add_argument(
        "--slreshape",
        action="store_true",
        default=False,
        help="Replace ops with SliceReshapePlugin or not",
    )
    parser.add_argument(
        "--addrelu",
        action="store_true",
        default=False,
        help="Replace ops with AddReluPlugin or not",
    )
    parser.add_argument(
        "--postemb",
        action="store_true",
        default=False,
        help="Replace ops with PostEmbeddingPlugin or not",
    )
    parser.add_argument(
        "--debug", "-D", action="store_true", default=False, help="Enable debug mode"
    )
    args = parser.parse_args()
    return args


args = get_args()
ENABLE_LAYERNORM_PLUGIN = args.ln
ENABLE_ADDLAYERNORM_PLUGIN = args.aln
ENABLE_SLICERESHAPE_PLUGIN = args.slreshape
ENABLE_FUSING_ADDRELU = args.addrelu
ENABLE_POSTEMBEDDING_PLUGIN = args.postemb
DEBUG = args.debug
SIM = args.onnxsim
DYNAMIC =args.dymshape
src_onnx_path = args.src
dst_onnx_path = args.dst

print("Load onnx model from {}".format(src_onnx_path))
graph = gs.import_onnx(onnx.load(src_onnx_path))
print("Nodes:{}".format(len(graph.nodes)))
graph.fold_constants().cleanup()

if DYNAMIC:
    for i in range(4):
        graph.inputs[i].shape=[-1,-1,1]
    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_dymshape.onnx")    
    
nodes = graph.nodes
nodes_dict = {}
for node in nodes:
    name = node.name
    nodes_dict.update({name: node})

passes = []
sys.path.append("src/python")

if ENABLE_LAYERNORM_PLUGIN:
    from onnx_opt.passes import LayernormPass

    passes.append(LayernormPass())
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_ln.onnx")

if ENABLE_ADDLAYERNORM_PLUGIN:
    from onnx_opt.passes import TowOpPass

    if not ENABLE_LAYERNORM_PLUGIN:
        passes.append(LayernormPass())
    passes.append(TowOpPass((["Add"], ["Layernorm"])))
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_aln.onnx")

if ENABLE_SLICERESHAPE_PLUGIN:
    from onnx_opt.passes import SliceReshapePass

    passes.append(SliceReshapePass())
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_slreshape.onnx")

if ENABLE_FUSING_ADDRELU:
    from onnx_opt.passes import TowOpPass

    passes.append(TowOpPass((["Add"], ["Relu"])))
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_addrelu.onnx")

if ENABLE_POSTEMBEDDING_PLUGIN:
    from onnx_opt.passes import PostEmbeddingPass

    passes.append(PostEmbeddingPass())
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_postemb.onnx")

from onnx_opt.passes import _deprecated_nodes_dict
from onnx_opt.fuser import Fuser

_deprecated_nodes_dict.update(nodes_dict)

fuser = Fuser(graph, passes=passes)
fuser.fuse()
if ENABLE_POSTEMBEDDING_PLUGIN:
    graph.inputs=graph.inputs[:5]
    graph.inputs[4].shape=(-1,8)
    graph.inputs[4].dtype=np.int32
    graph.inputs[4].name="read_file_0.tmp_6-13"
    
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
