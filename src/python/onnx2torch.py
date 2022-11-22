import argparse
import sys
import onnx
import onnx_graphsurgeon as gs
import numpy as np


def get_args():
    parser = argparse.ArgumentParser('Export Weights of Ernie', add_help=False)
    parser.add_argument('--onnx', required=True, type=str, help='Path of onnx file to load')
    parser.add_argument('--npy', default='./model/Ernie.npy', type=str, help='Path of npy to save')

    args = parser.parse_args()
    return args

def dump_weights(nodes, save_file):
    rst = dict()
    for node in nodes:
        for inp_variable in node.inputs :
            if "helper" in inp_variable.name:
                continue
            if hasattr(inp_variable, "values"):
                print("Export {}".format(inp_variable.name))
                rst[inp_variable.name] = inp_variable.values
    np.save(save_file, rst)

if __name__ == "__main__":
    args = get_args()

    graph = gs.import_onnx(onnx.load(args.onnx))
    print("Nodes:{}".format(len(graph.nodes)))
    graph.fold_constants().cleanup()
    print("Nodes:{}".format(len(graph.nodes)))
    nodes = graph.nodes
    dump_weights(nodes, args.npy)
    exit(0)


