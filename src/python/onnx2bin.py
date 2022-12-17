import argparse
import sys
import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import configparser


def get_args():
    parser = argparse.ArgumentParser('Export Weights of Ernie', add_help=False)
    parser.add_argument('--onnx', required=True, type=str, help='Path of onnx file to load')
    parser.add_argument('--bin', default='./model/bin',
                        type=str, help='Path of bin')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 mode or not, default is FP32')
    args = parser.parse_args()
    return args

def dump_weights(nodes):
    rst = dict()
    for node in nodes:
        for inp_variable in node.inputs :
            if "helper" in inp_variable.name:
                continue
            if hasattr(inp_variable, "values"):
                print("Export {}".format(inp_variable.name))
                rst[inp_variable.name] = inp_variable.values
    return rst

if __name__ == "__main__":
    args = get_args()
    os.system("mkdir -p " + args.bin)
    
    graph = gs.import_onnx(onnx.load(args.onnx))
    print("Nodes:{}".format(len(graph.nodes)))
    graph.fold_constants().cleanup()
    print("Nodes:{}".format(len(graph.nodes)))
    nodes = graph.nodes
    weights=dump_weights(nodes)
    
    conf = configparser.ConfigParser()
    conf.add_section("ernie")
    with open(os.path.join(args.bin, "config.ini"), 'w') as fid:
        if args.fp16:
            print("Extract weights in FP16 mode")
            conf.set("ernie", "weight_data_type", "fp16")
            npDataType = np.float16
        else:
            print("Extract weights in FP32 mode")
            conf.set("ernie", "weight_data_type", "fp32")
            npDataType = np.float32
        conf.write(fid)
    
    
    for name,value in weights.items():
        saved_path = os.path.join(args.bin, name+".bin")
        print(name, value.shape)
        value.astype(npDataType).tofile(saved_path)
    print("Succeed extracting weights of Ernie!")


