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

def dump_dequantize_node(node, rst):
    name = node.inputs[0].name.replace("w_0_0", "w_0")
    print("Export {}".format(name))
    rst[name] = node.inputs[0].values
    name = node.inputs[0].name.replace("w_0_0", "w_0_scales")
    print("Export {}".format(name))
    rst[name] = node.inputs[1].values

def dump_weights(nodes, nodes_dict):
    rst = dict()
    layer_num = 0
    for node in nodes:
        if node.op == "Add" and len(node.outputs[0].outputs) == 7:
            q_mm = nodes_dict["p2o.MatMul.{}".format(2 + layer_num * 16)]
            k_mm = nodes_dict["p2o.MatMul.{}".format(4 + layer_num * 16)]
            v_mm = nodes_dict["p2o.MatMul.{}".format(6 + layer_num * 16)]
            proj_mm = nodes_dict["p2o.MatMul.{}".format(12 + layer_num * 16)]
            ffn1_mm = nodes_dict["p2o.MatMul.{}".format(14 + layer_num * 16)]
            ffn2_mm = nodes_dict["p2o.MatMul.{}".format(16 + layer_num * 16)]
            
            dump_dequantize_node(q_mm.inputs[1].inputs[0], rst)
            dump_dequantize_node(k_mm.inputs[1].inputs[0], rst)
            dump_dequantize_node(v_mm.inputs[1].inputs[0], rst)
            dump_dequantize_node(proj_mm.inputs[1].inputs[0], rst)
            dump_dequantize_node(ffn1_mm.inputs[1].inputs[0], rst)
            dump_dequantize_node(ffn2_mm.inputs[1].inputs[0], rst)
            
            layer_num += 1
            continue
    return rst

if __name__ == "__main__":
    args = get_args()
    os.system("mkdir -p " + args.bin)

    graph = gs.import_onnx(onnx.load(args.onnx))
    print("Nodes:{}".format(len(graph.nodes)))
    graph.fold_constants().cleanup()
    print("Nodes:{}".format(len(graph.nodes)))
    nodes = graph.nodes
    nodes_dict = {}
    for node in nodes:
        name = node.name
        nodes_dict.update({name: node})
    weights=dump_weights(nodes, nodes_dict)
    
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


