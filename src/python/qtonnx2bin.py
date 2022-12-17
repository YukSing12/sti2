import argparse
import sys
import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import configparser

ACTIVATION_AMAX_NUM = 72
INT8O_GEMM_NUM = 8
TRT_AMAX_NUM = 3
SCALE_RESERVE_NUM = 21
HIDDEN_DIM = 768

def get_args():
    parser = argparse.ArgumentParser('Export Weights of Ernie', add_help=False)
    parser.add_argument('--onnx', required=True, type=str, help='Path of onnx file to load')
    parser.add_argument('--bin', default='./model/bin',
                        type=str, help='Path of bin')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 mode or not, default is FP32')
    args = parser.parse_args()
    return args

def fill_part1_amax(scale_list, amax):
    scale_list[0] = amax
    scale_list[1] = amax / 127.0
    scale_list[2] = amax / 127.0 / 127.0
    scale_list[3] = 127.0 / amax
    return scale_list

def dump_scales(nodes, nodes_dict):
    rst = dict()
    layer_num = 0
    for node in nodes:
        if node.op == "Add" and len(node.outputs[0].outputs) == 4:
            name = "encoder_layer_{}_scale_list".format(layer_num)
            q_flatten = nodes_dict["p2o.Flatten.{}".format(0 + layer_num * 12)]
            k_flatten = nodes_dict["p2o.Flatten.{}".format(2 + layer_num * 12)]
            v_flatten = nodes_dict["p2o.Flatten.{}".format(4 + layer_num * 12)]

            q_mm = nodes_dict["p2o.MatMul.{}".format(2 + layer_num * 16)]
            k_mm = nodes_dict["p2o.MatMul.{}".format(4 + layer_num * 16)]
            v_mm = nodes_dict["p2o.MatMul.{}".format(6 + layer_num * 16)]

            q_bias = nodes_dict["p2o.Add.{}".format(10 + layer_num * 26)]
            k_bias = nodes_dict["p2o.Add.{}".format(12 + layer_num * 26)]
            v_bias = nodes_dict["p2o.Add.{}".format(14 + layer_num * 26)]

            bmm1 = nodes_dict["p2o.MatMul.{}".format(8 + layer_num * 16)]
            softmax = nodes_dict["p2o.Softmax.{}".format(0 + layer_num * 1)]
            bmm2 = nodes_dict["p2o.MatMul.{}".format(10 + layer_num * 16)]

            proj_mm = nodes_dict["p2o.MatMul.{}".format(12 + layer_num * 16)]
            proj_bias_norm = nodes_dict["p2o.Add.{}".format(24 + layer_num * 26)]

            fc1_mm = nodes_dict["p2o.MatMul.{}".format(14 + layer_num * 16)]
            fc1_bias = nodes_dict["p2o.Add.{}".format(26 + layer_num * 26)]
            fc2_mm = nodes_dict["p2o.MatMul.{}".format(16 + layer_num * 16)]
            fc2_bias_norm = nodes_dict["p2o.Add.{}".format(34 + layer_num * 26)]

            size = ACTIVATION_AMAX_NUM + 9 * HIDDEN_DIM + INT8O_GEMM_NUM + TRT_AMAX_NUM
            scale_list = np.zeros(size)

            # Part 1 -- 72:
            #   First 72 are for activation amaxs. For each activation amax, there are 4 values: amax, amax/127.0f,
            #   amax/127.0f/127.0f, 127.0f/amax -- input_amax 0-3 , Q_aftergemm_amax 4-7, Qbias_amax 8-11, K_aftergemm_amax
            #   12-15, Kbias_amax 16-19, V_aftergemm_amax 20-23, Vbias_amax 24-27, bmm1_amax 28-31, Softmax_amax 32-35,
            #   bmm2_amax 36-39, Proj_aftergemm_scale 40-43, ProjBiasNorm_amax 44-47, FC1_aftergemm_amax 48-51, F1Bias_amax
            #   52-55, FC2_aftergemm_amax 56-59, F2BiasNorm_amax 60-63, reserve 64-71
            
            # input_amax
            input_amax = q_flatten.outputs[0].outputs[0].inputs[1].values * 127
            idx = 0
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], input_amax)

            # Q_aftergemm_amax
            q_aftergemm_amax = q_mm.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], q_aftergemm_amax)

            # Qbias_amax
            q_bias_amax = q_bias.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], q_bias_amax)

            # K_aftergemm_amax
            k_aftergemm_amax = k_mm.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], k_aftergemm_amax)

            # Kbias_amax
            k_bias_amax = k_bias.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], k_bias_amax)
            
            # V_aftergemm_amax
            v_aftergemm_amax = v_mm.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], v_aftergemm_amax)

            # Vbias_amax
            v_bias_amax = v_bias.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], v_bias_amax)

            # bmm1_amax
            bmm1_amax = bmm1.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], bmm1_amax)

            # Softmax_amax
            softmax_amax = softmax.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], softmax_amax)

            # bmm2_amax
            bmm2_amax = bmm2.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], bmm2_amax)

            # Proj_aftergemm_scale
            proj_aftergemm_amax = proj_mm.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], proj_aftergemm_amax)

            # ProjBiasNorm_amax
            proj_bias_norm_amax = proj_bias_norm.outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], proj_bias_norm_amax)

            # FC1_aftergemm_amax
            fc1_amax = fc1_mm.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], fc1_amax)

            # F1Bias_amax
            fc1_bias_amax = fc1_bias.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], fc1_bias_amax)

            # FC2_aftergemm_amax
            fc2_amax = fc2_mm.outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], fc2_amax)

            # F2BiasNorm_amax
            if(layer_num == 11):    # A little different in the finaly layer
                fc2_bias_norm_amax = fc2_bias_norm.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].values * 127
            else:
                fc2_bias_norm_amax = fc2_bias_norm.outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].values * 127
            idx += 1
            fill_part1_amax(scale_list[idx * 4: idx * 4 + 4], fc2_bias_norm_amax)

            # Part 2 -- 9*hidden_dim:
            #   Kernel amaxs, for each kernel amax list, there are output_channel values : query_weight_amax_list,
            #   key_weight_amax_list, value_weight_amax_list, proj_weight_amax_list, FC1_weight_amax_list, FC2_weight_amax_list
            p2_offset = ACTIVATION_AMAX_NUM

            # query_weight_amax_list
            query_weight_amax = q_mm.inputs[1].inputs[0].inputs[1].values
            idx = 0
            scale_list[p2_offset + idx * HIDDEN_DIM: p2_offset + idx * HIDDEN_DIM + HIDDEN_DIM] = query_weight_amax

            # key_weight_amax_list
            key_weight_amax = k_mm.inputs[1].inputs[0].inputs[1].values
            idx += 1
            scale_list[p2_offset + idx * HIDDEN_DIM: p2_offset + idx * HIDDEN_DIM + HIDDEN_DIM] = key_weight_amax

            # value_weight_amax_list
            value_weight_amax = v_mm.inputs[1].inputs[0].inputs[1].values
            idx += 1
            scale_list[p2_offset + idx * HIDDEN_DIM: p2_offset + idx * HIDDEN_DIM + HIDDEN_DIM] = value_weight_amax

            # proj_weight_amax_list
            proj_weight_amax = proj_mm.inputs[1].inputs[0].inputs[1].values
            idx += 1
            scale_list[p2_offset + idx * HIDDEN_DIM: p2_offset + idx * HIDDEN_DIM + HIDDEN_DIM] = proj_weight_amax

            # fc1_weight_amax_list
            fc1_weight_amax = fc1_mm.inputs[1].inputs[0].inputs[1].values
            idx += 1
            scale_list[p2_offset + idx * HIDDEN_DIM: p2_offset + idx * HIDDEN_DIM + HIDDEN_DIM] = fc1_weight_amax

            # fc2_weight_amax_list (4 x HIDDEN_DIM)
            fc2_weight_amax = fc2_mm.inputs[1].inputs[0].inputs[1].values
            idx += 1
            scale_list[p2_offset + idx * HIDDEN_DIM: p2_offset + idx * HIDDEN_DIM + HIDDEN_DIM * 4] = fc2_weight_amax

            # Part 3 -- 8:
            #   Int8 gemm deQFactor list (8 values): Q_deQ_scale, K_deQ_scale, V_deQ_scale, bmm1_deQ_scale, bmm2_deQ_scale,
            #   FC0_deQ_scale, FC1_deQ_scale, FC2_deQ_scale
            p3_offset_    = ACTIVATION_AMAX_NUM + 9 * 768
            
            # Q_deQ_scale
            q_deq_scale = q_mm.outputs[0].outputs[0].inputs[1].values
            idx = 0
            scale_list[p3_offset_ + idx] = q_deq_scale

            # K_deQ_scale
            k_deq_scale = k_mm.outputs[0].outputs[0].inputs[1].values
            idx += 1
            scale_list[p3_offset_ + idx] = k_deq_scale

            # V_deQ_scale
            v_deq_scale = v_mm.outputs[0].outputs[0].inputs[1].values
            idx += 1
            scale_list[p3_offset_ + idx] = v_deq_scale

            # bmm1_deQ_scale
            bmm1_deq_scale = bmm1.outputs[0].outputs[0].inputs[1].values
            idx += 1
            scale_list[p3_offset_ + idx] = bmm1_deq_scale

            # bmm2_deQ_scale
            bmm2_deq_scale = bmm2.outputs[0].outputs[0].inputs[1].values
            idx += 1
            scale_list[p3_offset_ + idx] = bmm2_deq_scale

            # FC0_deQ_scale
            fc1_mm_deq_scale = fc1_mm.outputs[0].outputs[0].inputs[1].values
            idx += 1
            scale_list[p3_offset_ + idx] = fc1_mm_deq_scale

            # FC1_deQ_scale
            fc2_mm_deq_scale = fc2_mm.outputs[0].outputs[0].inputs[1].values
            idx += 1
            scale_list[p3_offset_ + idx] = fc2_mm_deq_scale
            
            # FC2_deQ_scale = ?
            

            # Part 4 -- 3:
            #   Amax used in trt fused mha kernel (3 values) : QKVbias_amax, Softmax_amax, bmm2_amax
            p4_offset_    = ACTIVATION_AMAX_NUM + 9 * 768 + INT8O_GEMM_NUM
            QKVbias_amax = np.max([q_bias_amax, k_bias_amax, v_bias_amax])
            scale_list[p4_offset_] = QKVbias_amax
            scale_list[p4_offset_ + 1] = softmax_amax
            scale_list[p4_offset_ + 2] = bmm2_amax

            # Part 5 -- 21: reverse
            layer_num += 1
            print("Export {}".format(name))
            rst[name] = scale_list
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
    weights=dump_scales(nodes, nodes_dict)
    
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


