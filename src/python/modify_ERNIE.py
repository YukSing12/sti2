import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import numpy as np
import argparse
import onnxsim

def get_args():
    parser = argparse.ArgumentParser('Export ERNIE TensorRT', add_help=False)
    parser.add_argument('--src', required=True, type=str, help='Path of onnx file to load')
    parser.add_argument('--dst', required=True, type=str, help='Path of onnx file to save')
    parser.add_argument('--dymshape', action='store_true', default=False, help='modify dim2 dynamic shape')
    parser.add_argument('--onnxsim', action='store_true', default=False, help='pre simplify onnx by onnxsim library')
    parser.add_argument('--ln', action='store_true', default=False, help='Replace ops with LayernormPlugin or not')
    parser.add_argument('--aln', action='store_true', default=False, help='Replace ops with LayernormPlugin or not')
    parser.add_argument('--slreshape', action='store_true', default=False, help='Replace ops with SliceReshapePlugin or not')
    parser.add_argument('--addrelu', action='store_true', default=False, help='Replace ops with AddReluPlugin or not')
    parser.add_argument('--postemb', action='store_true', default=False, help='Replace ops with PostEmbeddingPlugin or not')
    parser.add_argument('--debug', '-D', action='store_true', default=False, help='Enable debug mode')
    args = parser.parse_args()
    return args

args = get_args()
ENABLE_LAYERNORM_PLUGIN = args.ln
ENABLE_ADDLAYERNORM_PLUGIN = args.aln
ENABLE_SLICERESHAPE_PLUGIN = args.slreshape
ENABLE_FUSING_ADDRELU = args.addrelu
ENABLE_POSTEMBEDDING_PLUGIN = args.postemb
DEBUG = args.debug
SIM=args.onnxsim
DYNAMIC=args.dymshape
def replace_with_layernorm(nodes_dict, mean_node):
    node_id = int(mean_node.name.split(".")[-1])
    if not (('p2o.Sub.{}'.format(node_id//2) in nodes_dict)
            and ('p2o.Pow.{}'.format(node_id//2) in nodes_dict)
            and ('p2o.Div.{}'.format(node_id//2) in nodes_dict)
            and ('p2o.Sqrt.{}'.format(node_id//2) in nodes_dict)):
        return None

    sub_node = nodes_dict['p2o.Sub.{}'.format(node_id//2)]
    div_node = nodes_dict['p2o.Div.{}'.format(node_id//2)]
    mul_node = div_node.outputs[0].outputs[0]
    add_node = mul_node.outputs[0].outputs[0]

    gamma = mul_node.inputs[1]
    beta = add_node.inputs[1]


    name = 'LayerNorm.{}'.format(node_id)
    layernorm = gs.Node(op="LayerNorm",
                        name=name,
                        inputs=[mean_node.inputs[0],gamma,beta],
                        outputs=[add_node.outputs[0]],
                        attrs={"epsilon": 1e-5})
    mean_node.inputs.clear()
    sub_node.inputs.clear()
    add_node.outputs.clear()
    return layernorm

def replace_with_addlayernorm(nodes_dict, mean_node):
    node_id = int(mean_node.name.split(".")[-1])
    if not (('p2o.Sub.{}'.format(node_id//2) in nodes_dict)  
    and ('p2o.Pow.{}'.format(node_id//2) in nodes_dict) 
    and ('p2o.Div.{}'.format(node_id//2) in nodes_dict)  
    and ('p2o.Sqrt.{}'.format(node_id//2) in nodes_dict)):
        return None

    add1_node = mean_node.inputs[0].inputs[0]
    add2_node = add1_node.inputs[0].inputs[0]
    # sub_node = nodes_dict['p2o.Sub.{}'.format(node_id//2)]
    div_node = nodes_dict['p2o.Div.{}'.format(node_id//2)]
    mul_node = div_node.outputs[0].outputs[0]
    add3_node = mul_node.outputs[0].outputs[0]

    gamma = mul_node.inputs[1]
    beta = add3_node.inputs[1]

    
    name = 'AddLayerNorm.{}'.format(node_id)
    addlayernorm = gs.Node(op="AddLayerNorm", 
                        name=name, 
                        inputs=[add1_node.inputs[1],add2_node.inputs[0],add2_node.inputs[1],gamma,beta], 
                        outputs=[add3_node.outputs[0]], 
                        attrs={"epsilon": 1e-5})
    add1_node.inputs.clear()
    add2_node.inputs.clear()
    # sub_node.inputs.clear()
    add3_node.outputs.clear()
    return addlayernorm

def replace_with_slice_reshape(nodes_dict, shape_node):
    node_id = int(shape_node.name.split(".")[-1])
    if not (('p2o.Slice.{}'.format(node_id//2) in nodes_dict)):
        return None

    slice_node = shape_node.outputs[0].outputs[0]

    # slice_node = nodes_dict['p2o.Slice.{}'.format(node_id//2)]
    # slice2_node = nodes_dict['p2o.Slice.{}'.format(node_id//2 + 1)]
    # shape2_node = slice2_node.inputs[0].inputs[0]
    concat_node = slice_node.outputs[0].outputs[0]
    reshape_node = concat_node.outputs[0].outputs[0]
    flatten_node = reshape_node.inputs[0].inputs[0].inputs[0].inputs[0]
    add_node = reshape_node.outputs[0].outputs[0]
    name = 'SliceReshape.{}'.format(node_id)
    slicereshape = gs.Node(op="SliceReshape",
                           name=name,
                           inputs=[shape_node.inputs[0],concat_node.inputs[1], flatten_node.inputs[0]],
                           outputs=[add_node.outputs[0]],
                           attrs={"epsilon": 1e-6})
    shape_node.inputs.clear()
    concat_node.inputs.clear()
    flatten_node.inputs.clear()
    add_node.outputs.clear()
    return slicereshape

def fuse_add_relu(nodes_dict, root_node):
    add_node = root_node.inputs[0].inputs[0]
    node_id = int(root_node.name.split(".")[-1])
    if ((add_node.op != 'Add')):
        return None

    if add_node.inputs[1].shape[0] != 3072:
        return None

    name = 'AddRelu.{}'.format(node_id)
    add_relu = gs.Node(op="AddRelu",
                       name=name,
                       inputs=[add_node.inputs[0],add_node.inputs[1]],
                       outputs=[root_node.outputs[0]])

    add_node.inputs.clear()
    root_node.outputs.clear()
    return add_relu
     
def replace_with_postembedding(nodes_dict, graph_inputs):
    squeeze_0 = graph_inputs[0].outputs[0].outputs[0].outputs[0]
    squeeze_1 = graph_inputs[1].outputs[0].outputs[0].outputs[0]
    squeeze_2 = graph_inputs[2].outputs[0].outputs[0].outputs[0]
    squeeze_3 = graph_inputs[3].outputs[0].outputs[0].outputs[0]
    squeeze_4 = graph_inputs[4].outputs[0].outputs[0].outputs[0]
    squeeze_5 = graph_inputs[5].outputs[0].outputs[0].outputs[0]
    squeeze_6 = graph_inputs[6].outputs[0].outputs[0].outputs[0]
    squeeze_7 = graph_inputs[7].outputs[0].outputs[0].outputs[0]

    emb_0 = squeeze_0.outputs[0].outputs[0].inputs[0]
    emb_1 = squeeze_1.outputs[0].outputs[0].inputs[0]
    emb_2 = squeeze_2.outputs[0].outputs[0].inputs[0]
    emb_3 = squeeze_3.outputs[0].outputs[0].inputs[0]
    emb_4 = squeeze_4.outputs[0].outputs[0].inputs[0]
    emb_5 = squeeze_5.outputs[0].outputs[0].inputs[0]
    emb_6 = squeeze_6.outputs[0].outputs[0].inputs[0]
    emb_7 = squeeze_7.outputs[0].outputs[0].inputs[0]

    reshape_node = graph_inputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0]
    output = reshape_node.outputs[0]
    graph_inputs[0].shape=(-1,8)
    graph_inputs[0].dtype=np.float32
    posemb = gs.Node(op="PostEmbedding",
                       name="PostEmbedding",
                       inputs=[graph_inputs[0],emb_0, emb_1, emb_2, emb_3, emb_4, emb_5, emb_6, emb_7],
                       outputs=[output])
    squeeze_0.inputs.clear()
    squeeze_1.inputs.clear()
    squeeze_2.inputs.clear()
    squeeze_3.inputs.clear()
    squeeze_4.inputs.clear()
    squeeze_5.inputs.clear()
    squeeze_6.inputs.clear()
    squeeze_7.inputs.clear()

    reshape_node.outputs.clear()
    return posemb

src_onnx_path = args.src
dst_onnx_path = args.dst

print("Load onnx model from {}".format(src_onnx_path))
graph = gs.import_onnx(onnx.load(src_onnx_path))
print("Nodes:{}".format(len(graph.nodes)))
graph.fold_constants().cleanup()

if DYNAMIC:
    for i in range(4):
        graph.inputs[i].shape=[-1,-1,1]
        
nodes = graph.nodes
nodes_dict = {}
for node in nodes:
    name = node.name
    nodes_dict.update({name: node})

if ENABLE_LAYERNORM_PLUGIN:
    print("Fuse ops into LayerNorm")
    count = 0
    for op_name in nodes_dict:
        if 'ReduceMean' not in op_name:
            continue
        layernorm = replace_with_layernorm(nodes_dict, nodes_dict[op_name])
        if layernorm:
            count += 1
            nodes.append(layernorm)
    print("Detected {} LayerNorms".format(count))
    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_ln.onnx")

if ENABLE_ADDLAYERNORM_PLUGIN:
    print("Fuse ops into AddLayerNorm")
    count = 0
    for op_name in nodes_dict:
        if 'ReduceMean' not in op_name:
            continue
        layernorm = replace_with_addlayernorm(nodes_dict, nodes_dict[op_name])
        if layernorm:
            count += 1
            nodes.append(layernorm)
    print("Detected {} AddLayerNorm".format(count))
    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_aln.onnx")

if ENABLE_SLICERESHAPE_PLUGIN:
    print("Fuse ops into slicereshape")
    count = 0
    for op_name in nodes_dict:
        if 'Shape' not in op_name:
            continue
        slicereshape = replace_with_slice_reshape(nodes_dict, nodes_dict[op_name])
        if slicereshape:
            count += 1
            nodes.append(slicereshape)
    print("Detected {} slicereshape".format(count))
    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_slreshape.onnx")

if ENABLE_FUSING_ADDRELU:
    print("Fuse ops into AddRelu")
    count = 0
    for op_name in nodes_dict:
        if 'Relu' not in op_name:
            continue
        add_relu = fuse_add_relu(nodes_dict, nodes_dict[op_name])
        if add_relu:
            count += 1
            nodes.append(add_relu)
    print("Detected {} AddRelu".format(count))
    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_addrelu.onnx")

if ENABLE_POSTEMBEDDING_PLUGIN:
    print("Fuse ops into PostEmbedding")
    posemb = replace_with_postembedding(nodes_dict, graph.inputs[4:])
    if posemb:
        graph.inputs=graph.inputs[:5]
        graph.inputs[4].shape=(-1,8)
        graph.inputs[4].dtype=np.float32
        graph.inputs[4].name="read_file_0.tmp_6-13"
        nodes.append(posemb)

    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_postemb.onnx")
        
if DEBUG:
    graph.cleanup().toposort()
    dst_onnx_path = './model/debug.onnx'
    # graph.toposort()
else:
    graph.cleanup().toposort()

print("Nodes:{}".format(len(graph.nodes)))
if SIM:
    onnx_model, check = onnxsim.simplify(gs.export_onnx(graph))
else:
    onnx_model=gs.export_onnx(graph)
onnx.save(onnx_model, dst_onnx_path)
# onnx.save(onnx.shape_inference.infer_shapes(onnx_model), dst_onnx_path)
print("Save modified onnx model to {}".format(dst_onnx_path))
