#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/21 18:17
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用

import onnx_graphsurgeon as gs
import numpy as np
from .base import ReplacePass, TowOpPass, RemovePass
from typing import Optional

class MaskedSoftmaxPass(ReplacePass):
    def replace(self, node, count):
        if node.op == "Softmax":
            node2 = node.inputs[0].inputs[0]
            if node2.op == "Add":
                masked_softmax_node = gs.Node(
                    op="MaskedSoftmax",
                    name=f"plugin.MaskedSoftmax.{count}",
                    inputs=[
                        node2.inputs[1]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0]
                        .inputs[0],
                        node2.inputs[0],
                    ],
                    outputs=[node.outputs[0]],
                )
                node2.inputs.clear()
                node.inputs.clear()
                node.outputs.clear()

                # clear(node)
                return masked_softmax_node


_deprecated_nodes_dict: dict = {}


class LayernormPass(ReplacePass):
    def replace(self, node, count):
        if node.op == "ReduceMean":
            mean_node = node
            node_id = int(node.name.split(".")[-1])
            if not (
                ("p2o.Sub.{}".format(node_id // 2) in _deprecated_nodes_dict)
                and ("p2o.Pow.{}".format(node_id // 2) in _deprecated_nodes_dict)
                and ("p2o.Div.{}".format(node_id // 2) in _deprecated_nodes_dict)
                and ("p2o.Sqrt.{}".format(node_id // 2) in _deprecated_nodes_dict)
            ):
                return None

            sub_node = _deprecated_nodes_dict["p2o.Sub.{}".format(node_id // 2)]
            div_node = _deprecated_nodes_dict["p2o.Div.{}".format(node_id // 2)]
            mul_node = div_node.outputs[0].outputs[0]
            add_node = mul_node.outputs[0].outputs[0]

            gamma = mul_node.inputs[1]
            beta = add_node.inputs[1]
            if len(add_node.outputs) == 0:
                return None
            name = "LayerNorm.{}".format(node_id)
            layernorm = gs.Node(
                op="LayerNorm",
                name=name,
                inputs=[mean_node.inputs[0], gamma, beta],
                outputs=[add_node.outputs[0]],
                attrs={"epsilon": 1e-5},
            )
            mean_node.inputs.clear()
            sub_node.inputs.clear()
            add_node.outputs.clear()
            return layernorm


class AddOpPass(TowOpPass):
    def __init__(self):
        super().__init__((["Add"], ["LayerNorm", "Relu"]))


class SliceReshapePass(TowOpPass):
    def __init__(self):
        super().__init__((["Slice"], ["Reshape"]))


class EmbLayerNormPass(ReplacePass):
    def replace(self, node, count):
        if node.op == "ReduceMean":
            node_id = int(node.name.split(".")[-1])
            if not (
                ("p2o.Sub.{}".format(node_id // 2) in _deprecated_nodes_dict)
                and ("p2o.Pow.{}".format(node_id // 2) in _deprecated_nodes_dict)
                and ("p2o.Div.{}".format(node_id // 2) in _deprecated_nodes_dict)
                and ("p2o.Sqrt.{}".format(node_id // 2) in _deprecated_nodes_dict)
            ):
                return None

            add1_node = node.inputs[0].inputs[0]

            if add1_node.inputs[1].inputs[0].op != "Gather":
                return None

            gather2_node = add1_node.inputs[1].inputs[0]
            add2_node = add1_node.inputs[0].inputs[0]
            gather0_node = add2_node.inputs[0].inputs[0]
            gather1_node = add2_node.inputs[1].inputs[0]
            squeeze2_node = gather2_node.inputs[1].inputs[0]
            squeeze0_node = gather0_node.inputs[1].inputs[0]
            squeeze1_node = gather1_node.inputs[1].inputs[0]

            div_node = _deprecated_nodes_dict["p2o.Div.{}".format(node_id // 2)]
            mul_node = div_node.outputs[0].outputs[0]
            add3_node = mul_node.outputs[0].outputs[0]

            gamma = mul_node.inputs[1]
            beta = add3_node.inputs[1]

            name = "EmbLayerNorm.{}".format(node_id)
            emblayernorm = gs.Node(
                op="EmbLayerNorm",
                name=name,
                inputs=[
                    gather2_node.inputs[0],
                    gather0_node.inputs[0],
                    gather1_node.inputs[0],
                    squeeze2_node.inputs[0],
                    squeeze0_node.inputs[0],
                    squeeze1_node.inputs[0],
                    gamma,
                    beta,
                ],
                outputs=[add3_node.outputs[0]],
                attrs={"epsilon": 1e-5},
            )
            gather0_node.inputs.clear()
            gather1_node.inputs.clear()
            gather2_node.inputs.clear()
            squeeze0_node.inputs.clear()
            squeeze1_node.inputs.clear()
            squeeze2_node.inputs.clear()
            add3_node.outputs.clear()
            return emblayernorm


class PreEmbeddingPass(ReplacePass):
    def replace(self, node, count):
        if node.name == "p2o.Add.1":
            node_id = int(node.name.split(".")[-1])
            # if not (('p2o.Sub.{}'.format(node_id//2) in _deprecated_nodes_dict)
            # and ('p2o.Pow.{}'.format(node_id//2) in _deprecated_nodes_dict)
            # and ('p2o.Div.{}'.format(node_id//2) in _deprecated_nodes_dict)
            # and ('p2o.Sqrt.{}'.format(node_id//2) in _deprecated_nodes_dict)):
            #     return None

            add1_node = node

            # if not ((add1_node.inputs[1].inputs[0].op == "Gather") and (add1_node.inputs[0].inputs[0].op == "Add")):
            #     return None

            gather2_node = add1_node.inputs[1].inputs[0]
            add2_node = add1_node.inputs[0].inputs[0]
            gather0_node = add2_node.inputs[0].inputs[0]
            gather1_node = add2_node.inputs[1].inputs[0]
            squeeze2_node = gather2_node.inputs[1].inputs[0]
            squeeze0_node = gather0_node.inputs[1].inputs[0]
            squeeze1_node = gather1_node.inputs[1].inputs[0]

            # div_node = _deprecated_nodes_dict['p2o.Div.{}'.format(node_id//2)]
            # mul_node = div_node.outputs[0].outputs[0]
            # add3_node = mul_node.outputs[0].outputs[0]

            # gamma = mul_node.inputs[1]
            # beta = add3_node.inputs[1]

            name = "PreEmbedding.{}".format(node_id)  # gamma,beta
            PreEmbedding = gs.Node(
                op="PreEmbedding",
                name=name,
                inputs=[
                    gather2_node.inputs[0],
                    gather0_node.inputs[0],
                    gather1_node.inputs[0],
                    squeeze2_node.inputs[0],
                    squeeze0_node.inputs[0],
                    squeeze1_node.inputs[0],
                ],
                outputs=[add1_node.outputs[0]],
                attrs={"epsilon": 1e-5},
            )
            gather0_node.inputs.clear()
            gather1_node.inputs.clear()
            gather2_node.inputs.clear()
            squeeze0_node.inputs.clear()
            squeeze1_node.inputs.clear()
            squeeze2_node.inputs.clear()
            add1_node.outputs.clear()
            return PreEmbedding        
        

class FFNReluPass(RemovePass):
    def remove(self, node):
        if node.op == "Relu":
            node_id = int(node.name.split(".")[-1])
            if len(node.outputs[0].outputs) == 2:
                if node.outputs[0].outputs[0].op == "Flatten" and node.outputs[0].outputs[1].op == "Shape":
                    flatten1_node = node.outputs[0].outputs[0]
                    shape1_node = node.outputs[0].outputs[1]
                    matmul1_node = flatten1_node.outputs[0].outputs[0]
                    reshape1_node = matmul1_node.outputs[0].outputs[0]
                    add1_node = reshape1_node.outputs[0].outputs[0]
                    add2_node = add1_node.outputs[0].outputs[0]

                    # clear flatten after relu
                    matmul1_node.inputs[0] = node.outputs[0]
                    flatten1_node.inputs.clear()
                    flatten1_node.outputs.clear()

                    # clear reshape after relu
                    add1_node.inputs[0] = matmul1_node.outputs[0]
                    shape1_node.inputs.clear()


                    reshape1_node.inputs.clear()
                    reshape1_node.outputs.clear()

                    # reconnect reshape
                    add0_node = node.inputs[0].inputs[0]
                    reshape0_node = add0_node.inputs[0].inputs[0]
                    concat0_node = reshape0_node.inputs[1].inputs[0]
                    concat0_node.inputs[1].values = np.array([768],dtype=np.int64)
                    matmul0_node = reshape0_node.inputs[0].inputs[0]
                    add0_node.inputs[0] = matmul0_node.outputs[0]
                    reshape0_node.inputs[0] = add1_node.outputs[0]
                    add2_node.inputs[0] = reshape0_node.outputs[0]

            return 1
        return 0


class FTErnie(ReplacePass):
    def __init__(self, pattern: Optional[str] = None):
        super().__init__(pattern)
        self.replaced = False

    def replace(self, node, count):
        # Only replace once
        if not self.replaced:
            self.replaced = True
            # four input
            squeeze_node_0 = _deprecated_nodes_dict['p2o.Squeeze.0']
            squeeze_node_1 = _deprecated_nodes_dict['p2o.Squeeze.1']
            squeeze_node_2 = _deprecated_nodes_dict['p2o.Squeeze.2']
            matmul_node_0  = _deprecated_nodes_dict['p2o.MatMul.0']
            word_ids       = squeeze_node_0.inputs[0]
            pos_ids        = squeeze_node_1.inputs[0]
            sent_ids       = squeeze_node_2.inputs[0]
            mask           = matmul_node_0.inputs[0]
            # one ouput
            add_node_0     = _deprecated_nodes_dict['p2o.Add.210']
            ernie_out      = add_node_0.outputs[0]
            ernie = gs.Node(
                op="ErniePlugin",
                name="plugin.ErniePlugin.0",
                inputs=[word_ids, pos_ids, sent_ids, mask],
                outputs=[ernie_out],
                attrs={},
            )

            # clear nodes
            squeeze_node_0.inputs.clear()
            squeeze_node_1.inputs.clear()
            squeeze_node_2.inputs.clear()
            matmul_node_0.inputs.clear()
            add_node_0.outputs.clear()

            return ernie
        return None
