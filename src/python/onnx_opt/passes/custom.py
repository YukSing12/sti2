#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/21 18:17
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用

import onnx_graphsurgeon as gs
import numpy as np
from .base import ReplacePass, TowOpPass


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
