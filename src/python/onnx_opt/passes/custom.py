#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/21 18:17
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用

import onnx_graphsurgeon as gs
import numpy as np
from .base import Pass, CustomPass, TowOpPass


class MaskedSoftmaxPass(Pass):
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


class LayernormPass(Pass):
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


class PostEmbeddingPass(CustomPass):
    def replace(self, graph: gs.Graph):
        graph_inputs = graph.inputs[4:]
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

        reshape_node = (
            graph_inputs[0]
            .outputs[0]
            .outputs[0]
            .outputs[0]
            .outputs[0]
            .outputs[0]
            .outputs[0]
            .outputs[0]
            .outputs[0]
            .outputs[0]
        )
        output = reshape_node.outputs[0]
        graph_inputs[0].shape=(-1,8)
        graph_inputs[0].dtype=np.int32
        posemb = gs.Node(
            op="PostEmbedding",
            name="PostEmbedding",
            inputs=[
                graph_inputs[0],
                emb_0,
                emb_1,
                emb_2,
                emb_3,
                emb_4,
                emb_5,
                emb_6,
                emb_7,
            ],
            outputs=[output],
        )

        squeeze_0.inputs.clear()
        squeeze_1.inputs.clear()
        squeeze_2.inputs.clear()
        squeeze_3.inputs.clear()
        squeeze_4.inputs.clear()
        squeeze_5.inputs.clear()
        squeeze_6.inputs.clear()
        squeeze_7.inputs.clear()

        reshape_node.outputs.clear()
        return [posemb]
