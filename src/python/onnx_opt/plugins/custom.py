#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/29 14:19
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import numpy as np
import onnx_graphsurgeon as gs
from .base import PluginBase


class PostEmbeddingPlugin(PluginBase):
    def __call__(self, graph: gs.Graph, passes):
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
        graph_inputs[0].shape = (-1, 8)
        graph_inputs[0].dtype = np.int32
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
        graph.nodes.extend([posemb])
