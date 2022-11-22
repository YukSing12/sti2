#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/21 18:17
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import re
import onnx_graphsurgeon as gs
from typing import Optional, List


class Pass:
    def __init__(self, pattern: Optional[str] = None):
        # pattern = r"\S+\.(\S+)\.(\d+)"
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern

    def __call__(self, nodes):
        self.replace(nodes)

    def replace(self, nodes: List[gs.Node]):
        raise NotImplementedError


class MaskedSoftmaxFuser(Pass):
    def replace(self, nodes):
        count = 0
        for node in nodes:
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

                    nodes.append(masked_softmax_node)
                    # clear(node)
                    count += 1
        print(count)
        return count
