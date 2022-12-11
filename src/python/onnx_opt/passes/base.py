#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/23 11:53
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
import re
import onnx_graphsurgeon as gs
from typing import Optional, List, Tuple


class PassBase:
    def __init__(self, pattern: Optional[str] = None):
        # pattern = r"\S+\.(\S+)\.(\d+)"
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern

    def __call__(self, nodes: List[gs.Node]) -> int:
        raise NotImplementedError


class ReplacePass(PassBase):
    def __call__(self, nodes: List[gs.Node]):
        count = 0
        for node in nodes:
            new_node = self.replace(node, count)
            if new_node is not None:
                nodes.append(new_node)
                count += 1
        return count

    def replace(self, node: gs.Node, count: int) -> Optional[gs.Node]:
        raise NotImplementedError


class RemovePass(PassBase):
    def __call__(self, nodes: List[gs.Node]):
        count = 0
        for node in nodes:
            count += self.remove(node)
        return count

    def remove(self, node: gs.Node) -> int:
        raise NotImplementedError


class TowOpPass(ReplacePass):
    def __init__(
        self, nodes_name: Tuple[List[str], List[str]], pattern: Optional[str] = None
    ):
        self.nodes_name = nodes_name
        super().__init__(pattern)

    def replace(self, node: gs.Node, count: int) -> Optional[gs.Node]:
        node_name = node.op
        if node_name in self.nodes_name[0]:
            for next_node in node.outputs:
                next_node_name = next_node.outputs[0].op
                if next_node_name in self.nodes_name[1]:
                    inputs = node.inputs
                    for v in next_node.inputs:
                        if not (v is node):
                            inputs.append(v)
                    output_node = gs.Node(
                        op=f"{node_name}{next_node_name}",
                        name=f"plugin.{node_name}{next_node_name}.{count}",
                        inputs=inputs,
                        outputs=next_node.outputs,
                    )
                    node.inputs.clear()
                    node.outputs.clear()
                    next_node.inputs.clear()
                    next_node.outputs.clear()
                    return output_node
