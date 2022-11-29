#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/29 14:13
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
from typing import TYPE_CHECKING, List
from enum import Enum

if TYPE_CHECKING:
    import onnx_graphsurgeon as gs
    from ..passes import PassBase


class PluginType(Enum):
    EnterPlugin = "EnterPlugin"
    ExitPlugin = "ExitPlugin"


class PluginBase:
    def __init__(self, plugin_type: PluginType = PluginType.EnterPlugin):
        self.type = plugin_type

    def __call__(self, graph: "gs.Graph", passes: List["PassBase"]):
        pass
