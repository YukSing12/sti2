#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/11/21 10:49
# @Author : Rongrui Zhan
# @desc : 本代码未经授权禁止商用
from enum import Enum
from typing import List
import yaml
import rich
import importlib


print(__file__)
with open(f"{__file__}/plugins.yaml", encoding="UTF-8") as file:
    plugins = yaml.load(file, Loader=yaml.FullLoader)
Plugin = Enum("Plugin", {p["name"]: p["short_name"] for p in plugins})

short2pass = {plugin.short_name: plugin.name for plugin in plugins}


# print(Plugin)
# print(Plugin.__dict__)
# # print(Plugin.AddRelu)
# a = Plugin.AddRelu
# print(a)
