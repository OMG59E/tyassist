#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : parser.py
@Time    : 2022/7/1 下午4:37
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import yaml


def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
