#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : postprocess.py
@Time    : 2022/7/14 下午1:58
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import numpy as np


def softmax(x, axis=1):
    """
    :param x:
    :param axis:
    :return:
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True)) 
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
