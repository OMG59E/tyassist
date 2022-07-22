#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : onnx_squeezenet_v1_1.py
@Time    : 2022/7/19 下午3:36
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
from utils.postprocess import softmax
from base.classification import Classifier


class SqueezeNetV1_1(Classifier):
    def _postprocess(self, outputs):
        return softmax(outputs, axis=1)
