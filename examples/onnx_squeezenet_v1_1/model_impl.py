#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : model_impl.py
@Time    : 2022/7/21 下午2:02
@Author  : xingwg
@Software: PyCharm
"""
from utils import logger
from utils.postprocess import softmax
from base.classification import Classifier


class SqueezeNetV1_1(Classifier):

    def _postprocess(self, outputs, cv_image=None):
        if len(outputs) != 1:
            logger.error("only support signal output, please check")
            exit(-1)
        outputs = outputs[0]  # [bs, num_cls] or [bs, num_cls, 1, 1]
        bs = outputs.shape[0]
        if bs != 1:
            logger.error("only support bs=1, please check")
            exit(-1)
        outputs = softmax(outputs)
        return outputs
