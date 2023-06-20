#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : model_impl.py
@Time    : 2022/7/21 下午2:02
@Author  : xingwg
@Software: PyCharm
"""
from base.classification import Classifier
from utils import logger
from utils.postprocess import softmax


class ResNet50(Classifier):

    def _postprocess(self, outputs, cv_images=None):
        if len(outputs) != 1:
            logger.error("only support signal output, please check")
            exit(-1)
        bs = len(cv_images)
        outputs = outputs[0]
        outputs = outputs[:bs, ...]
        outputs = softmax(outputs)
        return outputs
