#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : model_base.py
@Time    : 2022/7/18 上午10:32
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import abc
from utils.enum_type import PaddingMode


class ModelBase(object, metaclass=abc.ABCMeta):
    """模型描述基类，提供2个功能，demo和eval
    """
    @abc.abstractmethod
    def __init__(self, input_size: tuple, mean: tuple, std: tuple, use_rgb=False, use_norm=False,
                 resize_type=0, padding_value=114, padding_mode=PaddingMode.LEFT_TOP, dataset=None, test_num=0):
        pass

    @abc.abstractmethod
    def load(self, model_dir):
        pass

    @abc.abstractmethod
    def _preprocess(self, cv_image):
        pass

    def preprocess(self, img_path):
        pass

    @abc.abstractmethod
    def _postprocess(self, outputs):
        pass

    def postprocess(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def demo(self, img_path):
        pass