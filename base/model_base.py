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
                 resize_type=0, padding_value=128, padding_mode=PaddingMode.LEFT_TOP, dataset=None, test_num=0):
        pass

    @abc.abstractmethod
    def load(self, model_dir):
        """
        加载模型
        :param model_dir: 模型目录
        :return:
        """
        pass

    @abc.abstractmethod
    def _preprocess(self, cv_image):
        """
        内部预处理调用
        :param cv_image: opencv image
        :return:
        """
        pass

    @abc.abstractmethod
    def _postprocess(self, outputs, cv_image=None):
        """
        内部后处理调用
        :param outputs: 模型推理输出
        :param cv_image: 原图像
        :return:
        """
        pass

    @abc.abstractmethod
    def inference(self, cv_image):
        """
        推理接口，目前仅支持batch1
        :param cv_image: opencv image
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def ave_latency_ms(self):
        """模型芯片内部推理时间, 不是严格准确，仅供参考"""
        pass

    @property
    @abc.abstractmethod
    def end2end_latency_ms(self):
        """python推理时间, 包括数据传入传出、预处理、后处理时间"""
        pass

    @abc.abstractmethod
    def evaluate(self):
        """模型指标评估"""
        pass

    @abc.abstractmethod
    def demo(self, img_path):
        """
        模型demo
        :param img_path: 图片路径
        :return:
        """
        pass