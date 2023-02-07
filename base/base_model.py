#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : model_base.py
@Time    : 2022/7/18 上午10:32
@Author  : xingwg
@Software: PyCharm
"""
import abc
from utils import logger
from src.nnp3xx_infer import Nnp3xxSdkInfer, Nnp3xxTvmInfer
from src.nnp4xx_infer import Nnp4xxSdkInfer, Nnp4xxTvmInfer


class BaseModel(object, metaclass=abc.ABCMeta):
    """模型描述基类，提供2个功能，demo和eval
    """
    def __init__(self, **kwargs):
        """"""
        self.inputs = kwargs["inputs"]   # multi-input
        self.dataset = kwargs["dataset"]
        self.test_num = kwargs["test_num"]
        self.enable_aipp = kwargs["enable_aipp"]
        self.infer = None
        self.target = kwargs["target"]
        self.dtype = kwargs["dtype"]
        self.backend = kwargs["backend"]

        self.use_norm = True if self.dtype == "fp32" else False

        self.total = 0
        self.time_span = 0

        if self.target.startswith("nnp3"):
            if self.backend == "tvm":
                self.infer = Nnp3xxTvmInfer()
                self.infer.set_input_names([_input["name"] for _input in self.inputs])
                self.enable_aipp = False
            else:
                self.infer = Nnp3xxSdkInfer(enable_dump=0, enable_aipp=self.enable_aipp)
                self.infer.set_input_enable_aipps([_input["support"] for _input in self.inputs])
                self.infer.set_input_pixel_format([_input["pixel_format"] for _input in self.inputs])
        elif self.target.startswith("nnp4"):
            self.enable_aipp = False
            if self.backend == "tvm":
                self.infer = Nnp4xxTvmInfer()
                self.infer.set_input_names([_input["name"] for _input in self.inputs])
            else:
                self.infer = Nnp4xxSdkInfer(enable_dump=0, enable_aipp=self.enable_aipp)
        else:
            logger.error("Not support target -> {}".format(self.target))
            exit(-1)

        self.infer.backend = self.backend

    def load(self, model_path):
        """加载so模型
        :param model_path: 模型目录
        :return:
        """
        self.infer.load(model_path)

    def load_json(self, model_path):
        """加载json模型
        :param model_path: 模型目录
        :return:
        """
        self.infer.load_json(model_path)

    @abc.abstractmethod
    def _preprocess(self, cv_image):
        """内部预处理调用
        :param cv_image: opencv image
        :return:
        """
        pass

    @abc.abstractmethod
    def _postprocess(self, outputs, cv_image=None):
        """内部后处理调用
        :param outputs: 模型推理输出
        :param cv_image: 原图像
        :return:
        """
        pass

    @abc.abstractmethod
    def inference(self, cv_image):
        """推理接口，目前仅支持batch1
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
