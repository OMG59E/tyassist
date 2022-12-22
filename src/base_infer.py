#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: base_infer.py 
@time: 2022/12/21
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import os
import abc
from utils import logger


class BaseSdkInfer(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            net_cfg_file="/DEngine/tyhcp/net.cfg",
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            enable_dump=0,
            enable_aipp=False,
    ):
        """
        :param net_cfg_file:
        :param sdk_cfg_file:
        :param enable_dump:
        """
        self.net_cfg_file = net_cfg_file
        self.sdk_cfg_file = sdk_cfg_file
        self.enable_dump = enable_dump
        self.enable_aipp = enable_aipp
        self.time_span = 0
        self.total = 0

        self.engine = None
        self.sdk = None
        self.dump_root_path = ""
        self.result_dir = ""
        self.prefix = "chip"

        if not os.path.exists(sdk_cfg_file):
            logger.error("Not found sdk_cfg_file -> {}".format(sdk_cfg_file))
            exit(-1)

    @abc.abstractmethod
    def load(self, model_path):
        raise NotImplementedError

    @property
    def ave_latency_ms(self):
        if self.total == 0:
            return 0
        return self.time_span / self.total

    @abc.abstractmethod
    def run(self, in_datas: list, to_file=False):
        """
        @param in_datas: [{"data": numpy.ndarray, "pixel_format": str, "enable_aipp": bool}, ...]
        @param to_file:
        @return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unload(self):
        pass

    @abc.abstractmethod
    def compare_layer_out(self):
        raise NotImplementedError

