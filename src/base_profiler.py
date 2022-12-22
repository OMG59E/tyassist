#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: base_profiler.py 
@time: 2022/12/22
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import abc


class BaseSdkProfiler(object, metaclass=abc.ABCMeta):
    def __init__(self,
                 net_cfg_file="/DEngine/tyhcp/net.cfg",
                 sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
                 target="nnp300"):
        self.ip = "127.0.0.1"
        self.port = 9090
        self.net_cfg_file = net_cfg_file
        self.sdk_cfg_file = sdk_cfg_file
        self.engine = None
        self.sdk = None
        self.result_dir = ""
        self.target = target

    @property
    def targets(self):
        return {"nnp320": 768, "nnp300": 792, "nnp200": 750}

    @abc.abstractmethod
    def load(self, model_path):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, in_datas: list, to_file=False):
        """
        @param in_datas:
        @param to_file:
        @return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unload(self):
        pass

    @abc.abstractmethod
    def save_profile(self):
        pass
