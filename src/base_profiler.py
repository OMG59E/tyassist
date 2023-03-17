#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: base_profiler.py 
@time: 2022/12/22
@Author  : xingwg
@software: PyCharm 
"""
import abc


class BaseSdkProfiler(object, metaclass=abc.ABCMeta):
    def __init__(self,
                 sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
                 target="nnp300"):

        self.sdk_cfg_file = sdk_cfg_file
        self.engine = None
        self.sdk = None
        self.result_dir = ""
        self.target = target

    @property
    def targets(self):
        return {"nnp200": 750, "nnp300": 792, "nnp310": 792, "nnp3020": 792, "nnp320": 768, "nnp400": 500}

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
