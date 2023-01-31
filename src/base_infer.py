#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: base_infer.py 
@time: 2022/12/21
@Author  : xingwg
@software: PyCharm 
"""
import abc


class BaseInfer(object, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.time_span = 0
        self.total = 0
        self.engine = None
        self.target = "nnp300"  # nnp300 310 320 3020 400
        self.backend = "chip"  # chip/sdk_iss/tvm

    def load_json(self, model_path):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, model_path):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, in_datas: dict, to_file=False):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    @property
    def ave_latency_ms(self):
        if self.total == 0:
            return 0
        return self.time_span / self.total
