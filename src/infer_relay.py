#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: infer_relay.py 
@time: 2022/08/16
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import os
import time
import tvm

from utils import logger
from nnp4xx_func import (
    nnp4xx_load_from_json,
    nnp4xx_build_lib
)
from nnp3xx_func import (
    nnp3xx_load_from_json,
    nnp3xx_build_lib,
)


class InferRelay(object):
    def __init__(self, input_names: list):
        self._relay = None
        self._params = None
        self._input_names = input_names
        self._engine = None
        self._ave_latency_ms = 0
        self._total = 0
        self._target = "nnp300"

    def load_from_mem(self, callback=None, target="nnp300"):
        """从内存加载浮点模型
        """
        self._target = target
        self._relay, self._params = callback()
        self._build_model(target)

    def load_from_json(self, filepath, target="nnp300"):
        """从json文件加载模型
        """
        self._target = target
        if not os.path.exists(filepath):
            logger.error("Not found model file -> {}".format(filepath))
            exit(-1)
        if target.startswith("nnp3"):
            self._relay, self._params = nnp3xx_load_from_json(filepath)
        elif target.startswith("nnp4"):
            self._relay, self._params = nnp4xx_load_from_json(filepath)

        self._build_model(target)

    def _build_model(self, target="nnp300"):
        if target.startswith("nnp3"):
            self._engine = nnp3xx_build_lib(self._relay, self._params)
        elif target.startswith("nnp4"):
            self._engine = nnp4xx_build_lib(self._relay, self._params)

    @property
    def ave_latency_ms(self):
        if self._total == 0:
            return 0
        return self._ave_latency_ms / self._total

    def run(self, in_datas: list, to_file=False):
        """推理
        :param in_datas: list表示多输入
        :param to_file: 表示是否将结果输出至文件
        :return:
        """
        if len(in_datas) != len(self._input_names):
            pass
        _in_datas = dict()
        for idx in range(len(in_datas)):
            _in_datas[self._input_names[idx]] = in_datas[idx] if self._target.startswith("npp3") else tvm.nd.array(in_datas[idx])
        self._total += 1
        t_start = time.time()
        self._engine.set_input(**_in_datas)
        self._engine.run()
        outputs = list()
        for idx in range(self._engine.get_num_outputs()):
            outputs.append(self._engine.get_output(idx).asnumpy())
        self._ave_latency_ms += (time.time() - t_start) * 1000
        return outputs
