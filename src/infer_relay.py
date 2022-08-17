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
import json
import time

from utils import logger


class InferRelay(object):
    def __init__(self, input_names: list):
        self._relay = None
        self._params = None
        self._input_names = input_names
        self._engine = None
        self._ave_latency_ms = 0
        self._total = 0

    def load(self, callback):
        """加载浮点模型
        :param callback:
        :return:
        """
        import tvm
        from tvm import relay
        from tvm.contrib import graph_runtime
        from deepeye.relay_pass import rewrite_for_cpu

        self._relay, self._params = callback()
        relay_func = relay.relay_pass.bind_params(self._relay, self._params)
        relay_func.ret_type = None
        relay_func, cpu_params = rewrite_for_cpu(relay_func, sim_target="nnp300")
        ctx = tvm.cpu(0)
        with relay.build_config(opt_level=3):
            graph, lib, cpu_params = relay.build(relay_func, "llvm", params=cpu_params)
        self._engine = graph_runtime.create(graph, lib, ctx)
        self._engine.set_input(**cpu_params)

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
            _in_datas[self._input_names[idx]] = in_datas[idx]
        self._total += 1
        t_start = time.time()
        self._engine.set_input(**_in_datas)
        self._engine.run()
        outputs = list()
        for idx in range(self._engine.get_num_outputs()):
            outputs.append(self._engine.get_output(idx).asnumpy())
        self._ave_latency_ms += (time.time() - t_start) * 1000
        return outputs
