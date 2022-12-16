#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp3xx_func.py 
@time: 2022/12/13
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import json
from utils import logger


def nnp3xx_get_device_type(work_dir, node_name=None):
    import deepeye
    return deepeye.util.get_device_type(work_dir, node_name)


def nnp3xx_count_mac(relay_quant):
    from deepeye.util import count_mac
    return count_mac(relay_quant)


def nnp3xx_eval_relay(relay_func, params, in_datas):
    import deepeye
    return deepeye.eval_relay(relay_func, params, in_datas)


def nnp3xx_load_from_json(filepath):
    try:
        with open(filepath, "rb") as f:
            import tvm
            from tvm import relay
            relay_func = tvm.load_json(json.load(f))
            relay_func = relay.ir_pass.infer_type(relay_func)
            params = {}
            return relay_func, params
    except Exception as e:
        logger.error("Failed to load model from json -> {}".format(e))
        exit(-1)


def nnp3xx_build_lib(relay_func, params):
    """编译cpu模型"""
    try:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_runtime
        from deepeye.relay_pass import rewrite_for_cpu
        relay_func = relay.relay_pass.bind_params(relay_func, params)
        relay_func.ret_type = None
        relay_func, cpu_params = rewrite_for_cpu(relay_func, sim_target="nnp300")
        ctx = tvm.cpu(0)
        with relay.build_config(opt_level=3):
            graph, lib, cpu_params = relay.build(relay_func, "llvm", params=cpu_params)
        module = graph_runtime.create(graph, lib, ctx)
        module.set_input(**cpu_params)
        return module
    except Exception as e:
        logger.error("Failed to load model -> {}".format(e))
        exit(-1)
