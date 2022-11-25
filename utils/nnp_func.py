#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp_func.py 
@time: 2022/10/25
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import tvm
import json
from . import logger


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
            from tvm import relay
            relay_func = tvm.load_json(json.load(f))
            relay_func = relay.ir_pass.infer_type(relay_func)
            params = {}
            return relay_func, params
    except Exception as e:
        logger.error("Failed to load model from json -> {}".format(e))
        exit(-1)


def nnp4xx_estimate_flops(relay_quant):
    from tvm.contrib.edgex import estimate_FLOPs
    return estimate_FLOPs(relay_quant)


def nnp4xx_estimate_cycles(relay_quant):
    from tvm.contrib.edgex import estimate_cycles
    return estimate_cycles(relay_quant)


def nnp4xx_load_from_json(filepath):
    try:
        with open(filepath) as file_obj:
            json_str = json.load(file_obj)
            if "ir" not in tvm.__dict__:
                logger.error("Not found ir in tvm")
                exit(-1)
            relay_func = tvm.ir.load_json(json_str)
            return relay_func, {}
    except Exception as e:
        logger.error("Failed to load model from json -> {}".format(e))
        exit(-1)


def nnp3xx_build_lib(relay_func, params):
    try:
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


def nnp4xx_build_lib(relay, params, save_path=None):
    try:
        from tvm.contrib import graph_executor
        from tvm.relay import build
        cpu_target = tvm.target.Target("llvm")
        with tvm.transform.PassContext(opt_level=3):
            cpu_lib = build(relay, target=cpu_target, params=params)
            if save_path:
                cpu_lib.export_library(save_path)
        module = graph_executor.GraphModule(cpu_lib["default"](tvm.cpu()))
        return module
    except Exception as e:
        logger.error("Failed to load model -> {}".format(e))
        exit(-1)


def nnp4xx_iss_fixed(lib, in_datas):
    from tvm.contrib import graph_executor
    logger.info("Executing model on edgex...")
    edgex_module = graph_executor.GraphModule(lib["default"](tvm.edgex(), tvm.cpu()))
    return nnp4xx_inference(edgex_module, in_datas)


def nnp4xx_inference(module, in_datas):
    for input_name in in_datas:
        module.set_input(input_name, tvm.nd.array(in_datas[input_name]))
    module.run()
    outputs = list()
    for idx in range(len(in_datas)):
        outputs.append(module.get_output(idx).numpy())
    return outputs
