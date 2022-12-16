#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_func.py 
@time: 2022/12/13
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import json
from utils import logger


def nnp4xx_estimate_flops(relay_quant):
    from tvm.contrib.edgex import estimate_FLOPs
    return estimate_FLOPs(relay_quant)


def nnp4xx_estimate_cycles(relay_quant):
    from tvm.contrib.edgex import estimate_cycles
    return estimate_cycles(relay_quant)


def nnp4xx_load_from_json(filepath):
    try:
        import tvm
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


def nnp4xx_build_lib(relay, params, save_path=None):
    """编译cpu模型"""
    try:
        import tvm
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


def nnp4xx_inference(module, in_datas):
    import tvm
    for input_name in in_datas:
        module.set_input(input_name, tvm.nd.array(in_datas[input_name]))
    module.run()
    outputs = list()
    for idx in range(module.get_num_outputs()):
        outputs.append(module.get_output(idx).numpy())
    return outputs


def nnp4xx_iss_fixed(lib, in_datas):
    """nnp4xx量化后模型定点仿真"""
    import tvm
    from tvm.contrib import graph_executor
    logger.info("Executing model on edgex...")
    edgex_module = graph_executor.GraphModule(lib["default"](tvm.edgex(), tvm.cpu()))
    return nnp4xx_inference(edgex_module, in_datas)
