#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : check.py
@Time    : 2022/7/13 上午10:02
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import os
from utils import logger


def check_config(cfg):
    """配置文件参数合法性检查
    :param cfg:
    :return:
    """
    # 检查是否缺少关键字
    if "model" not in cfg:
        logger.error("The key(model) must be in cfg")
        return False

    if "quantization" not in cfg:
        logger.error("The key(quantize) must be in cfg")
        return False

    if "weight" not in cfg["model"]:
        logger.error("The key(weight) must be in cfg")
        return False

    if not os.path.exists(cfg["model"]["weight"]):
        logger.error("The model weight not exist -> {}".format(cfg["model"]["weight"]))
        return False

    # 多输入必须定义预处理
    if len(cfg["model"]["inputs"]) > 1 and not cfg["quantization"]["custom_preprocess"]:
        logger.error("Multi-input must be setting custom_preprocess")
        return False

    for _input in cfg["model"]["inputs"]:
        if _input["layout"] not in ["NCHW", "NHWC"]:
            logger.error("layout must be in [NCHW, NHWC]")
            return False

        if _input["pixel_format"] not in ["None", "RGB", "BGR", "GRAY"]:
            logger.error("pixel_format must be in [None, RGB, BGR, GRAY]")
            return False

        if _input["padding_mode"] not in [0, 1]:
            logger.error("padding_mode must be in [0, 1]")
            return False

    # TODO
    # 检查是否缺少关键字

    return True


def check_benchmark_config(cfg):
    if "benchmark" not in cfg:
        logger.error("Not found key(benchmark) in config")
        return False

    if "data_dir" not in cfg["benchmark"]:
        logger.error("Not found key(data_dir) in config")
        return False

    if "test_num" not in cfg["benchmark"]:
        logger.error("Not found key(test_num) in config")
        return False

    if "dataset_module" not in cfg["benchmark"]:
        logger.error("Not found key(dataset_module) in config")
        return False

    if "dataset_name" not in cfg["benchmark"]:
        logger.error("Not found key(dataset_name) in config")
        return False

    if "py_module" not in cfg["model"]:
        logger.error("Not found key(py_module) in config")
        return False

    if "cls_name" not in cfg["model"]:
        logger.error("Not found key(cls_name) in config")
        return False

    data_dir = cfg["benchmark"]["data_dir"]
    if not os.path.exists(data_dir):
        logger.error("Not found data_dir -> {}".format(data_dir))
        return False

    num = cfg["benchmark"]["test_num"]
    if not isinstance(num, int):
        logger.error("Not found benchmark_num type not int, -> {}".format(num))
        return False

    if num < 0:
        logger.error("Not found benchmark_num must be >= 0, -> {}".format(num))
        return False

    if not cfg["model"]["py_module"]:
        logger.error("py_module is null")
        return False

    if not cfg["model"]["cls_name"]:
        logger.error("cls_name is null")
        return False

    return True


def check_demo_config(cfg):
    if "demo" not in cfg:
        logger.error("Not found key(demo) in config")
        return False

    if "data_dir" not in cfg["demo"]:
        logger.error("Not found key(data_dir) in config")
        return False

    if "num" not in cfg["demo"]:
        logger.error("Not found key(num) in config")
        return False

    if "py_module" not in cfg["model"]:
        logger.error("Not found key(py_module) in config")
        return False

    if "cls_name" not in cfg["model"]:
        logger.error("Not found key(cls_name) in config")
        return False

    data_dir = cfg["demo"]["data_dir"]
    if not os.path.exists(data_dir):
        logger.error("Not found data_dir -> {}".format(data_dir))
        return False

    num = cfg["demo"]["num"]
    if not isinstance(num, int):
        logger.error("Not found demo_num type not int, -> {}".format(num))
        return False

    if num < 0:
        logger.error("Not found demo_num must be >= 0, -> {}".format(num))
        return False

    if not cfg["model"]["py_module"]:
        logger.error("py_module is null")
        return False

    if not cfg["model"]["cls_name"]:
        logger.error("cls_name is null")
        return False

    return True