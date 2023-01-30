#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : check.py
@Time    : 2022/7/13 上午10:02
@Author  : xingwg
@Software: PyCharm
"""
import os
from utils import logger
from enum import Enum


class Phase(Enum):
    BUILD = 0
    COMPARE = 1
    DEMO = 2
    TEST = 3


def check_config(cfg, phase="build"):
    """配置文件参数合法性检查
    :param cfg:
    :param phase:
    :return:
    """
    # 检查是否缺少关键字
    if "model" not in cfg:
        logger.error("The key(model) must be in cfg")
        return False

    if "framework" not in cfg["model"]:
        logger.error("The key(framework) must be in cfg[model]")
        return False

    framework = cfg["model"]["framework"]
    framework_lists = ["caffe", "onnx", "pytorch", "mxnet", "tensorflow", "tflite", "tflite-qnn"]
    if framework not in framework_lists:
        logger.error("framework({}) must be in {}".format(framework, framework_lists))
        return False

    if "build" not in cfg:
        logger.error("The key(build) must be in cfg")
        return False

    if "weight" not in cfg["model"]:
        logger.error("The key(weight) must be in cfg[model]")
        return False

    if "enable_quant" not in cfg["build"]:
        logger.error("The key(enable_quant) must be in cfg[build]")
        return False

    if "enable_build" not in cfg["build"]:
        logger.error("The key(enable_build) must be in cfg[build]")
        return False

    if "enable_dump" not in cfg["build"]:
        logger.error("The key(enable_dump) must be in cfg[build]")
        return False

    enable_dump = cfg["build"]["enable_dump"]
    enable_dump_lists = [0, 1, 2]
    if enable_dump not in enable_dump_lists:
        logger.error("The enable_dump({}) must be in {}".format(enable_dump, enable_dump_lists))
        return False

    if "target" not in cfg["build"]:
        logger.error("The key(target) must be in cfg")
        return False

    target = cfg["build"]["target"]
    target_lists = ["nnp300", "nnp3020", "nnp310", "nnp320", "nnp400"]
    if target not in target_lists:
        logger.error("target({}) not in {}".format(target, target_lists))
        return False

    if target.startswith("nnp4") and framework not in ["onnx"]:
        logger.error("tytvm only support onnx framework")
        return False

    if "quant" not in cfg["build"]:
        logger.error("The key(quant) must be in cfg[build]")
        return False

    if "data_dir" not in cfg["build"]["quant"]:
        logger.error("The key(data_dir) must be in cfg[build][quant]")
        return False

    if "prof_img_num" not in cfg["build"]["quant"]:
        logger.error("The key(prof_img_num) must be in cfg[build][quant]")
        return False

    if "similarity_img_num" not in cfg["build"]["quant"]:
        logger.error("The key(similarity_img_num) must be in cfg[build][quant]")
        return False

    if "debug_level" not in cfg["build"]["quant"]:
        logger.error("The key(debug_level) must be in cfg[build][quant]")
        return False

    debug_level = cfg["build"]["quant"]["debug_level"]
    debug_level_lists = [-1, 0, 1, 2, 3]
    if debug_level not in debug_level_lists:
        logger.error("debug_level({}) must be in {}".format(debug_level, debug_level_lists))
        return False

    if "calib_method" not in cfg["build"]["quant"]:
        logger.error("The key(calib_method) must be in cfg[build][quant]")
        return False

    calib_method = cfg["build"]["quant"]["calib_method"]
    calib_method_lists = ["kld", "min_max", "l2norm"]
    if calib_method.startswith("percentile_"):
        pass
    else:
        if calib_method not in calib_method_lists:
            logger.error("calib_method({}) must be in {}".format(calib_method, calib_method_lists))
            return False

    if phase == "build":
        weight = cfg["model"]["weight"]
        if not os.path.exists(weight):
            logger.error("The model weight not exist -> {}".format(weight))
            return False

    # 多输入且不使用随机数据的情况下必须定义预处理
    input_lists = cfg["model"]["inputs"]

    # 以下情况必须设置自定义预处理:
    # 1.多输入指定量化数据目录，表示量化使用指定数据
    custom_preprocess_cls = cfg["build"]["quant"]["custom_preprocess_cls"]
    custom_preprocess_module = cfg["build"]["quant"]["custom_preprocess_module"]
    quant_data_dir = cfg["build"]["quant"]["data_dir"]
    if len(input_lists) > 1 and quant_data_dir and not custom_preprocess_cls:
        logger.error("multi-input must be setting custom_preprocess")
        return False
    # 2.某输入为非图像数据，且指定输入数据，表示推理仿真使用指定数据
    for _input in input_lists:
        if _input["pixel_format"] == "None" and _input["data_path"] and not custom_preprocess_cls:
            logger.error("There is non-image data, while specifying the input data_path,"
                         " custom preprocessing must be configured")
            return False

    if custom_preprocess_cls:
        if not custom_preprocess_module:
            logger.error("custom_preprocess_cls, must be setting custom_preprocess_module")
            return False

    for _input in input_lists:
        layout = _input["layout"]
        layout_lists = ["NCHW", "NHWC"]
        if layout not in layout_lists:
            logger.error("layout({}) must be in {}".format(layout, layout_lists))
            return False

        if "shape" not in _input:
            logger.error("shape must be in cfg[model][inputs]")
            return False

        if "dtype" in _input:
            dtype = _input["dtype"]
            dype_lists = ["uint8", "float32", "int16", "float16"]
            if dtype not in dype_lists:
                logger.error("dtype({}) must be in {}".format(dtype, dype_lists))
                return False

        if "mean" not in _input:
            logger.error("mean must be in cfg[model][inputs]")
            return False

        if "std" not in _input:
            logger.error("std must be in cfg[model][inputs]")
            return False

        if "resize_type" not in _input:
            logger.error("resize_type must be in cfg[model][inputs]")
            return False

        if "pixel_format" not in _input:
            logger.error("pixel_format must be in cfg[model][inputs]")
            return False

        if "layout" not in _input:
            logger.error("layout must be in cfg[model][inputs]")
            return False

        if "padding_value" not in _input:
            logger.error("padding_value must be in cfg[model][inputs]")
            return False

        if "padding_mode" not in _input:
            logger.error("padding_mode must be in cfg[model][inputs]")
            return False

        resize_type = _input["resize_type"]
        resize_type_lists = [0, 1]
        if resize_type not in resize_type_lists:
            logger.error("resize_type({}) must be in {}".format(resize_type, resize_type_lists))
            return False

        shape = _input["shape"]
        if len(shape) != 4:
            logger.error("input dim must be equal 4")
            return False

        n, c, h, w = shape
        if _input["layout"] == "NHWC":
            n, h, w, c = shape

        mean = _input["mean"]
        std = _input["std"]
        if mean is None:
            mean = [0.0 for _ in range(c)]
        else:
            if len(mean) == 1:
                mean = [mean[0] for _ in range(c)]
        if std is None:
            std = [1.0 for _ in range(c)]
            if len(std) == 1:
                std = [std[0] for _ in range(c)]

        if c != len(mean) or c != len(std) or len(mean) != len(std):
            logger.error("input channel must be equal len(mean/std)")
            return False

        pixel_format = _input["pixel_format"]
        pixel_format_lists = ["None", "RGB", "BGR", "GRAY"]
        if pixel_format not in pixel_format_lists:
            logger.error("pixel_format({}) must be in {}".format(pixel_format, pixel_format_lists))
            return False

        padding_mode = _input["padding_mode"]
        padding_mode_lists = [0, 1]
        if padding_mode not in padding_mode_lists:
            logger.error("padding_mode({}) must be in {}".format(padding_mode, padding_mode_lists))
            return False

        if _input["data_path"]:
            if not os.path.exists(_input["data_path"]):
                logger.error("data_path not exist -> {}".format(_input["data_path"]))
                return False

    # TODO
    # 检查是否缺少关键字

    return True


def check_test_config(cfg):
    if "test" not in cfg:
        logger.error("Not found key(test) in config")
        return False

    if "data_dir" not in cfg["test"]:
        logger.error("Not found key(data_dir) in config")
        return False

    if "test_num" not in cfg["test"]:
        logger.error("Not found key(test_num) in config")
        return False

    if "dataset_module" not in cfg["test"]:
        logger.error("Not found key(dataset_module) in config")
        return False

    if "dataset_cls" not in cfg["test"]:
        logger.error("Not found key(dataset_cls) in config")
        return False

    if "model_impl_module" not in cfg["model"]:
        logger.error("Not found key(model_impl_module) in config")
        return False

    if "model_impl_cls" not in cfg["model"]:
        logger.error("Not found key(model_impl_cls) in config")
        return False

    data_dir = cfg["test"]["data_dir"]
    if not os.path.exists(data_dir):
        logger.error("Not found data_dir -> {}".format(data_dir))
        return False

    num = cfg["test"]["test_num"]
    if not isinstance(num, int):
        logger.error("Not found test_num type not int, -> {}".format(num))
        return False

    if num < 0:
        logger.error("Not found test_num must be >= 0, -> {}".format(num))
        return False

    if not cfg["model"]["model_impl_module"]:
        logger.error("model_impl_module is null")
        return False

    if not cfg["model"]["model_impl_cls"]:
        logger.error("model_impl_cls is null")
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

    if "model_impl_module" not in cfg["model"]:
        logger.error("Not found key(model_impl_module) in config")
        return False

    if "model_impl_cls" not in cfg["model"]:
        logger.error("Not found key(model_impl_cls) in config")
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

    if not cfg["model"]["model_impl_module"]:
        logger.error("model_impl_module is null")
        return False

    if not cfg["model"]["model_impl_cls"]:
        logger.error("model_impl_cls is null")
        return False

    return True


def check_file_exist(filepath):
    if not os.path.exists(filepath):
        logger.error("Not found file -> {}".format(filepath))
        exit(-1)
