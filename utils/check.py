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


def check_tf_nhwc2nchw(inputs):
    if len(inputs) > 1:
        # 数据布局格式必须一致
        for idx in range(1, len(inputs)):
            if inputs[0]["layout"] != inputs[idx]["layout"]:
                logger.error("input layout must be the same")
                exit(-1)
        

def check_model_cfg(cfg):
    # 检查模型信息相关描述
    if "model" not in cfg:
        logger.error("The model field must be in cfg")
        return False

    if "target" not in cfg["build"]:
        logger.error("The target field must be in cfg[build]")
        return False
    target = cfg["build"]["target"]
    target_lists = ["nnp300", "nnp3020", "nnp310", "nnp315m", "nnp320", "nnp400"]
    if target not in target_lists:
        logger.error("target({}) not in {}".format(target, target_lists))
        return False

    model_cfg = cfg["model"]
    if "framework" not in model_cfg:
        logger.error("The framework field must be in cfg[model]")
        return False
    framework = model_cfg["framework"]
    if target.startswith("nnp3"):
        framework_lists = ["caffe", "onnx", "pytorch", "mxnet", "tensorflow", "tflite", "tflite-qnn", "onnx-qnn"]
    elif target.startswith("nnp4"):
        # 内部支持mxnet，对外不支持
        framework_lists = ["onnx", "onnx-qnn", "mxnet"]
    else:
        logger.error("Not support target -> {}".format(target))
        return False
    if framework not in framework_lists:
        logger.error("framework({}) must be in {}".format(framework, framework_lists))
        return False
    
    model_name = model_cfg.get("name")
    if model_name and not isinstance(model_name, str):
        logger.error("model name expect str type")
        return False
    
    save_dir = model_cfg.get("save_dir")
    if save_dir and not isinstance(save_dir, str):
        logger.error("save_dir expect str type")
        return False  
        
    if "weight" not in cfg["model"]:
        logger.error("The weight field must be in cfg[model]")
        return False
    weight = model_cfg["weight"]
    if target.startswith("nnp4") and framework == "mxnet":
            weight += "-0000.params"
    if not os.path.exists(weight):
        logger.error("The weight path not exist, weight: {}".format(weight))
        return False

    graph = cfg["model"].get("graph")
    if target.startswith("nnp3") and framework in ["caffe", "mxnet"] and not os.path.exists(graph):
        logger.error("The graph path not exist, graph: {}".format(graph))
        return False  
    if target.startswith("nnp4") and framework == "mxnet":
        graph += "-symbol.json"
        if not os.path.exists(graph):
            logger.error("The graph path not exist, graph: {}".format(graph))
            return False
    
    if "inputs" not in model_cfg:
        logger.error("The inputs field must in cfg[model]")
        return False
    inputs_cfg = model_cfg["inputs"]
    if not isinstance(inputs_cfg, list):
        logger.error("The inputs field expect list type")
        return False
    
    for input_cfg in inputs_cfg:
        fields = ["name", "shape", "layout", "pixel_format"]
        for field in fields:
            if field not in input_cfg:
                logger.error("The {} field must be in cfg[model][inputs]".format(field))
                return False
            if not input_cfg[field]:
                logger.error("The {} field must be set".format(field))
                return False
            if field == "shape":
                continue
            if not isinstance(input_cfg[field], str):
                logger.error("The {} field expect str type".format(field))
                return False  
        shape = input_cfg["shape"]
        if not isinstance(shape, list):
            logger.error("The shape field expect list type")
            return False  
        layout = input_cfg["layout"]
        if layout not in ["NCHW", "NHWC", "None"]:
            logger.error("The layout({}) not in [NCHW, NHWC, None]".format(layout))
            return False
        if layout in ["NCHW", "NHWC"] and len(shape) != 4:
            logger.error("shape error -> {}".format(shape))
            return False 
        pixel_format = input_cfg["pixel_format"]
        if pixel_format not in ["BGR", "RGB", "GRAY", "None"]:
            logger.error("The pixel_format({}) not in [BGR, RGB, GRAY, None]".format(pixel_format))
            return False        
        if layout == "None" and pixel_format != "None":
            logger.error("The layout is None, pixel_format must be None")
            return False
        
        mean = input_cfg.get("mean")
        std = input_cfg.get("std")
        if (mean is None and std is not None) or (mean is not None and std is None):
            logger.error("The mean/std is None, then std/mean must be None")
            return False
        
        # 配置均值方差情况
        if mean and std:
            norm_axis = input_cfg.get("norm_axis")
            if norm_axis is None:
                norm_axis = 1
            if norm_axis >= len(shape) and norm_axis < 0:
                logger.error("The norm_axis out of range")
                return False
            dim = shape[norm_axis]
            if len(mean) != len(std):
                logger.error("mean_len != std_len")
                return False
            # 配置均值方差的情况
            if len(mean) != 1 and dim != len(mean):
                logger.error("mean size not match norm_dim")
                return False
        # 检查数据类型
        dtype = input_cfg.get("dtype")
        dtype_lists = ["uint8", "float32", "int16", "float16"]
        if dtype is not None and dtype not in dtype_lists:
            logger.error("dtype({}) must be in {}".format(dtype, dtype_lists))
            return False
        # 检查测试数据
        data_path = input_cfg.get("data_path")
        if data_path:
            if not (isinstance(data_path, str) or isinstance(data_path, list)):
                logger.error("data_path expect str or list type")
                return False
            if isinstance(data_path, str) and not os.path.exists(data_path):
                logger.error("data_path not exist -> {}".format(data_path))
                return False
            if isinstance(data_path, list):
                for path in data_path:
                    if not isinstance(path, str):
                        logger.error("data_path expect str type")
                        return False
                    if not os.path.exists(data_path):
                        logger.error("data_path not exist -> {}".format(path))
                        return False  
        # 检查自定义预处理设置
        # pixel_format为None或存在多个输入，同时指定data_path或quant_data_dir
        quant_data_dir = cfg["build"]["quant"].get("data_dir")
        if (pixel_format == "None" or len(inputs_cfg) > 1) and (data_path or quant_data_dir):
            custom_preprocess_module = cfg["build"]["quant"].get("custom_preprocess_module")
            custom_preprocess_cls = cfg["build"]["quant"].get("custom_preprocess_cls")
            if (not custom_preprocess_module) or (not custom_preprocess_cls):
                logger.error("custom_preprocess_module/custom_preprocess_cls not set")
                return False
            
        # 输入为图像数据 
        if pixel_format != "None" and not cfg["build"]["quant"].get("custom_preprocess_module"):
            resize_type = input_cfg.get("resize_type")
            resize_type_lists = [0, 1]
            if resize_type is not None and resize_type not in resize_type_lists:
                logger.error("resize_type({}) must be in {}".format(resize_type, resize_type_lists))
                return False
            padding_value = input_cfg.get("padding_value")
            if padding_value is not None:
                if not isinstance(padding_value, int):
                    logger.error("padding_value expect int type")
                    return False
                if padding_value < 0 or padding_value > 255:
                    logger.error("padding_value must be >=0 or <= 255")
                    return False
            padding_mode = input_cfg.get("padding_mode")
            padding_mode_lists = [0, 1]
            if padding_mode is not None and padding_mode not in padding_mode_lists:
                logger.error("padding_mode({}) must be in {}".format(padding_mode, padding_mode_lists))
                return False
            # 输入为图像检查aipp
            enable_aipp = input_cfg.get("enable_aipp")
            if enable_aipp is not None and enable_aipp not in [True, False]:
                logger.error("enable_aipp must be in [True, False]")
                return False
        
    # 检查输出配置
    outputs = model_cfg.get("outputs", [])
    if not isinstance(outputs, list):
        logger.error("model outputs expect list type")
        return False
    if framework in ["tensorflow"] and not outputs:
        logger.error("Not set outputs, when framework is tensorflow, must be set it")
        return False
    
    # TODO 目前限制多个输入的数据布局必须一致
    # framework_lists = ["tensorflow", "tflite", "tflite-qnn"]
    # if framework in framework_lists:
    #     check_tf_nhwc2nchw(inputs_cfg)
    return True


def check_build_cfg(cfg):
    # 检查量化编译相关配置
    if "build" not in cfg:
        logger.error("The build field must be in cfg")
        return False
    
    build_cfg = cfg["build"]
    if "target" not in build_cfg:
        logger.error("The target field must be in cfg[build]")
        return False
    
    target = build_cfg["target"]
    target_lists = ["nnp300", "nnp3020", "nnp310", "nnp315m", "nnp320", "nnp400"]
    if target not in target_lists:
        logger.error("target({}) not in {}".format(target, target_lists))
        return False
    
    enable_quant = build_cfg.get("enable_quant")
    enable_build = build_cfg.get("enable_build")
    suppress_long_func = build_cfg.get("suppress_long_func")
    bool_lists = [True, False]
    if enable_quant is not None and enable_quant not in bool_lists:
        logger.error("The enable_quant expect bool type")
        return False
 
    if enable_build is not None and enable_build not in bool_lists:
        logger.error("The enable_build expect bool type")
        return False

    if suppress_long_func is not None and suppress_long_func not in bool_lists:
        logger.error("The suppress_long_func expect bool type")
        return False
    
    opt_level = build_cfg.get("opt_level")
    if opt_level is not None and opt_level not in [0, 2]:
        logger.error("build opt_level must be in [0, 2]")
        return False

    enable_dump = build_cfg.get("enable_dump")
    enable_dump_lists = [0, 1, 2]
    if enable_dump is not None and enable_dump not in enable_dump_lists:
        logger.error("The enable_dump({}) must be in {}".format(enable_dump, enable_dump_lists))
        return False
    
    # 检查量化配置
    if "quant" not in build_cfg:
        logger.error("The quant field must be in cfg[build]")
        return False
    
    quant_cfg = build_cfg["quant"]
    prof_img_num = quant_cfg.get("prof_img_num")
    if prof_img_num is None:
        logger.error("The prof_img_num is None")
        return False
    if not isinstance(prof_img_num, int):
        logger.error("The prof_img_num expect int type")
        return False
        
    data_dir = quant_cfg.get("data_dir")
    if not data_dir:
        if prof_img_num == 0:
            logger.error("The prof_img_num must be > 0, when quant_data_dir is None")
            return False
    else:
        if not isinstance(data_dir, str):
            logger.error("The data_dir expect str type")
            return False
        if prof_img_num < 0:
            logger.error("The prof_img_num must be >= 0")
            return False
        if not os.path.exists(data_dir):
            logger.error("The quant_data_dir not exist!")
            return False

    similarity_img_num = quant_cfg.get("similarity_img_num")
    if similarity_img_num is not None:
        if not isinstance(similarity_img_num, int):
            logger.error("The similarity_img_num expect int type")
            return False
        if similarity_img_num < 0:
            logger.error("The similarity_img_num must be > 0")
            return False

    similarity_dataset = quant_cfg.get("similarity_dataset")
    if similarity_dataset:
        if not isinstance(similarity_dataset, str):
            logger.error("The similarity_dataset expect str type")
            return False
        if not os.path.exists(similarity_dataset):
            logger.error("Not found similarity_dataset -> {}".format(similarity_dataset))
            return False
        
    debug_level = quant_cfg.get("debug_level")
    debug_level_lists = [-1, 0, 1, 2, 3]
    if debug_level is not None and debug_level not in debug_level_lists:
        logger.error("debug_level({}) must be in {}".format(debug_level, debug_level_lists))
        return False

    calib_method = quant_cfg.get("calib_method")
    if calib_method:
        calib_method_lists = ["kld", "min_max", "l2norm"]
        if not calib_method.startswith("percentile_") and calib_method not in calib_method_lists:
            logger.error("calib_method({}) must be in {}".format(calib_method, calib_method_lists))
            return False
    
    disable_pass = quant_cfg.get("disable_pass")
    if disable_pass is not None and not isinstance(disable_pass, list):
        logger.error("The disable_pass expect list type")
        return False

    num_cube = quant_cfg.get("num_cube")
    num_cube_lists = [1, 2, 3]
    if num_cube is not None and num_cube not in num_cube_lists:
        if not isinstance(disable_pass, list):
            logger.error("num_cube({}) must be in {}".format(num_cube, num_cube_lists))
            return False
        
    skip_layer_idxes = quant_cfg.get("skip_layer_idxes")
    skip_layer_types = quant_cfg.get("skip_layer_types")
    skip_layer_names = quant_cfg.get("skip_layer_names")
    if skip_layer_idxes is not None and not isinstance(skip_layer_idxes, list):
        logger.error("The skip_layer_idxes expect list type")
        return False

    if skip_layer_types is not None and not isinstance(skip_layer_types, list):
        logger.error("The skip_layer_types expect list type")
        return False

    if skip_layer_names is not None and not isinstance(skip_layer_names, list):
        logger.error("The skip_layer_names expect list type")
        return False
    return True


def check_config(cfg, phase="build"):
    """配置文件参数合法性检查
    :param cfg:
    :param phase:
    :return:
    """
    if not check_build_cfg(cfg):
        return False
    if not check_model_cfg(cfg):
        return False
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
