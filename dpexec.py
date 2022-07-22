#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : dpexec.py
@Time    : 2022/7/1 下午2:17
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import os
import sys

import numpy as np
import argparse
import importlib
import logging
import time
from utils import logger
from utils.glog_format import GLogFormatter
from utils.parser import read_yaml_to_dict
from utils.dist_metrics import cosine_distance
from utils.check import check_config, check_demo_config, check_test_config
from src.dpexec_impl import DpExec


def set_logger(op, log_dir, filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(GLogFormatter())
    logger.addHandler(file_handler)


def main(args):
    config_file = args.config
    if not os.path.exists(config_file):
        logger.error("Not found ini file -> {}".format(config_file))
        exit(-1)
    basename, _ = os.path.splitext(os.path.basename(config_file))
    set_logger(args.type, args.log_dir, basename)

    config = read_yaml_to_dict(config_file)
    # 补充自定义预处理文件所在目录，必须与配置文件同目录
    config_abspath = os.path.abspath(config_file)
    config_dir = os.path.dirname(config_abspath)
    sys.path.append(config_dir)  # 自定义模块的环境变量
    # config_name, _ = os.path.splitext(os.path.basename(config_abspath))
    # config["build"]["quant"]["custom_preprocess_dir"] = config_dir

    # 参数检查
    # mode_name = config["model"]["name"]
    # if config_name != mode_name:
    #     logger.error("config_name({}) must be equal model_name({})".format(config_name, mode_name))
    #     exit(-1)

    if not check_config(config):
        exit(-1)

    dpexec = DpExec(config)

    # 获取预处理数据
    in_datas = dpexec.get_datas(use_norm=True)

    if args.type == "build":
        # 将模型转译成relay格式
        dpexec.x2relay()

        # 转译后的浮点模型仿真，仿真结果可视为原模型的浮点结果，作为绝对参考
        # 这里需要预处理后的float32数据
        host_tvm_float_output = dpexec.tvm_float_output(in_datas)

        # 量化编译阶段
        # - 存在自定义预处理的情况下输入必须是float32数据，不使用内部CR和norm
        # - 使用内部CR和norm预处理，输入必须是uint8数据；但如果打开dump功能，则又不能使能内部CR，但norm有效
        if not dpexec.has_custom_preprocess:
            in_datas = dpexec.get_datas(use_norm=False)  # 仅需要自行resize和cvtColor

        host_iss_fixed_output = None
        if config["build"]["enable_quant"]:
            # 量化模型
            dpexec.relay_quantization()
            # 编译生成芯片模型，
            host_iss_fixed_output = dpexec.make_netbin(in_datas)
        else:
            # 加载已生成的量化模型
            dpexec.load_relay_from_json()

        # 量化后定点模型仿真，目前只支持在cpu上进行软仿
        host_tvm_fixed_output = dpexec.tvm_fixed_output(in_datas)

        # 计算相似度
        for idx in range(len(host_tvm_float_output)):
            dist = cosine_distance(host_tvm_float_output[idx], host_tvm_fixed_output[idx])
            logger.info("[runoncpu] float(tvm) output tensor [{}] shape:{} dtype:{}".format(
                idx, host_tvm_float_output[idx].shape, host_tvm_float_output[idx].dtype))
            logger.info("[runoncpu] fixed(tvm) output tensor [{}] shape:{} dtype:{}".format(
                idx, host_tvm_fixed_output[idx].shape, host_tvm_fixed_output[idx].dtype))
            if host_iss_fixed_output:
                logger.info("[runoncpu] fixed(iss) output tensor [{}] shape:{} dtype:{}".format(
                    idx, host_iss_fixed_output[idx].shape, host_iss_fixed_output[idx].dtype))
            logger.info("[runoncpu] float(tvm) vs fixed(tvm) output tensor [{}] similarity: {:.6f}".format(idx, dist))
            if host_iss_fixed_output:
                dist = cosine_distance(host_tvm_float_output[idx], host_iss_fixed_output[idx])
                logger.info("[runoncpu] float(tvm) vs fixed(iss) output tensor [{}] similarity: {:.6f}".format(idx, dist))

    elif args.type == "infer":
        from src import Infer

        # device端 硬仿
        infer = Infer(
            net_cfg_file="/DEngine/tyhcp/net.cfg",
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            ip="127.0.0.1",  # 存在net_cfg会覆盖ip设置
            port=9090,
            enable_dump=dpexec.enable_dump,
            max_batch=1  # 目前仅支持最大batch 1
        )
        # 加载模型
        infer.load(dpexec.model_dir)

        if not dpexec.has_custom_preprocess:
            in_datas = dpexec.get_datas(use_norm=False)  # 仅需要自行resize和cvtColor

        in_datas = [in_datas[key] for key in in_datas]
        outputs = infer.run(in_datas, to_file=True)

        # 比对相似度
        for idx, chip_fixed_output in enumerate(outputs):
            tvm_float_out_path = os.path.join(dpexec.model_dir, "result", "host_tvm_float_out_{}.bin".format(idx))
            tvm_fixed_out_path = os.path.join(dpexec.model_dir, "result", "host_tvm_fixed_out_{}.bin".format(idx))
            if not os.path.exists(tvm_fixed_out_path):
                logger.error("Not found tvm_fixed_out_path -> {}".format(tvm_fixed_out_path))
                exit(-1)
            if not os.path.exists(tvm_float_out_path):
                logger.error("Not found tvm_float_out_path -> {}".format(tvm_float_out_path))
                exit(-1)
            tvm_fixed_out = np.fromfile(tvm_fixed_out_path, dtype=chip_fixed_output.dtype)
            tvm_float_out = np.fromfile(tvm_float_out_path, dtype=chip_fixed_output.dtype)
            dist0 = cosine_distance(chip_fixed_output, tvm_fixed_out)
            dist1 = cosine_distance(chip_fixed_output, tvm_float_out)
            logger.info("[runonchip] fixed(chip) vs fixed(tvm) output tensor: [{}] similarity={:.6f}".format(idx, dist0))
            logger.info("[runonchip] fixed(chip) vs float(tvm) output tensor: [{}] similarity={:.6f}".format(idx, dist1))

    elif args.type == "test":
        if not check_test_config(config):
            exit(-1)

        data_dir = config["test"]["data_dir"]
        test_num = config["test"]["test_num"]
        dataset_module = config["test"]["dataset_module"]
        dataset_cls = config["test"]["dataset_cls"]

        _, c, h, w = dpexec.shape(0)

        dataset = None
        m = importlib.import_module(dataset_module)
        if hasattr(m, dataset_cls):
            # 实例化预处理对象
            dataset = getattr(m, dataset_cls)(data_dir)
        else:
            logger.error("{}.py has no class named {}".format(dataset_module, dataset_cls))
            exit(-1)

        model_impl_module = config["model"]["model_impl_module"]
        model_impl_cls = config["model"]["model_impl_cls"]

        model = None
        m = importlib.import_module(model_impl_module)
        if hasattr(m, model_impl_cls):
            # 实例化预处理对象
            model = getattr(m, model_impl_cls)(
                (h, w),
                mean=dpexec.mean(0),
                std=dpexec.std(0),
                use_norm=False,
                use_rgb=True if dpexec.data_layout(0) == "RGB" else False,
                resize_type=dpexec.resize_type(0),
                padding_value=dpexec.padding_value(0),
                padding_mode=dpexec.padding_mode(0),
                dataset=dataset,
                test_num=test_num
            )
        else:
            logger.error("{}.py has no class named {}".format(model_impl_module, model_impl_cls))
            exit(-1)

        model.load(
            dpexec.model_dir,
            net_cfg_file="/DEngine/tyhcp/net.cfg",
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            ip="127.0.0.1",  # 存在net_cfg会覆盖ip设置
            port=9090,
            enable_dump=False,
            max_batch=1  # 目前仅支持最大batch 1
        )

        model.evaluate()

    elif args.type == "demo":
        if dpexec.num_inputs > 1:
            logger.error("Not support demo multi-input model")
            exit(-1)

        if not check_demo_config(config):
            exit(-1)

        data_dir = config["demo"]["data_dir"]
        num = config["demo"]["num"]
        model_impl_module = config["model"]["model_impl_module"]
        model_impl_cls = config["model"]["model_impl_cls"]

        file_list = os.listdir(data_dir)
        file_list = file_list if num > len(file_list) else file_list[0:num]

        _, c, h, w = dpexec.shape(0)

        model = None

        m = importlib.import_module(model_impl_module)
        if hasattr(m, model_impl_cls):
            # 实例化预处理对象
            model = getattr(m, model_impl_cls)(
                (h, w),
                mean=dpexec.mean(0),
                std=dpexec.std(0),
                use_norm=False,
                use_rgb=True if dpexec.data_layout(0) == "RGB" else False,
                resize_type=dpexec.resize_type(0),
                padding_value=dpexec.padding_value(0),
                padding_mode=dpexec.padding_mode(0),
                dataset=None
            )
        else:
            logger.error("{}.py has no class named {}".format(model_impl_module, model_impl_cls))
            exit(-1)

        model.load(
            dpexec.model_dir,
            net_cfg_file="/DEngine/tyhcp/net.cfg",
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            ip="127.0.0.1",  # 存在net_cfg会覆盖ip设置
            port=9090,
            enable_dump=False,
            max_batch=1  # 目前仅支持最大batch 1
        )

        for filename in file_list:
            _, ext = os.path.splitext(filename)
            if ext not in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
                logger.warning("file ext invalid -> {}".format(filename))
                continue

            filepath = os.path.join(data_dir, filename)
            model.demo(filepath)

    elif args.type == "benchmark":
        pass

    logger.info("success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TyAssist Tool")
    parser.add_argument("type", type=str, choices=("demo", "test", "infer", "benchmark", "build"),
                        help="Please choose one of them")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Please specify a configuration file")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Please specify a log dir, default ./logs")

    args = parser.parse_args()

    main(args)
