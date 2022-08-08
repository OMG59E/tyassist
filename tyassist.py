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
from utils.check import (
    check_config,
    check_demo_config,
    check_test_config,
    check_file_exist
)
from src.dpexec_impl import DpExec


def set_logger(op, log_dir, filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(GLogFormatter())
    logger.addHandler(file_handler)


def build(cfg):
    dpexec = DpExec(cfg)

    # 获取预处理数据
    in_datas = dpexec.get_datas(use_norm=True)

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
    if cfg["build"]["enable_quant"]:
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


def compare(cfg):
    from src.infer import Infer

    dpexec = DpExec(cfg)

    # 获取预处理数据
    in_datas = dpexec.get_datas(use_norm=True)

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


def test(cfg):
    if not check_test_config(cfg):
        exit(-1)

    dpexec = DpExec(cfg)

    data_dir = cfg["test"]["data_dir"]
    test_num = cfg["test"]["test_num"]
    dataset_module = cfg["test"]["dataset_module"]
    dataset_cls = cfg["test"]["dataset_cls"]

    _, c, h, w = dpexec.shape(0)

    dataset = None
    m = importlib.import_module(dataset_module)
    if hasattr(m, dataset_cls):
        # 实例化预处理对象
        dataset = getattr(m, dataset_cls)(data_dir)
    else:
        logger.error("{}.py has no class named {}".format(dataset_module, dataset_cls))
        exit(-1)

    model_impl_module = cfg["model"]["model_impl_module"]
    model_impl_cls = cfg["model"]["model_impl_cls"]

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

    res = model.evaluate()
    del sys.modules[dataset_module]
    del sys.modules[model_impl_module]

    logger.info("chip ave cost {:.6f}ms".format(model.ave_latency_ms))
    logger.info("[python] end2end ave cost: {:.6f}ms".format(model.end2end_latency_ms))
    logger.info("{}".format(res))
    return res


def demo(cfg):
    dpexec = DpExec(cfg)

    if dpexec.num_inputs > 1:
        logger.error("Not support demo multi-input model")
        exit(-1)

    if not check_demo_config(cfg):
        exit(-1)

    data_dir = cfg["demo"]["data_dir"]
    num = cfg["demo"]["num"]
    model_impl_module = cfg["model"]["model_impl_module"]
    model_impl_cls = cfg["model"]["model_impl_cls"]

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
    logger.info("[chip] ave cost {:.6f}ms".format(model.ave_latency_ms))
    logger.info("[python end2end] ave cost: {:.6f}ms".format(model.end2end_latency_ms))


def run(config_filepath, phase, log_dir):
    check_file_exist(config_filepath)
    basename, _ = os.path.splitext(os.path.basename(config_filepath))
    set_logger(phase, log_dir, basename)

    # 补充自定义预处理文件所在目录，必须与配置文件同目录
    config_abspath = os.path.abspath(config_filepath)
    config_dir = os.path.dirname(config_abspath)
    sys.path.insert(0, config_dir)  # 自定义模块环境变量

    config = read_yaml_to_dict(config_abspath)
    if not check_config(config):
        exit(-1)

    res = dict()
    if phase == "build":
        build(config)
    elif phase == "infer":
        compare(config)
    elif phase == "test":
        res = test(config)
    elif phase == "demo":
        demo(config)

    sys.path.remove(config_dir)
    logger.info("success")
    return res


def benchmark(mapping_file, log_dir):
    import csv
    from prettytable import PrettyTable

    header = ["ModelName", "InputSize", "Dataset", "Num", "Acc./mAP.", "Latency(ms)"]
    table = PrettyTable(header)
    csv_filepath = "benchmark.csv"
    f = open(csv_filepath, "w")
    f_csv = csv.writer(f)
    f_csv.writerow(header)

    check_file_exist(mapping_file)
    models_dict = read_yaml_to_dict(mapping_file)["models"]
    root = os.getcwd()
    for model_name in models_dict:
        config_filepath = models_dict[model_name]
        config_abspath = os.path.abspath(config_filepath)
        config_dir = os.path.dirname(config_abspath)
        # 判断是否已存在模型
        model_cfg = read_yaml_to_dict(config_abspath)
        save_path = os.path.abspath(model_cfg["model"]["save_dir"])
        if not os.path.join(save_path, "net_combine.bin"):
            logger.warning("Model not found -> {}".format(save_path))
            continue

        os.chdir(config_dir)  # 切换至模型目录
        res = run(config_abspath, "test", log_dir)
        # logger.info("{}".format(res))
        os.chdir(root)  # 切换根目录

        row = list()
        if "top1" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["top1"], res["top5"]), res["latency"]]
        elif "map" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["map"], res["map50"]), res["latency"]]
        table.add_row(row)
        f_csv.writerow(row)
    f.close()
    print(table, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TyAssist Tool")
    parser.add_argument("type", type=str, choices=("demo", "test", "infer", "benchmark", "build"),
                        help="Please choose one of them")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Please specify a configuration file")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Please specify a log dir, default ./logs")

    args = parser.parse_args()

    if args.type == "benchmark":
        benchmark(args.config, args.log_dir)
    else:
        _ = run(args.config, args.type, args.log_dir)
