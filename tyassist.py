#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: tyexec.py 
@time: 2022/12/19
@Author  : xingwg
@software: PyCharm 
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
from utils.enum_type import PixelFormat, DataType
from utils.check import (
    check_config,
    check_demo_config,
    check_test_config,
    check_file_exist
)
from src.nnp3xx_tyexec import Nnp3xxTyExec
from src.nnp4xx_tyexec import Nnp4xxTyExec


def set_logger(op, log_dir, filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filepath = os.path.join(log_dir, "{}-{}-{}.log".format(filename, op, t))
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(GLogFormatter())
    logger.addHandler(file_handler)


def get_tyexec(cfg):
    target = cfg["build"]["target"]
    if target.startswith("nnp3"):
        return Nnp3xxTyExec(cfg)
    elif target.startswith("nnp4"):
        tyexec = Nnp4xxTyExec(cfg)
        tyexec.set_nnp4xx_env()
        return tyexec
    else:
        logger.error("Not support target -> {}".format(target))
        exit(-1)


def build(cfg):
    logger.info("{}".format(cfg))

    tyexec = get_tyexec(cfg)

    tyexec.get_version()
    tyexec.x2relay()  # model to relay_func

    in_datas = tyexec.get_datas(use_norm=False, force_cr=True, to_file=True)  # 量化后模型输入数据
    tyexec.quantization(in_datas)
    tvm_fixed_output = tyexec.tvm_fixed_inference(in_datas, to_file=True)

    in_datas = tyexec.get_datas(use_norm=True, force_cr=True, to_file=True)  # 原模型输入数据
    tvm_float_output = tyexec.tvm_float_inference(in_datas, to_file=True)

    iss_fixed_output = tyexec.build(in_datas)

    tyexec.compress_analysis()
    tyexec.get_relay_mac()  # print mac/flops/cycles info
    tyexec.get_device_type()  # print op backend info

    # 计算相似度
    for idx in range(len(tvm_float_output)):
        dist = cosine_distance(tvm_float_output[idx], tvm_fixed_output[idx])
        logger.info("[Build] float(tvm) output tensor[{}] shape:{} dtype:{}".format(
            idx, tvm_float_output[idx].shape, tvm_float_output[idx].dtype))
        logger.info("[Build] fixed(tvm) output tensor[{}] shape:{} dtype:{}".format(
            idx, tvm_fixed_output[idx].shape, tvm_fixed_output[idx].dtype))
        if iss_fixed_output:
            logger.info("[Build] fixed(iss) output tensor[{}] shape:{} dtype:{}".format(
                idx, iss_fixed_output[idx].shape, iss_fixed_output[idx].dtype))
        logger.info("[Build] float(tvm) vs fixed(tvm) output tensor[{}] similarity={:.6f}".format(idx, dist))
        if iss_fixed_output:
            dist = cosine_distance(tvm_float_output[idx], iss_fixed_output[idx])
            logger.info("[Build] float(tvm) vs fixed(iss) output tensor[{}] similarity={:.6f}".format(idx, dist))

    for idx in range(len(tvm_fixed_output)):
        if iss_fixed_output:
            dist = cosine_distance(tvm_fixed_output[idx], iss_fixed_output[idx])
            logger.info("[Build] fixed(tvm) vs fixed(iss) output tensor[{}] similarity={:.6f}".format(idx, dist))


def compare(cfg):
    logger.info("{}".format(cfg))
    tyexec = get_tyexec(cfg)
    fixed_outputs, backend = tyexec.infer()

    # compare
    for idx, fixed_output in enumerate(fixed_outputs):
        tvm_float_out_path = os.path.join(tyexec.result_dir, "tvm_float_out_{}.bin".format(idx))
        tvm_fixed_out_path = os.path.join(tyexec.result_dir, "tvm_fixed_out_{}.bin".format(idx))
        if not os.path.exists(tvm_fixed_out_path):
            logger.error("Not found tvm_fixed_out_path -> {}".format(tvm_fixed_out_path))
            exit(-1)
        if not os.path.exists(tvm_float_out_path):
            logger.error("Not found tvm_float_out_path -> {}".format(tvm_float_out_path))
            exit(-1)
        tvm_fixed_out = np.fromfile(tvm_fixed_out_path, dtype=fixed_output.dtype)
        tvm_float_out = np.fromfile(tvm_float_out_path, dtype=fixed_output.dtype)
        dist0 = cosine_distance(fixed_output, tvm_fixed_out)
        dist1 = cosine_distance(fixed_output, tvm_float_out)
        logger.info("[Compare] fixed({}) vs fixed(tvm) output tensor[{}] similarity={:.6f}".format(backend, idx, dist0))
        logger.info("[Compare] fixed({}) vs float(tvm) output tensor[{}] similarity={:.6f}".format(backend, idx, dist1))
        if tyexec.enable_dump:
            iss_fixed_out_path = os.path.join(tyexec.result_dir, "iss_fixed_out_{}.bin".format(idx))
            if not os.path.exists(iss_fixed_out_path):
                logger.error("Not found iss_fixed_out_path -> {}".format(iss_fixed_out_path))
                exit(-1)
            iss_fixed_out = np.fromfile(iss_fixed_out_path, dtype=fixed_output.dtype)
            dist2 = cosine_distance(fixed_output, iss_fixed_out)
            logger.info("[Compare] fixed({}) vs fixed(iss) output tensor[{}] similarity={:.6f}".format(backend, idx, dist2))


def profile(cfg):
    logger.info("{}".format(cfg))
    tyexec = get_tyexec(cfg)
    tyexec.profile()


def test(cfg, dtype):
    logging.getLogger("deepeye").setLevel(logging.WARNING)

    if not check_test_config(cfg):
        exit(-1)

    tyexec = get_tyexec(cfg)

    if tyexec.num_inputs > 1:
        logger.error("Not support multi-input model")
        exit(-1)

    data_dir = cfg["test"]["data_dir"]
    test_num = cfg["test"]["test_num"]
    dataset_module = cfg["test"]["dataset_module"]
    dataset_cls = cfg["test"]["dataset_cls"]

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
            inputs=tyexec.inputs,
            dataset=dataset,
            test_num=test_num,
            enable_aipp=True,
            target=tyexec.target,
            dtype=dtype,   # int8/tvm-fp32/tvm-int8
        )
    else:
        logger.error("{}.py has no class named {}".format(model_impl_module, model_impl_cls))
        exit(-1)

    if dtype == "int8":
        model.load(tyexec.model_path)
    elif dtype == "tvm-fp32":
        model.load(tyexec.original_json_path)
    elif dtype == "tvm-int8":
        model.load(tyexec.quant_json_path)

    res = model.evaluate()
    del sys.modules[dataset_module]
    del sys.modules[model_impl_module]

    logger.info("average cost {:.6f}ms".format(model.ave_latency_ms))
    logger.info("[end2end] average cost: {:.6f}ms".format(model.end2end_latency_ms))
    logger.info("{}".format(res))
    return res


def demo(cfg, dtype):
    logging.getLogger("deepeye").setLevel(logging.WARNING)
    logger.info(cfg)

    tyexec = get_tyexec(cfg)

    if tyexec.num_inputs > 1:
        logger.error("Not support multi-input model")
        exit(-1)

    if not check_demo_config(cfg):
        exit(-1)

    data_dir = cfg["demo"]["data_dir"]
    num = cfg["demo"]["num"]
    model_impl_module = cfg["model"]["model_impl_module"]
    model_impl_cls = cfg["model"]["model_impl_cls"]

    file_list = os.listdir(data_dir)
    file_list = file_list if num > len(file_list) else file_list[0:num]

    model = None
    m = importlib.import_module(model_impl_module)
    if hasattr(m, model_impl_cls):
        # 实例化预处理对象
        model = getattr(m, model_impl_cls)(
            inputs=tyexec.inputs,
            dataset=None,
            test_num=0,
            enable_aipp=True,
            target=tyexec.target,
            dtype=dtype,   # int8/tvm-fp32/tvm-int8
        )
    else:
        logger.error("{}.py has no class named {}".format(model_impl_module, model_impl_cls))
        exit(-1)

    if dtype == "int8":
        model.load(tyexec.model_path)
    elif dtype == "tvm-fp32":
        model.load(tyexec.original_json_path)
    elif dtype == "tvm-int8":
        model.load(tyexec.quant_json_path)

    for filename in file_list:
        _, ext = os.path.splitext(filename)
        if ext not in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
            logger.warning("file ext invalid -> {}".format(filename))
            continue

        filepath = os.path.join(data_dir, filename)
        model.demo(filepath)
    logger.info("average cost {:.6f}ms".format(model.ave_latency_ms))
    logger.info("[end2end] average cost: {:.6f}ms".format(model.end2end_latency_ms))


def run(config_filepath, phase, dtype, target):
    # 补充自定义预处理文件所在目录，必须与配置文件同目录
    config_abspath = os.path.abspath(config_filepath)
    config_dir = os.path.dirname(config_abspath)
    sys.path.insert(0, config_dir)  # 自定义模块环境变量

    config = read_yaml_to_dict(config_abspath)
    if not check_config(config, phase):
        exit(-1)
    # 更新target，优先使用命令行
    if target is not None:
        config["build"]["target"] = target

    res = dict()
    if phase == "build":
        build(config)
    elif phase == "compare":
        compare(config)
    elif phase == "test":
        res = test(config, dtype)
    elif phase == "demo":
        demo(config, dtype)
    elif phase == "profile":
        profile(config)

    sys.path.remove(config_dir)
    logger.info("success")
    return res


def benchmark(mapping_file, dtype, target):
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
        res = run(config_abspath, "test", dtype, target)
        # logger.info("{}".format(res))
        os.chdir(root)  # 切换根目录

        row = list()
        if "top1" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["top1"], res["top5"]), res["latency"]]
        elif "map" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["map"], res["map50"]), res["latency"]]
        elif "easy" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}/{}".format(res["easy"], res["medium"], res["hard"]), res["latency"]]
        table.add_row(row)
        f_csv.writerow(row)
    f.close()
    logger.info("\n{}".format(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TyAssist Tool")
    parser.add_argument("type", type=str, choices=("demo", "test", "compare", "benchmark", "build", "profile"),
                        help="Please specify a operator")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Please specify a configuration file")
    parser.add_argument("--target", type=str, required=False,
                        choices=("nnp300", "nnp3020", "nnp310", "nnp320", "nnp400"),
                        help="Please specify a chip target")
    parser.add_argument("--dtype", "-t", type=str, default="int8", choices=("int8", "tvm-fp32", "tvm-int8"),
                        help="Please specify one of them")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Please specify a log dir, default ./logs")

    args = parser.parse_args()

    check_file_exist(args.config)
    basename, _ = os.path.splitext(os.path.basename(args.config))
    set_logger(args.type, args.log_dir, basename)

    dirname, filename = os.path.split(os.path.abspath(__file__))
    version_path = os.path.join(dirname, "version")
    if not os.path.exists(version_path):
        logger.warning("Not found version file")
    with open(version_path, "rb") as f:
        VERSION = f.readline().decode("gbk").strip()
        logger.info("{} with TyAssist version: {}".format(args.type, VERSION))

    if args.type == "benchmark":
        benchmark(args.config, args.dtype, args.target)
    else:
        _ = run(args.config, args.type, args.dtype, args.target)
