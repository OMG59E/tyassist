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
import traceback
import numpy as np
import argparse
import importlib
import logging
import time
from utils.gen_config import gen_default_config
from prettytable import PrettyTable
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
        return Nnp4xxTyExec(cfg)
    else:
        logger.error("Not support target -> {}".format(target))
        exit(-1)


def build(cfg):
    try:
        t_start = time.time()
        logger.info("{}".format(cfg))
        tyexec = get_tyexec(cfg)
        tyexec.set_env()
        tyexec.get_version()
        tyexec.x2relay()  # model to relay_func

        in_datas = tyexec.get_datas(force_cr=True, to_file=True)  # 量化后模型输入

        # step1 quantization
        tyexec.quantization(in_datas)

        # step2 build
        tyexec.build(in_datas)

        # step3 dump output layer by layer
        tyexec.iss_dump_output(in_datas)

        # step4 iss/tvm fixed simu
        iss_fixed_output = tyexec.iss_fixed_inference(in_datas, to_file=True)
        tvm_fixed_output = tyexec.tvm_fixed_inference(in_datas, to_file=True)

        # step5 tvm float simu
        if not tyexec.is_qnn:
            in_datas = tyexec.get_datas(use_norm=True, force_cr=True, to_file=True)  # 浮点模型输入
            tvm_float_output = tyexec.tvm_float_inference(in_datas, to_file=True)

        # tyexec.model_analysis()
        # tyexec.compress_analysis()
        tyexec.get_profile_info()
        tyexec.get_relay_mac()  # print mac/flops/cycles info
        # tyexec.get_device_type()  # print op backend info

        # print span
        header = ["Phase", "Span/s"]
        table = PrettyTable(header)
        table.add_row(["x2relay", "{:.3f}".format(tyexec.x2relay_span)])
        table.add_row(["quantization", "{:.3f}".format(tyexec.quantization_span)])
        table.add_row(["build", "{:.3f}".format(tyexec.build_span)])
        table.add_row(["tvm_float_simu", "{:.3f}".format(tyexec.tvm_float_simu_span)])
        table.add_row(["tvm_fixed_simu", "{:.3f}".format(tyexec.tvm_fixed_simu_span)])
        table.add_row(["iss_fixed_simu", "{:.3f}".format(tyexec.iss_simu_span)])
        table.add_row(["tvm_layerwise_dump", "{:.3f}".format(tyexec.tvm_layerwise_dump_span)])
        table.add_row(["iss_layerwise_dump", "{:.3f}".format(tyexec.iss_layerwise_dump_span)])
        table.add_row(["total", "{:.3f}".format(time.time() - t_start)])
        logger.info("\n{}".format(table))

        # 计算相似度
        header = ["Idx", "Tensor-A", "Tensor-B", "Cosine similarity"]
        table = PrettyTable(header)
        for idx in range(len(tvm_fixed_output)):
            if not tyexec.is_qnn:
                dist = cosine_distance(tvm_float_output[idx], tvm_fixed_output[idx])
                logger.info("[Build] float(tvm) output tensor[{}] shape: {}, dtype: {}".format(idx, tvm_float_output[idx].shape, tvm_float_output[idx].dtype))
                table.add_row([idx, "float(tvm)", "fixed(tvm)", "{:.6f}".format(dist)])
            logger.info("[Build] fixed(tvm) output tensor[{}] shape: {}, dtype: {}".format(idx, tvm_fixed_output[idx].shape, tvm_fixed_output[idx].dtype))
            if iss_fixed_output:
                logger.info("[Build] fixed(iss) output tensor[{}] shape: {}, dtype: {}".format(idx, iss_fixed_output[idx].shape, iss_fixed_output[idx].dtype))
                if not tyexec.is_qnn:
                    dist = cosine_distance(tvm_float_output[idx], iss_fixed_output[idx])
                    table.add_row([idx, "float(tvm)", "fixed(iss)", "{:.6f}".format(dist)])
                dist = cosine_distance(tvm_fixed_output[idx], iss_fixed_output[idx])
                table.add_row([idx, "fixed(tvm)", "fixed(iss)", "{:.6f}".format(dist)])
        logger.info("\n{}".format(table))
        logger.info("success")
    except Exception as e:
        logger.error("Failed to build \n{}".format(traceback.format_exc()))
        exit(-1)


def compare(cfg, backend, device_id, node_id):
    try:
        if backend not in ["sdk_iss", "chip"]:
            logger.error("Compare phase only support iss and chip")
            exit(-1)
        logger.info("{}".format(cfg))
        tyexec = get_tyexec(cfg)
        tyexec.backend = backend
        if tyexec.target.startswith("nnp4"):
            fixed_outputs = tyexec.infer(device_id, node_id)  # default disable aipp
        elif tyexec.target.startswith("nnp3"):
            fixed_outputs = tyexec.infer()  # default disable aipp
        else:
            logger.error("Not support target -> {}".format(tyexec.target))
            exit(-1)

        # compare
        for idx, fixed_output in enumerate(fixed_outputs):
            if not tyexec.is_qnn:
                # tvm-float vs chip-fixed or iss-fixed
                tvm_float_out_path = os.path.join(tyexec.result_dir, "tvm_float_out_{}.bin".format(idx))
                if not os.path.exists(tvm_float_out_path):
                    logger.error("Not found tvm_float_out_path -> {}".format(tvm_float_out_path))
                    exit(-1)
                tvm_float_out = np.fromfile(tvm_float_out_path, dtype=fixed_output.dtype)
                dist = cosine_distance(fixed_output, tvm_float_out)
                logger.info("[Compare] fixed({}) vs float(tvm) output tensor[{}] similarity={:.6f}".format(tyexec.backend, idx, dist))

            # tvm-fixed vs chip-fixed or iss-fixed
            tvm_fixed_out_path = os.path.join(tyexec.result_dir, "tvm_fixed_out_{}.bin".format(idx))
            if not os.path.exists(tvm_fixed_out_path):
                logger.error("Not found tvm_fixed_out_path -> {}".format(tvm_fixed_out_path))
                exit(-1)
            tvm_fixed_out = np.fromfile(tvm_fixed_out_path, dtype=fixed_output.dtype)
            dist = cosine_distance(fixed_output, tvm_fixed_out)
            logger.info("[Compare] fixed({}) vs fixed(tvm) output tensor[{}] similarity={:.6f}".format(tyexec.backend, idx, dist))

            if tyexec.enable_dump:
                iss_fixed_out_path = os.path.join(tyexec.result_dir, "iss_fixed_out_{}.bin".format(idx))
                if not os.path.exists(iss_fixed_out_path):
                    logger.error("Not found iss_fixed_out_path -> {}".format(iss_fixed_out_path))
                    exit(-1)
                iss_fixed_out = np.fromfile(iss_fixed_out_path, dtype=fixed_output.dtype)
                dist = cosine_distance(fixed_output, iss_fixed_out)
                logger.info("[Compare] fixed({}) vs fixed(iss) output tensor[{}] similarity={:.6f}".format(tyexec.backend, idx, dist))
        logger.info("success")
    except Exception as e:
        logger.error("Failed to compare {}".format(traceback.format_exc()))


def compare2(cfg, target, data_path, device_id, node_id):
    """
    批量比较tvm-fp32、tvm-int8、iss、chip
    目前阻塞于iss太慢, tvm环境和sdk冲突
    """
    try:
        tyexec = get_tyexec(cfg)
        if target.startswith("nnp4"):
            if tyexec.is_qnn:
                logger.error("Not support qnn model")
                exit(-1)
            from src.nnp4xx_infer import Nnp4xxSdkInfer, Nnp4xxTvmInfer
            tvm_fp32_infer = Nnp4xxTvmInfer()
            tvm_int8_infer = Nnp4xxTvmInfer()
            tvm_fp32_infer.load(tyexec.cpu_model_float_path)
            tvm_int8_infer.load(tyexec.cpu_model_fixed_path)
        elif target.startswith("nnp3"):
            from src.nnp3xx_infer import Nnp3xxSdkInfer, Nnp3xxTvmInfer
            tvm_fp32_infer = Nnp3xxTvmInfer()
            tvm_int8_infer = Nnp3xxTvmInfer()
            tvm_fp32_infer.load_json(tyexec.original_json_path)
            tvm_int8_infer.load_json(tyexec.quant_json_path)
        else:
            logger.error("Not support target -> {}".format(target))
            exit(-1)

        from prettytable import PrettyTable
        table = None
        first = True

        file_list = os.listdir(data_path)
        for filename in file_list:
            img_path = os.path.join(data_path, filename)
            if not os.path.exists(img_path) or not os.path.isfile(img_path):
                continue
            float_datas = tyexec.get_datas(img_path, use_norm=True, force_cr=True, force_random=False, to_file=False)
            fixed_datas = tyexec.get_datas(img_path, use_norm=False, force_cr=True, force_random=False, to_file=False)

            float_outputs = tvm_fp32_infer.run(float_datas, to_file=False)
            fixed_outputs = tvm_int8_infer.run(fixed_datas, to_file=False)

            if first:
                out_list = ["Output{}".format(idx) for idx in range(len(float_outputs))]
                header = ["Image"]
                header.extend(out_list)
                table = PrettyTable(header)
                first = False
            line = [filename]
            for idx, (float_output, fixed_output) in enumerate(zip(float_outputs, fixed_outputs)):
                dist = cosine_distance(float_output, fixed_output)
                line.append("{:.6f}".format(dist))
            logger.info(" ".join(line))
            table.add_row(line)
        logger.info("TVM-FP32 vs TVM-INT8:\n{}".format(table))
    except Exception as e:
        logger.error("Failed to compare\n{}".format(traceback.format_exc()))
        exit(-1)
        

def compare_layerwise(cfg, target, data_path, device_id, node_id, backend):
    try:
        tyexec = get_tyexec(cfg)
        if target.startswith("nnp4"):
            tyexec.compare_layerwise(data_path, backend)
        else:
            logger.error("Not support target -> {}".format(target))
            exit(-1)
    except Exception as e:
        logger.error("Failed to compare_layerwise ->\n{}".format(traceback.format_exc()))
        exit(-1)


def profile(cfg, device_id, node_id):
    try:
        logger.info("{}".format(cfg))
        tyexec = get_tyexec(cfg)
        if tyexec.target.startswith("nnp4"):
            tyexec.profile(device_id, node_id)
        elif tyexec.target.startswith("nnp3"):
            tyexec.profile()
        else:
            logger.error("Not support target -> {}".format(tyexec.target))
            exit(-1)
        logger.info("success")
    except Exception as e:
        logger.error("TyAssist failed to profile -> {}".format(traceback.format_exc()))


def test(cfg, dtype, backend, device_id, node_id):
    try:
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
        enable_aipp = cfg["test"].get("enable_aipp", False)

        m = importlib.import_module(dataset_module)
        if hasattr(m, dataset_cls):
            # 实例化预处理对象
            dataset = getattr(m, dataset_cls)(data_dir)
        else:
            logger.error("{}.py has no class named {}".format(dataset_module, dataset_cls))
            exit(-1)

        model_impl_module = cfg["model"]["model_impl_module"]
        model_impl_cls = cfg["model"]["model_impl_cls"]

        m = importlib.import_module(model_impl_module)
        if hasattr(m, model_impl_cls):
            # 实例化预处理对象
            model = getattr(m, model_impl_cls)(
                inputs=tyexec.inputs,
                dataset=dataset,
                test_num=test_num,
                enable_aipp=enable_aipp,
                target=tyexec.target,  # nnp3xx/nnp4xx
                dtype=dtype,   # int8/fp32
                backend=backend,  # tvm/iss/chip
                device_id=device_id,  # 目前仅用于4xx
                node_id=node_id,  # 目前仅用于4xx
            )
        else:
            logger.error("{}.py has no class named {}".format(model_impl_module, model_impl_cls))
            exit(-1)

        if backend == "tvm":
            if tyexec.target.startswith("nnp4"):
                if dtype == "fp32":
                    model.load(tyexec.cpu_model_float_path)
                elif dtype == "int8":
                    model.load(tyexec.cpu_model_fixed_path)
                else:
                    logger.error("Not support dtype -> {}".format(dtype))
                    exit(-1)
            elif tyexec.target.startswith("nnp3"):
                if dtype == "fp32":
                    model.load_json(tyexec.original_json_path)
                elif dtype == "int8":
                    model.load_json(tyexec.quant_json_path)
                else:
                    logger.error("Not support dtype -> {}".format(dtype))
                    exit(-1)
        elif backend == "onnx":
            model.load(tyexec.weight)
        else:  # chip/iss
            if tyexec.target.startswith("nnp4"):
                model_path = tyexec.model_path_x86_64 if backend == "sdk_iss" else tyexec.model_path_aarch64
            elif tyexec.target.startswith("nnp3"):
                model_path = tyexec.model_path
            else:
                logger.error("Not support target({})".format(tyexec.target))
                exit(-1)
            model.load(model_path)

        res = model.evaluate()
        del sys.modules[dataset_module]
        del sys.modules[model_impl_module]

        logger.info("average cost {:.6f}ms".format(model.ave_latency_ms))
        logger.info("[end2end] average cost: {:.6f}ms".format(model.end2end_latency_ms))
        logger.info("{}".format(res))
        logger.info("success")
        return res
    except Exception as e:
        logger.error("Failed to test \n{}".format(traceback.format_exc()))
        exit(-1)


def demo(cfg, dtype, backend, device_id, node_id):
    try:
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
        # dataset_module = cfg["test"]["dataset_module"]
        # dataset_cls = cfg["test"]["dataset_cls"]
        model_impl_module = cfg["model"]["model_impl_module"]
        model_impl_cls = cfg["model"]["model_impl_cls"]
        enable_aipp = cfg["demo"].get("enable_aipp", False)
        bs = cfg["model"]["inputs"][0]["shape"][0]

        file_list = os.listdir(data_dir)
        file_list = file_list if num > len(file_list) else file_list[0:num]

        # m = importlib.import_module(dataset_module)
        # if hasattr(m, dataset_cls):
        #     # 实例化预处理对象
        #     dataset = getattr(m, dataset_cls)(data_dir)
        # else:
        #     logger.error("{}.py has no class named {}".format(dataset_module, dataset_cls))
        #     exit(-1)

        m = importlib.import_module(model_impl_module)
        if hasattr(m, model_impl_cls):
            # 实例化预处理对象
            model = getattr(m, model_impl_cls)(
                inputs=tyexec.inputs,
                dataset=None,
                test_num=0,
                enable_aipp=enable_aipp,
                target=tyexec.target,  # nnp3xx/nnp4xx
                dtype=dtype,   # int8/fp32
                backend=backend,  # tvm/iss/chip
                device_id=device_id,  # 目前仅用于4xx
                node_id=node_id,  # 目前仅用于4xx
            )
        else:
            logger.error("{}.py has no class named {}".format(model_impl_module, model_impl_cls))
            exit(-1)

        if backend == "tvm":
            if tyexec.target.startswith("nnp4"):
                if dtype == "fp32":
                    model.load(tyexec.cpu_model_float_path)
                elif dtype == "int8":
                    model.load(tyexec.cpu_model_fixed_path)
                else:
                    logger.error("Not support dtype -> {}".format(dtype))
                    exit(-1)
            elif tyexec.target.startswith("nnp3"):
                if dtype == "fp32":
                    model.load_json(tyexec.original_json_path)
                elif dtype == "int8":
                    model.load_json(tyexec.quant_json_path)
                else:
                    logger.error("Not support dtype -> {}".format(dtype))
                    exit(-1)
        elif backend == "onnx":
            model.load(tyexec.weight)
        else:  # chip/iss
            if tyexec.target.startswith("nnp4"):
                model_path = tyexec.model_path_x86_64 if backend == "sdk_iss" else tyexec.model_path_aarch64
            elif tyexec.target.startswith("nnp3"):
                model_path = tyexec.model_path
            else:
                logger.error("Not support target({})".format(tyexec.target))
                exit(-1)
            model.load(model_path)

        filepaths = list()
        for idx, img_name in enumerate(file_list):
            _, ext = os.path.splitext(img_name)
            if ext not in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
                logger.warning("file ext invalid -> {}".format(img_name))
                continue

            filepath = os.path.join(data_dir, img_name)
            if not os.path.exists(filepath):
                continue

            filepaths.append(filepath)

            if (idx + 1) % bs == 0:
                model.demo(filepaths)
                filepaths.clear()
        if len(filepaths) > 0:
            model.demo(filepaths)
        logger.info("average cost {:.6f}ms".format(model.ave_latency_ms))
        logger.info("[end2end] average cost: {:.6f}ms".format(model.end2end_latency_ms))
        logger.info("success")
    except Exception as e:
        logger.error("Failed to demo\n{}".format(traceback.format_exc()))
        exit(-1)


def run(config_filepath, phase, dtype, target, backend, data_path, enable_layers, opt_level, device_id, node_id):
    # 补充自定义预处理文件所在目录，必须与配置文件同目录
    config_abspath = os.path.abspath(config_filepath)
    config_dir = os.path.dirname(config_abspath)
    sys.path.insert(0, config_dir)  # 自定义模块环境变量

    config = read_yaml_to_dict(config_abspath)
    # 更新target，优先使用命令行
    if target is not None:
        config["build"]["target"] = target
    target = config["build"]["target"]
    
    # 更新4xx编译优化等级
    if opt_level is not None:
        config["build"]["opt_level"] = opt_level
    # 检查配置
    if not check_config(config, phase):
        exit(-1)
        
    # update
    if backend == "iss":
        backend = "sdk_iss"

    res = dict()
    if phase == "build":
        build(config)
    elif phase == "compare":
        if data_path:
            if os.path.isfile(data_path) and enable_layers:
                compare_layerwise(config, target, data_path, device_id, node_id, backend)
            elif os.path.isdir(data_path):
                compare2(config, target, data_path, device_id, node_id)
            else:
                logger.error("\n1. compare with layers, data_path should be a filepath\n2. compare2, data_path should be a dir")
        else:
            compare(config, backend, device_id, node_id)
    elif phase == "test":
        res = test(config, dtype, backend, device_id, node_id)
    elif phase == "demo":
        demo(config, dtype, backend, device_id, node_id)
    elif phase == "profile":
        profile(config, device_id, node_id)

    sys.path.remove(config_dir)
    return res


def benchmark(mapping_file, dtype, target, backend, version, opt_level, device_id, node_id):
    import csv
    from prettytable import from_csv

    header = ["ModelName", "InputSize", "Dataset", "Num", "Acc./mAP.", "Latency(ms)"]
    csv_filepath = "benchmark_{}_{}_{}_{}.csv".format(backend, dtype, target, version)
    is_exist = os.path.exists(csv_filepath)
    f = open(csv_filepath, "a+")
    f_csv = csv.writer(f)
    if not is_exist:
        f_csv.writerow(header)

    check_file_exist(mapping_file)
    models_dict = read_yaml_to_dict(mapping_file)["models"]
    root = os.getcwd()
    for model_name in models_dict:
        logger.info("Process {}".format(model_name))
        config_filepath = models_dict[model_name]
        config_abspath = os.path.abspath(config_filepath)
        config_dir = os.path.dirname(config_abspath)
        # 判断是否已存在模型
        model_cfg = read_yaml_to_dict(config_abspath)
        save_path = os.path.abspath(model_cfg["model"]["save_dir"])
        if not os.path.join(save_path, model_cfg["model"].get(
                "name", "net_combine.ty" if target.startswith("nnp3") else "net_combine_aarch64.ty")):
            logger.warning("Model not found -> {}".format(save_path))
            continue

        os.chdir(config_dir)  # 切换至模型目录
        res = run(config_abspath, "test", dtype, target, backend, None, opt_level, device_id, node_id)
        # logger.info("{}".format(res))
        os.chdir(root)  # 切换根目录

        row = list()
        if "top1" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["top1"], res["top5"]), res["latency"]]
        elif "map" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}".format(res["map"], res["map50"]), res["latency"]]
        elif "easy" in res:
            row = [model_name, res["input_size"], res["dataset"], res["num"], "{}/{}/{}".format(res["easy"], res["medium"], res["hard"]), res["latency"]]
        f_csv.writerow(row)
        logger.info("Finish {}".format(model_name))
    f.close()
    fp = open(csv_filepath, "r")
    table = from_csv(fp)
    fp.close()
    logger.info("\n{}".format(table))
    logger.info("success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TyAssist Tool")
    parser.add_argument("type", type=str, choices=("demo", "test", "compare", "benchmark", "build", "profile"),
                        help="Please specify a operator")
    parser.add_argument("--config", "-c", type=str, required=True if "--onnx" not in sys.argv else False,
                        help="Please specify a configuration file")
    parser.add_argument("--target", type=str, required=False,
                        choices=("nnp300", "nnp3020", "nnp310", "nnp315m", "nnp320", "nnp400"),
                        help="Please specify a chip target")
    parser.add_argument("--dtype", "-t", type=str, default="int8", choices=("int8", "fp32"),
                        help="Please specify one of them, default int8")
    parser.add_argument("--opt_level", type=int, required=False, choices=(0, 2),
                        help="Please specify one of them, default 0, only for nnp4xx!")
    parser.add_argument("--data_path", type=str,
                        help="Please specify a dir or path, required only comapre specify images")
    parser.add_argument("--layers", action="store_true", help="Debug layer by layer, required only comapre")
    parser.add_argument("--backend", type=str, required="demo" in sys.argv or "test" in sys.argv or "compare" in sys.argv,
                        choices=("chip", "iss", "tvm", "onnx") if "compare" not in sys.argv else ("chip", "iss"),
                        help="Please specify one of them")
    parser.add_argument("--device_id", type=int, default=0, help="specify a device, default is 0")
    parser.add_argument("--node_id", type=int, default=0, help="specify a Die, default is 0")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Please specify a log dir, default is ./logs")
    parser.add_argument("--log_level", type=int, required=False, default=2, choices=(1, 2, 3, 4, 5),
                        help="Please specify a log level, default 2, 1:DEBUG, 2:INFO, 3:WARNING, 4:ERROR, 5:FATAL")
    parser.add_argument("--version", type=str, required=("benchmark" in sys.argv),
                        help="Please specify a tytvm version, required only onnx backend")
    parser.add_argument("--onnx", type=str, required="--config" not in sys.argv and "-c" not in sys.argv,
                        help="Please specify a onnx file")
    
    args = parser.parse_args()
    logger.setLevel(args.log_level*10)

    dirname, filename = os.path.split(os.path.abspath(__file__))
    version_path = os.path.join(dirname, "version")
    if not os.path.exists(version_path):
        logger.warning("Not found version file")
    with open(version_path, "rb") as f:
        VERSION = f.readline().decode("gbk").strip()
        logger.info("{} with TyAssist version: {}".format(args.type, VERSION))
      
    if "--onnx" not in sys.argv:
        check_file_exist(args.config)
        basename, _ = os.path.splitext(os.path.basename(args.config))
        set_logger(args.type, args.log_dir, basename)
    elif "--onnx" in sys.argv and ("--config" not in sys.argv and "-c" not in sys.argv):
        # 简便方式快速编译，方便评估编译测试性能
        check_file_exist(args.onnx)
        basename, ext = os.path.splitext(os.path.basename(args.onnx))
        if ext != ".onnx":
            logger.error("Only support onnx model")
            exit(-1)
        set_logger(args.type, args.log_dir, basename)
        config = gen_default_config(args.onnx)
        build(config)
        exit(0)

    # check
    if args.backend == "tvm":
        pass
    elif args.backend == "onnx":
        if args.dtype != "fp32":
            logger.warning("Onnx only support fp32")
        args.dtype = "fp32"
    elif args.backend == "iss":
        if args.dtype not in ["int8"]:
            logger.error("iss not support dtype({})".format(args.dtype))
            exit(-1)
    elif args.backend == "chip":
        if args.dtype not in ["int8"]:
            logger.error("chip not support dtype({})".format(args.dtype))
            exit(-1)

    if args.type == "benchmark":
        benchmark(args.config, args.dtype, args.target, args.backend, args.version, args.opt_level, args.device_id, args.node_id)
    else:
        _ = run(args.config, args.type, args.dtype, 
                args.target, args.backend, args.data_path, args.layers, args.opt_level, args.device_id, args.node_id)
