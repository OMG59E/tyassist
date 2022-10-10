#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : compare.py
@Time    : 2022/7/13 下午3:12
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import os
import pickle
import json
import numpy as np
from utils.dist_metrics import cosine_distance


def compare_dump_out(chip_dump_path, iss_fixed_dump_path):
    """
    逐层比较chip和iss输出结果，chip输出的结果将layername中存在"/"替换为"_"
    :param chip_dump_path:  芯片输出结果目录，是目录
    :param iss_fixed_dump_path:  ISS序列输出pickle路径，是文件
    :return:
    """
    assert os.path.isdir(chip_dump_path)
    assert os.path.isfile(iss_fixed_dump_path)

    f = open(iss_fixed_dump_path, "rb")
    iss_fixed_dump_out = pickle.load(f)
    f.close()

    descript_filepath = os.path.join(chip_dump_path, "descript")
    assert os.path.isfile(descript_filepath)
    f = open(descript_filepath, "r")
    descript = json.loads(f.read())
    f.close()

    op_infos = descript["blockDescOutLoop"][0]["blockDescInLoop"][0]["layerDesc"][0]["opList"]

    import csv
    from prettytable import PrettyTable

    header = ["Id", "OpName", "Dtype", "Shape", "Chip(fixed) vs ISS(fixed)"]
    table = PrettyTable(header)
    csv_filepath = os.path.join(chip_dump_path, "similarity.csv")
    f = open(csv_filepath, "w")
    f_csv = csv.writer(f)
    f_csv.writerow(header)
    for idx, op_info in enumerate(op_infos):
        op_name = op_info["opName"]
        iss_fixed_out = iss_fixed_dump_out[op_name]
        shape = iss_fixed_out.shape
        dtype = iss_fixed_out.dtype
        chip_op_name = str(op_name).replace("/", "_")
        chip_fixed_out_path = os.path.join(chip_dump_path, "0", "{}_out0.bin".format(chip_op_name))
        assert os.path.isfile(chip_fixed_out_path), "chip_fixed_out_path -> {}".format(chip_fixed_out_path)
        chip_fixed_out = np.fromfile(chip_fixed_out_path, dtype=iss_fixed_out.dtype)
        chip_fixed_vs_iss_fixed_dist = "{:.6f}".format(cosine_distance(iss_fixed_out, chip_fixed_out))
        result = [idx, op_name, dtype, shape, chip_fixed_vs_iss_fixed_dist]
        table.add_row(result)
        f_csv.writerow(result)
    f.close()
    print(table, flush=True)


def compare_dump_out2(tvm_fixed_dump_path, tvm_fp32_dump_path):
    assert os.path.isfile(tvm_fixed_dump_path)
    assert os.path.isfile(tvm_fp32_dump_path)

    f = open(tvm_fixed_dump_path, "rb")
    tvm_fixed_dump_out = pickle.load(f)
    f.close()

    f = open(tvm_fp32_dump_path, "rb")
    tvm_fp32_dump_out = pickle.load(f)
    f.close()

    from prettytable import PrettyTable

    header = ["Id", "OpName", "TVM(fixed) vs TVM(float)"]
    table = PrettyTable(header)
    for idx, op_name in enumerate(tvm_fixed_dump_out):
        assert op_name in tvm_fp32_dump_out
        tvm_fp32_out = tvm_fp32_dump_out[op_name]
        tvm_fixed_out = tvm_fixed_dump_out[op_name]
        tvm_fixed_vs_tvm_fp32_dist = "{:.6f}".format(cosine_distance(tvm_fp32_out, tvm_fixed_out))
        result = [idx, op_name, tvm_fixed_vs_tvm_fp32_dist]
        table.add_row(result)
    print(table, flush=True)

