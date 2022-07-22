#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : infer.py
@Time    : 2022/7/12 下午1:55
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import os
import sys
sys.path.append("/DEngine/tyassist2")
import time
import shutil
from utils import logger
from utils.utils import get_host_ip
from utils.compare import compare_dump_out

import tvm
import tvm.rpc
import dcl
from dcl.desdk import DumpProfileSel


class Infer(object):
    def __init__(self, net_cfg_file="/DEngine/tyhcp/net.cfg", sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
                 ip="127.0.0.1", port=9090, enable_dump=False, max_batch=1):
        """
        :param net_cfg_file:
        :param sdk_cfg_file:
        :param ip:
        :param port:
        :param enable_dump:
        """
        self._ip = ip
        self._port = port
        self._net_cfg_file = net_cfg_file
        self._sdk_cfg_file = sdk_cfg_file
        self._enable_dump = enable_dump
        self._engine = None
        self._sdk = None
        self._max_batch = max_batch
        self._model_dir = ""
        self._result_dir = ""
        self._dump_root_path = ""
        self._ave_latency_ms = 0

        assert max_batch == 1, "Not support max_batch > 1 yet."
        assert os.path.exists(self._net_cfg_file), "Not found net_cfg_file -> {}".format(self._net_cfg_file)
        assert os.path.exists(self._sdk_cfg_file), "Not found sdk_cfg_file -> {}".format(self._sdk_cfg_file)

        f = open(self._net_cfg_file, "r")
        net_cfg = f.read().strip()
        f.close()
        self._ip = net_cfg.split(":")[0]

        logger.info("Try to connect to {}:{}".format(self._ip, self._port))
        remote = tvm.rpc.connect(self._ip, self._port)
        logger.info("connection succeed.")

        self._sdk = dcl.DeSDKModule(remote)
        logger.info("tyhcp version: {}".format(self._sdk.version))

        if enable_dump:
            self._sdk.select_dump_profile(DumpProfileSel.Dump)
            self._sdk.set_dump_server_ip(get_host_ip(), self._port)

        self._sdk.sdk_init(self._sdk_cfg_file)
        logger.info("tyhcp init succeed.")

    def load(self, model_dir):
        """加载芯片模型
        :param model_dir:
        :param max_batch:
        :return:
        """
        self._model_dir = model_dir
        self._result_dir = os.path.join(self._model_dir, "result")
        if not os.path.exists(self._result_dir):
            logger.error("Not found result_dir -> {}".format(self._result_dir))
            exit(-1)
        self._result_dir = os.path.abspath(self._result_dir)
        if self._enable_dump:
            self._dump_root_path = self._sdk.set_dump_work_path(self._result_dir)
            logger.info("dump root path: {}".format(self._dump_root_path))

        netbin_file = os.path.join(self._model_dir, "net_combine.bin")
        logger.info("load model " + netbin_file)

        # iss上目前不支持aipp，需要去使能
        self._sdk.disable_aipp()
        self._engine = self._sdk.create_model(self._max_batch)
        self._engine.load_model(netbin_file)

    def run(self, in_datas: list, to_file=False):
        """推理
        :param in_datas: list表示多输入
        :param to_file: 表示是否将结果输出至文件
        :return:
        """
        t_start = time.time()
        for idx, in_data in enumerate(in_datas):
            logger.info("inputs[{}], shape={}, dtype={}".format(idx, in_data.shape, in_data.dtype))
            self._engine.set_input(0, idx, in_data.copy())

        logger.info("model is running...")
        self._engine.run()
        self._engine.clear_input_data(0, 0)  # 第二个参数未使用

        outputs = list()
        for idx in range(self._engine.get_num_outputs()):
            outputs.append(self._engine.get_output(0, idx).numpy())

        cost = time.time() - t_start
        logger.info("[runonchip] predict cost: {:.3f}ms".format(cost * 1000))

        if self._enable_dump:
            self._compare_dump_out()

        # dump输出
        if to_file:
            logger.info("[runonchip] predict result: outputs size -> {}".format(self._engine.get_num_outputs()))
            for idx, output in enumerate(outputs):
                logger.info("outputs[{}], shape={}, dtype={}".format(idx, output.shape, output.dtype))
                filepath_txt = os.path.join(self._result_dir, "chip_fixed_out_{}.txt".format(idx))
                filepath_bin = os.path.join(self._result_dir, "chip_fixed_out_{}.bin".format(idx))
                output.tofile(filepath_txt, sep="\n")
                output.tofile(filepath_bin)
        return outputs

    def _compare_dump_out(self):
        # model_name = os.path.basename(model_path)
        filename_list = os.listdir(self._dump_root_path)
        if len(filename_list) != 1:
            logger.error("dump_root_path file num must be == 1")
            exit(-1)
        model_name = filename_list[0]

        src = os.path.join(self._dump_root_path, model_name)
        if not os.path.exists(src):
            logger.error("Not found model dump path -> {}".format(src))
            exit(-1)

        chip_dump_out = os.path.join(self._result_dir, "chip_dump_out")
        if os.path.exists(chip_dump_out):
            shutil.rmtree(chip_dump_out)
        logger.info("cp {} -> {}".format(src, chip_dump_out))
        shutil.copytree(src, chip_dump_out)

        iss_dump_out = os.path.join(self._result_dir, "host_iss_fused_out.pickle")
        if not os.path.join(iss_dump_out):
            logger.error("Not found iss_dump_out -> {}".format(iss_dump_out))
            exit(-1)
        compare_dump_out(chip_dump_out, iss_dump_out)

    def __del__(self):
        if self._engine:
            self._engine.unload_model()
        if self._sdk:
            self._sdk.sdk_finalize()


if __name__ == "__main__":
    infer = Infer()

