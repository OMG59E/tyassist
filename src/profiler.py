#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: profiler.py 
@time: 2022/11/23
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: CLion 
"""
import os
import json
from utils import logger


class SdkProfiler(object):
    def __init__(self, net_cfg_file="/DEngine/tyhcp/net.cfg",
                 sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg", max_batch=1):
        self._ip = "127.0.0.1"
        self._port = 9090
        self._net_cfg_file = net_cfg_file
        self._sdk_cfg_file = sdk_cfg_file
        self._engine = None
        self._sdk = None
        self._max_batch = max_batch
        self._model_dir = ""
        self._result_dir = ""
        self._ave_latency_ms = 0
        self._total = 0

        if max_batch != 1:
            logger.error("Not support max_batch > 1 yet.")
            exit(-1)

        if not os.path.exists(self._net_cfg_file):
            logger.error("Not found net_cfg_file -> {}".format(self._net_cfg_file))
            exit(-1)

        if not os.path.exists(self._sdk_cfg_file):
            logger.error("Not found sdk_cfg_file -> {}".format(self._sdk_cfg_file))
            exit(-1)

        with open(self._net_cfg_file, "r") as f:
            net_cfg = f.read().strip()
            self._ip = net_cfg.split(":")[0]

        if self._ip == "127.0.0.1":   # TODO 非127.0.0.1的地址也可能是ISS服务
            logger.error("ISS mode not support profile")
            exit(-1)

        try:
            import tvm
            import tvm.rpc
            import dcl
            from dcl.desdk import DumpProfileSel

            logger.info("Try to connect to {}:{}".format(self._ip, self._port))
            remote = tvm.rpc.connect(self._ip, self._port)
            logger.info("connection succeed.")

            self._sdk = dcl.DeSDKModule(remote)
            logger.info("tyhcp version: {}".format(self._sdk.version))
            dump_server_ip = os.getenv("DUMP_SERVER_IP")
            dump_server_port = os.getenv("DUMP_SERVER_PORT")
            self._sdk.set_dump_server_ip(dump_server_ip, int(dump_server_port))
            self._sdk.select_dump_profile(DumpProfileSel.ProfileFile)

            logger.info("sdk config path: {}".format(sdk_cfg_file))
            self._sdk.sdk_init(self._sdk_cfg_file)
            logger.info("tyhcp init succeed.")
        except Exception as e:
            logger.error("Import failed -> {}".format(e))
            exit(-1)

    def load(self, model_dir):
        self._model_dir = model_dir
        self._result_dir = os.path.join(self._model_dir, "result")
        if not os.path.exists(self._result_dir):
            logger.error("Not found result_dir -> {}".format(self._result_dir))
            exit(-1)
        self._result_dir = os.path.abspath(self._result_dir)
        logger.info("sdk config: {}".format(self._sdk.get_sdk_config()))

        netbin_file = os.path.join(self._model_dir, "net_combine.bin")
        if not os.path.isfile(netbin_file):
            logger.error("netbin_file not file -> {}".format(netbin_file))
            exit(-1)

        logger.info("load model " + netbin_file)

        self._sdk.disable_aipp()
        self._engine = self._sdk.create_model(self._max_batch)
        self._engine.load_model(netbin_file)
        logger.info(self._engine.get_model_info())

    def run(self, in_datas: list, to_file=False):

        for idx, in_data in enumerate(in_datas):
            self._engine.set_input(0, idx, in_data.copy())

        self._engine.run()
        self._engine.clear_input_data(0, 0)

        outputs = list()
        for idx in range(self._engine.get_num_outputs()):
            outputs.append(self._engine.get_output(0, idx).numpy())
        return outputs

    def unload(self):
        if self._engine:
            self._engine.unload_model()
            logger.info("unload model")
        if self._sdk:
            self._sdk.sdk_finalize()

    def save_profile(self):
        filepath = os.path.join(self._result_dir, "profile_result.tar.gz")
        self._sdk.copy_profile_file_to(filepath)
        logger.info("save profile to {}".format(filepath))
        if os.path.exists(filepath):
            os.system("tar -xvf {} -C {}".format(filepath, self._result_dir))

    @staticmethod
    def _merge(filepath: str):
        if not os.path.exists(filepath):
            logger.error("Not found file -> {}".format(filepath))
            exit(-1)
        idx = int(filepath.split(".")[-1])
        f = open(filepath, "r")
        lines = f.readlines()
        f.close()
        idx += 1
        filepath = filepath[:-1] + str(idx)
        while os.path.exists(filepath):
            f = open(filepath, "r")
            lines.extend(f.readlines())
            f.close()
            idx += 1
            filepath = filepath[:-1] + str(idx)
        return lines

    def parse(self):
        model_drv_block_lines = self._merge(os.path.join(self._result_dir, "MODEL_DRV.block.slice.0"))
        ge_model_desc_lines = self._merge(os.path.join(self._result_dir, "GE.model_desc.slice.0"))
        dcl_model_slice_lines = self._merge(os.path.join(self._result_dir, "DCL.dcl_model.slice.0"))

        model_desc = json.loads("".join(ge_model_desc_lines))
        op_desc_lists = model_desc["blockDescOutLoop"][0]["blockDescInLoop"][0]["layerDesc"][0]["opList"]

        # TODO multi-iter
        from prettytable import PrettyTable
        header = ["Id", "OpName", "Cycles", "Span/ms"]
        table = PrettyTable(header)
        for n in range(0, len(model_drv_block_lines), 2):
            first = model_drv_block_lines[n].strip().split()
            second = model_drv_block_lines[n+1].strip().split()
            assert int(first[3]) == int(second[3])
            iter_id = int(first[3])
            hw_cycles = int(first[-3])
            span = int(first[-1])
            logger.info("iter_id[{}] hw_cycles: {}, hw_span: {:.3f}ms, span: {:.3f}ms".format(
                iter_id, hw_cycles, hw_cycles * 2.0 * 10**-3 / 792, span * 10**-6))
            num_ops = int(second[7])
            for idx in range(num_ops):
                op_name = op_desc_lists[idx]["opName"]
                hw_cycles = int(second[9 + idx])
                hw_span = hw_cycles * 2.0 * 10**-3 / 792
                table.add_row([idx, op_name, hw_cycles, "{:.3f}".format(hw_span)])
        logger.info("\n{}".format(table))

