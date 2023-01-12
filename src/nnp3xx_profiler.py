#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp3xx_profiler.py 
@time: 2022/12/22
@Author  : xingwg
@software: PyCharm 
"""
import os
import abc
import json
from .base_profiler import BaseSdkProfiler
from utils import logger


class Nnp3xxSdkProfiler(BaseSdkProfiler, abc.ABC):
    def __init__(self, net_cfg_file="/DEngine/tyhcp/net.cfg", sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
                 target="nnp300"):
        super(Nnp3xxSdkProfiler, self).__init__(net_cfg_file, sdk_cfg_file, target)

        with open(net_cfg_file, "r") as f:
            net_cfg = f.read().strip()
            self.ip = net_cfg.split(":")[0]

        if self.ip == "127.0.0.1":   # TODO 非127.0.0.1的地址也可能是ISS服务
            logger.error("ISS mode not support profile")
            exit(-1)

    def load(self, model_path):
        try:
            import tvm
            import tvm.rpc
            import dcl
            from dcl.desdk import DumpProfileSel

            logger.info("Try to connect to {}:{}".format(self.ip, self.port))
            remote = tvm.rpc.connect(self.ip, self.port)
            logger.info("connection succeed.")

            self.sdk = dcl.DeSDKModule(remote)
            logger.info("tyhcp version: {}".format(self.sdk.version))
            dump_server_ip = os.getenv("DUMP_SERVER_IP")
            dump_server_port = os.getenv("DUMP_SERVER_PORT")
            self.sdk.set_dump_server_ip(dump_server_ip, int(dump_server_port))
            self.sdk.select_dump_profile(DumpProfileSel.ProfileFile)

            logger.info("sdk config path: {}".format(self.sdk_cfg_file))
            self.sdk.sdk_init(self.sdk_cfg_file)
            logger.info("tyhcp init succeed.")
        except Exception as e:
            logger.error("Import failed -> {}".format(e))
            exit(-1)

        self.result_dir = os.path.join(os.path.dirname(model_path), "result")
        if not os.path.exists(self.result_dir):
            logger.error("Not found result_dir -> {}".format(self.result_dir))
            exit(-1)

        logger.info("sdk config: {}".format(self.sdk.get_sdk_config()))

        if not os.path.isfile(model_path):
            logger.error("netbin_file not file -> {}".format(model_path))
            exit(-1)

        logger.info("load model " + model_path)

        max_batch = 1
        self.sdk.disable_aipp()
        self.engine = self.sdk.create_model(max_batch)
        self.engine.load_model(model_path)
        logger.info(self.engine.get_model_info())

    def run(self, in_datas: list, to_file=False):
        for idx, in_data in enumerate(in_datas):
            self.engine.set_input(0, idx, in_data.copy())

        self.engine.run()
        self.engine.clear_input_data(0, 0)

        outputs = list()
        for idx in range(self.engine.get_num_outputs()):
            outputs.append(self.engine.get_output(0, idx).numpy())
        return outputs

    def save_profile(self):
        filepath = os.path.join(self.result_dir, "profile_result.tar.gz")
        self.sdk.copy_profile_file_to(filepath)
        if os.path.exists(filepath):
            os.system("tar -xvf {} -C {}".format(filepath, self.result_dir))
            logger.info("save profile to {}".format(filepath))
        else:
            logger.error("Failed to save profile to {}".format(filepath))
            exit(-1)

    def unload(self):
        if self.engine:
            self.engine.unload_model()
            logger.info("unload model")
        if self.sdk:
            self.sdk.sdk_finalize()

    def __del__(self):
        self.unload()

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
        model_drv_block_lines = self._merge(os.path.join(self.result_dir, "MODEL_DRV.block.slice.0"))
        ge_model_desc_lines = self._merge(os.path.join(self.result_dir, "GE.model_desc.slice.0"))
        dcl_model_slice_lines = self._merge(os.path.join(self.result_dir, "DCL.dcl_model.slice.0"))

        model_desc = json.loads("".join(ge_model_desc_lines))
        op_desc_lists = model_desc["blockDescOutLoop"][0]["blockDescInLoop"][0]["layerDesc"][0]["opList"]

        # TODO multi-iter
        from prettytable import PrettyTable
        header = ["Id", "OpName", "MAC.", "DDR/R", "DDR/W", "Cycles", "Span/ms"]
        table = PrettyTable(header)
        for n in range(0, len(model_drv_block_lines), 2):
            first = model_drv_block_lines[n].strip().split()
            second = model_drv_block_lines[n+1].strip().split()
            assert int(first[3]) == int(second[3])
            iter_id = int(first[3])
            hw_cycles = int(first[-3])
            span = int(first[-1])
            logger.info("iter_id[{}] hw_cycles: {}, hw_span: {:.3f}ms, span: {:.3f}ms".format(
                iter_id, hw_cycles, hw_cycles * 2.0 * 10**-3 / self.targets[self.target], span * 10**-6))
            num_ops = int(second[7])
            for idx in range(num_ops):
                op_name = op_desc_lists[idx]["opName"]
                mac_num = op_desc_lists[idx]["macNum"]
                ddr_r = op_desc_lists[idx]["ddrRd"]
                ddr_w = op_desc_lists[idx]["ddrWt"]
                hw_cycles = int(second[9 + idx])
                hw_span = hw_cycles * 2.0 * 10**-3 / 792
                table.add_row([idx, op_name, mac_num, ddr_r, ddr_w, hw_cycles, "{:.3f}".format(hw_span)])
        logger.info("\n{}".format(table))
