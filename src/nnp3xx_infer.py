#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp3xx_infer.py 
@time: 2022/12/21
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import os
import shutil
from abc import ABC
from utils import logger
from utils.compare import compare_dump_out, compare_dump_out2
from .base_infer import BaseSdkInfer


class Nnp3xxSdkInfer(BaseSdkInfer, ABC):
    def __init__(
            self,
            net_cfg_file="/DEngine/tyhcp/net.cfg",
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            enable_dump=0,
            enable_aipp=False,
    ):
        super(Nnp3xxSdkInfer, self).__init__(
            net_cfg_file=net_cfg_file,
            sdk_cfg_file=sdk_cfg_file,
            enable_dump=enable_dump,
            enable_aipp=enable_aipp,
        )

        if not os.path.exists(net_cfg_file):
            logger.error("Not found net_cfg_file -> {}".format(net_cfg_file))
            exit(-1)

        self.ip = "127.0.0.1"
        self.port = 9090
        with open(net_cfg_file, "r") as f:
            net_cfg = f.read().strip()
            self.ip = net_cfg.split(":")[0]

        if self.ip == "127.0.0.1":   # TODO 非127.0.0.1的地址也可能是ISS服务
            self.prefix = "sdk_iss"

    def load(self, model_path):
        if not os.path.exists(model_path):
            logger.error("Not found net_cfg_file -> {}".format(model_path))
            exit(-1)
        self.result_dir = os.path.join(os.path.dirname(model_path), "result")

        if self.enable_dump:
            logger.warning("ISS mode not support dump server, will be disable")

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

            if self.enable_dump == 1 and self.prefix != "sdk_iss":
                self.sdk.select_dump_profile(DumpProfileSel.Dump)
                dump_server_ip = os.getenv("DUMP_SERVER_IP")
                dump_server_port = os.getenv("DUMP_SERVER_PORT")
                self.sdk.set_dump_server_ip(dump_server_ip, int(dump_server_port))
            else:
                self.sdk.select_dump_profile(DumpProfileSel.Profile)

            logger.info("sdk config path: {}".format(self.sdk_cfg_file))
            self.sdk.sdk_init(self.sdk_cfg_file)
            logger.info("tyhcp init succeed.")
        except Exception as e:
            logger.error("load failed -> {}".format(e))
            exit(-1)

        if self.enable_dump and self.prefix != "sdk_iss":
            if not os.path.exists(self.result_dir):
                logger.error("Not found result_dir -> {}".format(self.result_dir))
                exit(-1)
            self.dump_root_path = self.sdk.set_dump_work_path(os.path.abspath(self.result_dir))
            logger.info("dump root path: {}".format(self.dump_root_path))
        logger.info("sdk config: {}".format(self.sdk.get_sdk_config()))

        logger.info("load model " + model_path)

        if not self.enable_aipp:
            self.sdk.disable_aipp()

        max_batch = 1
        self.engine = self.sdk.create_model(max_batch)
        self.engine.load_model(model_path)
        logger.info(self.engine.get_model_info())

    def run(self, in_datas: list, to_file=False):
        for idx, in_data in enumerate(in_datas):
            data = in_data["data"]
            if self.enable_aipp and in_data["enable_aipp"]:
                shape = data.shape
                h = shape[2]
                w = shape[3]
                image_format = 70  # 70 -> RGB888, 71 -> BGR888
                if shape[1] == 3:
                    image_format = 70 if in_data["pixel_format"] == "RGB" else 71
                elif shape[1] == 1:
                    image_format = 0
                else:
                    logger.error("Not support image shape -> {}".format(shape))
                    exit(-1)
                self.engine.set_aipp(batch_idx=0, input_idx=idx, image_format=image_format, image_size=[w, h])
            self.engine.set_input(0, idx, data.copy())

        self.engine.run()
        self.engine.clear_input_data(0, 0)  # 第二个参数未使用

        outputs = list()
        for idx in range(self.engine.get_num_outputs()):
            outputs.append(self.engine.get_output(0, idx).numpy())

        self.total += 1
        chip_cost = self.engine.get_profile_result()["last_model_exec_time"] * 0.001
        self.time_span += chip_cost

        if self.enable_dump == 1 and self.prefix != "sdk_iss":
            self.compare_layer_out()

        # dump输出
        if to_file:
            logger.info("[{}] predict result: outputs size -> {}".format(self.prefix, self.engine.get_num_outputs()))
            for idx, output in enumerate(outputs):
                logger.info("outputs[{}], shape={}, dtype={}".format(idx, output.shape, output.dtype))
                filepath_txt = os.path.join(self.result_dir, "{}_fixed_out_{}.txt".format(self.prefix, idx))
                filepath_bin = os.path.join(self.result_dir, "{}_fixed_out_{}.bin".format(self.prefix, idx))
                output.tofile(filepath_txt, sep="\n")
                output.tofile(filepath_bin)
                logger.info("save {}_fixed_output[{}] to {}".format(self.prefix, idx, filepath_txt))
                logger.info("save {}_fixed_output[{}] to {}".format(self.prefix, idx, filepath_bin))
        return outputs

    def unload(self):
        if self.engine:
            self.engine.unload_model()
            logger.info("unload model")
        if self.sdk:
            self.sdk.sdk_finalize()

    def compare_layer_out(self):
        model_name = "opt_ir"
        src = os.path.join(self.dump_root_path, model_name)
        if not os.path.exists(src):
            logger.error("Not found model dump path -> {}".format(src))
            exit(-1)

        chip_dump_out = os.path.join(self.result_dir, "chip_dump_out")
        if os.path.exists(chip_dump_out):
            shutil.rmtree(chip_dump_out)
        logger.info("cp {} -> {}".format(src, chip_dump_out))
        shutil.copytree(src, chip_dump_out)

        iss_fixed_dump_out = os.path.join(self.result_dir, "iss_fused_out.pickle")
        if not os.path.join(iss_fixed_dump_out):
            logger.error("Not found iss_fixed_dump_out -> {}".format(iss_fixed_dump_out))
            exit(-1)
        logger.info("###################### Chip(fixed) vs ISS(fixed) #######################")
        compare_dump_out(chip_dump_out, iss_fixed_dump_out)

        tvm_fixed_dump_out = os.path.join(self.result_dir, "quant", "output_tensors.params")
        if not os.path.exists(tvm_fixed_dump_out):
            logger.warning("Not found tvm_fixed_dump_out -> {}".format(tvm_fixed_dump_out))
            tvm_fixed_dump_out = None
        tvm_fp32_dump_out = os.path.join(self.result_dir, "fp32", "output_tensors.params")
        if not os.path.exists(tvm_fp32_dump_out):
            logger.warning("Not found tvm_fp32_dump_out -> {}".format(tvm_fp32_dump_out))
            tvm_fp32_dump_out = None

        if tvm_fp32_dump_out and tvm_fixed_dump_out:
            logger.info("###################### TVM(fixed) vs TVM(float) #######################")
            compare_dump_out2(tvm_fp32_dump_out, tvm_fixed_dump_out)

    def __del__(self):
        self.unload()
