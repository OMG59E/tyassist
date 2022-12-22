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
# import sys
# sys.path.append("/DEngine/tyassist")
import time
import shutil
from utils import logger
from utils.utils import get_host_ip
from utils.enum_type import PixelFormat
from utils.compare import compare_dump_out, compare_dump_out2
from utils.utils import get_md5


class Infer(object):
    def __init__(self, net_cfg_file="/DEngine/tyhcp/net.cfg", sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
                 enable_dump=0, max_batch=1):
        """
        :param net_cfg_file:
        :param sdk_cfg_file:
        :param enable_dump:
        """
        self._ip = "127.0.0.1"
        self._port = 9090
        self._net_cfg_file = net_cfg_file
        self._sdk_cfg_file = sdk_cfg_file
        self._enable_dump = enable_dump
        self._prefix = "chip"
        self._engine = None
        self._sdk = None
        self._max_batch = max_batch
        self._enable_aipp = False
        self._model_dir = ""
        self._result_dir = ""
        self._dump_root_path = ""
        self._ave_latency_ms = 0
        self._total = 0
        self._pixel_formats = [PixelFormat.RGB]  # 70

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

        if self._enable_dump:
            if self._ip == "127.0.0.1":   # TODO 非127.0.0.1的地址也可能是ISS服务
                self._prefix = "sdk_iss"
                logger.warning("ISS mode not support dump server, will be disable")
                self._enable_dump = 0

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
            if self._enable_dump == 1:
                self._sdk.select_dump_profile(DumpProfileSel.Dump)
                dump_server_ip = os.getenv("DUMP_SERVER_IP")
                dump_server_port = os.getenv("DUMP_SERVER_PORT")
                self._sdk.set_dump_server_ip(dump_server_ip, int(dump_server_port))
            else:
                self._sdk.select_dump_profile(DumpProfileSel.Profile)

            logger.info("sdk config path: {}".format(sdk_cfg_file))
            self._sdk.sdk_init(self._sdk_cfg_file)
            logger.info("tyhcp init succeed.")
        except Exception as e:
            logger.error("Import failed -> {}".format(e))
            exit(-1)

    def load(self, model_dir, model_name, enable_aipp=False):
        """加载芯片模型
        """
        self._enable_aipp = enable_aipp
        self._model_dir = model_dir
        self._result_dir = os.path.join(self._model_dir, "result")
        if not os.path.exists(self._result_dir):
            logger.error("Not found result_dir -> {}".format(self._result_dir))
            exit(-1)
        self._result_dir = os.path.abspath(self._result_dir)
        if self._enable_dump:
            self._dump_root_path = self._sdk.set_dump_work_path(self._result_dir)
            logger.info("dump root path: {}".format(self._dump_root_path))
        logger.info("sdk config: {}".format(self._sdk.get_sdk_config()))

        netbin_file = os.path.join(self._model_dir, "{}.ty".format(model_name))
        if not os.path.isfile(netbin_file):
            logger.error("netbin_file not file -> {}".format(netbin_file))
            exit(-1)

        logger.info("load model " + netbin_file)

        if not self._enable_aipp:
            self._sdk.disable_aipp()
        self._engine = self._sdk.create_model(self._max_batch)
        self._engine.load_model(netbin_file)
        logger.info(self._engine.get_model_info())

    def set_pixel_format(self, pixel_formats):
        self._pixel_formats = pixel_formats

    @property
    def prefix(self):
        return self._prefix

    @property
    def ave_latency_ms(self):
        if self._total == 0:
            return 0
        return self._ave_latency_ms / self._total

    def run(self, in_datas: list, input_enable_aipps=None, to_file=False):
        """推理
        :param in_datas: list表示多输入
        :param input_enable_aipps: 每个输入的aipp使能情况
        :param to_file: 表示是否将结果输出至文件
        :return:
        """
        if input_enable_aipps:
            if len(in_datas) != len(input_enable_aipps):
                logger.error("len(in_datas) must == len(input_enable_aipps) -> {} vs {}".format(
                    len(in_datas), len(input_enable_aipps)))
                exit(-1)
        else:
            input_enable_aipps = [False for _ in in_datas]  # 不传入aipp使能情况，默认关闭

        for idx, in_data in enumerate(in_datas):
            if self._enable_aipp and input_enable_aipps[idx]:
                shape = in_data.shape
                h = shape[2]
                w = shape[3]
                image_format = 70  # 70 -> RGB888, 71 -> BGR888
                if shape[1] == 3:
                    image_format = 70 if self._pixel_formats[idx] == PixelFormat.RGB else 71
                elif shape[1] == 1:
                    image_format = 0
                else:
                    logger.error("Not support image shape -> {}".format(shape))
                    exit(-1)
                self._engine.set_aipp(batch_idx=0, input_idx=idx, image_format=image_format, image_size=[w, h])
            # logger.info("md5: {}".format(get_md5(in_data.copy().tostring())))
            self._engine.set_input(0, idx, in_data.copy())

        # logger.info("model is running...")
        self._engine.run()
        self._engine.clear_input_data(0, 0)  # 第二个参数未使用

        outputs = list()
        for idx in range(self._engine.get_num_outputs()):
            outputs.append(self._engine.get_output(0, idx).numpy())

        self._total += 1
        chip_cost = self._engine.get_profile_result()["last_model_exec_time"] * 0.001
        self._ave_latency_ms += chip_cost

        if self._enable_dump == 1:
            self._compare_dump_out()

        # dump输出
        if to_file:
            logger.info("[{}] predict result: outputs size -> {}".format(self._prefix, self._engine.get_num_outputs()))
            for idx, output in enumerate(outputs):
                logger.info("outputs[{}], shape={}, dtype={}".format(idx, output.shape, output.dtype))
                filepath_txt = os.path.join(self._result_dir, "{}_fixed_out_{}.txt".format(self._prefix, idx))
                filepath_bin = os.path.join(self._result_dir, "{}_fixed_out_{}.bin".format(self._prefix, idx))
                output.tofile(filepath_txt, sep="\n")
                output.tofile(filepath_bin)
                logger.info("save {}_fixed_output[{}] to {}".format(self._prefix, idx, filepath_txt))
                logger.info("save {}_fixed_output[{}] to {}".format(self._prefix, idx, filepath_bin))
        return outputs

    def _compare_dump_out(self):
        model_name = "opt_ir"
        # filename_list = os.listdir(self._dump_root_path)
        # if len(filename_list) != 1:
        #     logger.error("dump_root_path file num must be == 1")
        #     exit(-1)
        # model_name = filename_list[0]

        src = os.path.join(self._dump_root_path, model_name)
        if not os.path.exists(src):
            logger.error("Not found model dump path -> {}".format(src))
            exit(-1)

        chip_dump_out = os.path.join(self._result_dir, "chip_dump_out")
        if os.path.exists(chip_dump_out):
            shutil.rmtree(chip_dump_out)
        logger.info("cp {} -> {}".format(src, chip_dump_out))
        shutil.copytree(src, chip_dump_out)

        iss_fixed_dump_out = os.path.join(self._result_dir, "iss_fused_out.pickle")
        if not os.path.join(iss_fixed_dump_out):
            logger.error("Not found iss_fixed_dump_out -> {}".format(iss_fixed_dump_out))
            exit(-1)
        logger.info("###################### Chip(fixed) vs ISS(fixed) #######################")
        compare_dump_out(chip_dump_out, iss_fixed_dump_out)

        tvm_fixed_dump_out = os.path.join(self._result_dir, "quant", "output_tensors.params")
        if not os.path.exists(tvm_fixed_dump_out):
            logger.warning("Not found tvm_fixed_dump_out -> {}".format(tvm_fixed_dump_out))
            tvm_fixed_dump_out = None
        tvm_fp32_dump_out = os.path.join(self._result_dir, "fp32", "output_tensors.params")
        if not os.path.exists(tvm_fp32_dump_out):
            logger.warning("Not found tvm_fp32_dump_out -> {}".format(tvm_fp32_dump_out))
            tvm_fp32_dump_out = None

        if tvm_fp32_dump_out and tvm_fixed_dump_out:
            logger.info("###################### TVM(fixed) vs TVM(float) #######################")
            compare_dump_out2(tvm_fp32_dump_out, tvm_fixed_dump_out)

    def __del__(self):
        if self._engine:
            self._engine.unload_model()
            logger.info("unload model")
        if self._sdk:
            self._sdk.sdk_finalize()


if __name__ == "__main__":
    infer = Infer()
