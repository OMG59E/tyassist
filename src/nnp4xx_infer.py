#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_infer.py 
@time: 2022/11/30
@Author  : xingwg
@software: PyCharm 
"""
import os
import shutil
import uuid
import json
import traceback
from abc import ABC
from utils import logger
from .base_infer import BaseInfer
from .nnp4xx_tyexec import Nnp4xxTyExec
from .nnp3xx_infer import Nnp3xxTvmInfer
from .nnp4xx_profiler import Nnp4xxProfileTypeEnum


class Nnp4xxSdkInfer(BaseInfer, ABC):
    def __init__(
            self,
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            enable_dump=0,
            enable_aipp=False
    ):
        super(Nnp4xxSdkInfer, self).__init__()

        self.sdk_cfg_file = sdk_cfg_file

        with open(self.sdk_cfg_file) as f:
            cfg = json.load(f)
        self.profile_dir = cfg["profiler"]["host_output"]

        self.enable_dump = enable_dump
        self.enable_aipp = enable_aipp

        self.dump_root_path = ""
        self.result_dir = ""
        self.uuid = str(uuid.uuid1())

    def load(self, model_path):
        self.result_dir = os.path.join(os.path.dirname(model_path), "result")
        if not os.path.exists(self.result_dir):
            logger.error("Not found result_dir -> {}".format(self.result_dir))
            exit(-1)

        if not os.path.isfile(model_path):
            logger.error("netbin_file not file -> {}".format(model_path))
            exit(-1)

        if self.enable_aipp:
            logger.warning("Nnp4xx not support aipp")

        if self.enable_dump:
            logger.warning("Nnp4xx not support dump")

        try:
            import python._sdk as _sdk
            logger.info("sdk config path: {}".format(self.sdk_cfg_file))
            _sdk.init(self.sdk_cfg_file)
            logger.info("tyhcp init succeed.")

            self.engine = _sdk.CNetOperator()

            if not self.engine.profile(Nnp4xxProfileTypeEnum.DCL_PROF_DCL_API):  # profile
                logger.error("Failed to set profile")
                exit(-1)

            logger.info("load model " + model_path)
            if not self.engine.load(model_path):
                logger.error("Failed to load model")
                exit(-1)
            logger.info("load model success")
        except Exception as e:
            logger.error("load failed -> {}".format(e))
            exit(-1)

    def run(self, in_datas: dict, to_file=False):
        if isinstance(in_datas, dict):
            in_datas = [in_datas[key] for key in in_datas]  # to list
        outputs = self.engine.inference(in_datas)
        self.total += 1
        if to_file:
            logger.info("[{}] predict result: outputs size -> {}".format(self.backend, len(outputs)))
            for idx, output in enumerate(outputs):
                logger.info("outputs[{}], shape={}, dtype={}".format(idx, output.shape, output.dtype))
                filepath_txt = os.path.join(self.result_dir, "{}_fixed_out_{}.txt".format(self.backend, idx))
                filepath_bin = os.path.join(self.result_dir, "{}_fixed_out_{}.bin".format(self.backend, idx))
                output.tofile(filepath_txt, sep="\n")
                output.tofile(filepath_bin)
                logger.info("save {}_fixed_output[{}] to {}".format(self.backend, idx, filepath_txt))
                logger.info("save {}_fixed_output[{}] to {}".format(self.backend, idx, filepath_bin))
        return outputs

    def unload(self):
        if self.engine:
            self.engine.unload()
            logger.info("unload model")
            self.engine = None
            import python._sdk as _sdk
            _sdk.finalize()
            # rename
            shutil.move(os.path.join(self.profile_dir, "dcl_api.bin"),
                        os.path.join(self.profile_dir, "dcl_api_{}.bin".format(self.uuid)))

    def __del__(self):
        self.unload()

    @property
    def ave_latency_ms(self):
        profile_file = os.path.join(self.profile_dir, "dcl_api_{}.bin".format(self.uuid))
        if not os.path.exists(profile_file):
            logger.error("Not found profile file -> {}".format(profile_file))
            exit(-1)

        try:
            import python._sdk as _sdk
            profile_json = _sdk.parse_dcl_api(profile_file)
            profile = json.loads(profile_json)
            total_time = profile["dclmdlExecute"] / 10**6 / self.total
            return total_time

        except Exception as e:
            logger.error("Failed to parse profile -> {}\n{}".format(e, traceback.format_exc()))
            exit(-1)


class Nnp4xxTvmInfer(Nnp3xxTvmInfer, ABC):
    def __init__(self):
        super().__init__()

    def load_json(self, model_path):
        import tvm
        from tvm import relay
        relay_func = tvm.relay.quantization.get_ir_from_json(model_path)
        self.engine = Nnp4xxTyExec.build_x86_64(relay_func, {}, self.target)

    def load(self, model_path):
        import tvm
        from tvm.contrib import graph_executor
        lib = tvm.runtime.load_module(model_path)
        self.engine = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu()))
