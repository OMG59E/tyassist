#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_profiler.py 
@time: 2022/12/22
@Author  : xingwg
@software: PyCharm 
"""
import os
import abc
import json
import traceback
from .base_profiler import BaseSdkProfiler
from utils import logger


class Nnp4xxSdkProfiler(BaseSdkProfiler, abc.ABC):
    def __init__(
            self,
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg"
    ):
        super(Nnp4xxSdkProfiler, self).__init__(sdk_cfg_file, "nnp400")

        with open(self.sdk_cfg_file) as f:
            cfg = json.load(f)
        self.profile_dir = cfg["profiler"]["host_output"]
        self.result_dir = ""

    def load(self, model_path):
        self.result_dir = os.path.join(os.path.dirname(model_path), "result")
        if not os.path.exists(self.result_dir):
            logger.error("Not found result_dir -> {}".format(self.result_dir))
            exit(-1)

        if not os.path.isfile(model_path):
            logger.error("netbin_file not file -> {}".format(model_path))
            exit(-1)

        try:
            import python._sdk as _sdk
            logger.info("sdk config path: {}".format(self.sdk_cfg_file))
            _sdk.init(self.sdk_cfg_file)
            logger.info("tyhcp init succeed.")

            self.engine = _sdk.CNetOperator()

            if not self.engine.profile():  # profile
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
        for _ in range(5):
            outputs = self.engine.inference(in_datas)

    def unload(self):
        if self.engine:
            self.engine.unload()
            logger.info("unload model")
            self.engine = None
            import python._sdk as _sdk
            _sdk.finalize()

    def __del__(self):
        self.unload()

    def parse(self):
        profile_file = os.path.join(self.profile_dir, "ai_core.bin")
        if not os.path.exists(profile_file):
            logger.error("Not found profile file -> {}".format(profile_file))
            exit(-1)
        try:
            import python._sdk as _sdk
            profile_str = _sdk.parse(profile_file)
            logger.info(profile_str)
            res = json.loads(profile_str)
            # dump
            profile_str = json.dumps(res, indent=2)
            with open("profile.json", "w") as f:
                f.write(profile_str)
        except Exception as e:
            logger.error("Failed to parse profile -> {}\n{}".format(e, traceback.format_exc()))
            exit(-1)




