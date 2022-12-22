#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_infer.py 
@time: 2022/11/30
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import os
from abc import ABC
from utils import logger
from .base_infer import BaseSdkInfer


class Nnp4xxSdkInfer(BaseSdkInfer, ABC):
    def __init__(
            self,
            net_cfg_file="/DEngine/tyhcp/net.cfg",
            sdk_cfg_file="/DEngine/tyhcp/simu/config/sdk.cfg",
            enable_dump=0,
            enable_aipp=False
    ):
        super(Nnp4xxSdkInfer, self).__init__(
            net_cfg_file=net_cfg_file,
            sdk_cfg_file=sdk_cfg_file,
            enable_dump=enable_dump,
            enable_aipp=enable_aipp
        )
        self.prefix = "sdk_iss"

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

            logger.info("load model " + model_path)
            if not self.engine.load(model_path):
                logger.error("Failed to load model")
                exit(-1)
            logger.info("load model success")
            self.sdk = True
        except Exception as e:
            logger.error("load failed -> {}".format(e))
            exit(-1)

    def run(self, in_datas, to_file=False):
        outputs = self.engine.inference(in_datas)
        if to_file:
            logger.info("[{}] predict result: outputs size -> {}".format(self.prefix, len(outputs)))
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
            self.engine.unload()
            logger.info("unload model")
            self.engine = None
        if self.sdk:
            import python._sdk as _sdk
            _sdk.finalize()
            self.sdk = False

    def __del__(self):
        self.unload()

    def compare_layer_out(self):
        logger.warning("Nnp4xx not support compare")
