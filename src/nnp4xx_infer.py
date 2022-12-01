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
import python._sdk as _sdk
from utils.enum_type import PixelFormat
from utils import logger


class Infer(object):
    def __init__(self, net_cfg_file="/DEngine/tyhcp/net.cfg", sdk_cfg_file="/DEngine/tyhcp/simu/config/sdk.cfg",
                 enable_dump=0, max_batch=1):
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
        self._engine = None
        try:
            logger.info("sdk config path: {}".format(sdk_cfg_file))
            _sdk.init(self._sdk_cfg_file)
            logger.info("tyhcp init succeed.")
        except Exception as e:
            logger.error("Import failed -> {}".format(e))
            exit(-1)

        self._engine = _sdk.CNetOperator()

    @property
    def prefix(self):
        return self._prefix

    @property
    def ave_latency_ms(self):
        if self._total == 0:
            return 0
        return self._ave_latency_ms / self._total

    def load(self, model_dir, model_name, enable_aipp=False):
        self._enable_aipp = enable_aipp
        self._model_dir = model_dir
        self._result_dir = os.path.join(self._model_dir, "result")
        if not os.path.exists(self._result_dir):
            logger.error("Not found result_dir -> {}".format(self._result_dir))
            exit(-1)
        self._result_dir = os.path.abspath(self._result_dir)

        netbin_file = os.path.join(self._model_dir, "{}.ty".format(model_name))
        if not os.path.isfile(netbin_file):
            logger.error("netbin_file not file -> {}".format(netbin_file))
            exit(-1)

        logger.info("load model " + netbin_file)
        if not self._engine.load(netbin_file):
            logger.error("Failed to load model")
            exit(-1)
        logger.info("load model success")

    def run(self, in_datas):
        return self._engine.inference(in_datas)

    def __del__(self):
        if self._engine:
            self._engine.unload()
            logger.info("unload model")
        _sdk.finalize()
