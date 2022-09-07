#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : classification.py
@Time    : 2022/7/21 上午10:23
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import time
from abc import ABC

import numpy as np
import os
import cv2
import tqdm

from .model_base import ModelBase
from utils import logger
from utils.preprocess import default_preprocess
from utils.enum_type import PaddingMode, DataType


class Classifier(ModelBase, ABC):
    def __init__(self, input_size: tuple, mean: tuple, std: tuple, use_rgb=False, use_norm=False,
                 resize_type=0, padding_value=128, padding_mode=PaddingMode.LEFT_TOP, dataset=None, test_num=0):
        self._dataset = dataset
        self._test_num = test_num
        self._infer = None
        self._input_size = input_size
        self._mean = mean
        self._std = std
        self._use_rgb = use_rgb
        self._use_norm = use_norm
        self._resize_type = resize_type
        self._padding_value = padding_value
        self._padding_mode = padding_mode
        self._end2end_latency_ms = 0
        self._total = 0
        self._enable_aipp = False
        self._model_dir = ""
        self._dtype = DataType.INT8
        self._input_enable_aipps = None
        self._input_pixel_format = None

    def load(self, model_dir: str, net_cfg_file="/DEngine/tyhcp/net.cfg",
             sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg", enable_aipp=False, enable_dump=False, max_batch=1):
        from src.infer import Infer
        self._infer = Infer(
            net_cfg_file=net_cfg_file,
            sdk_cfg_file=sdk_cfg_file,
            enable_dump=enable_dump,
            max_batch=max_batch
        )
        self._enable_aipp = enable_aipp if DataType.INT8 == self._dtype else False   # 除芯片定点外强制关闭aipp
        self._infer.load(model_dir, self._enable_aipp)

    def set_dtype(self, dtype=DataType.INT8):
        self._dtype = dtype

    def set_input_enable_aipps(self, input_enable_aipps):
        """设置传入每个输入的aipp使能情况，支持进行aipp推理"""
        self._input_enable_aipps = input_enable_aipps

    def set_input_pixel_format(self, pixel_format: list):
        """传入每个输入的像素格式，支持进行aipp推理"""
        self._input_pixel_format = pixel_format
        self._infer.set_pixel_format(pixel_format)

    @property
    def dtype(self):
        return self._dtype

    @property
    def use_norm(self):
        if DataType.INT8 == self._dtype or DataType.TVM_INT8 == self._dtype:
            return False
        return True

    def load_relay_from_mem(self, input_names: list, callback):
        from src.infer_relay import InferRelay
        self._infer = InferRelay(input_names)
        self._infer.load_from_mem(callback)

    def load_relay_from_json(self, input_names: list, filepath):
        from src.infer_relay import InferRelay
        self._infer = InferRelay(input_names)
        self._infer.load_from_json(filepath)

    def _preprocess(self, cv_image):
        return default_preprocess(
            cv_image,
            self._input_size,
            mean=self._mean,
            std=self._std,
            use_norm=self.use_norm,
            use_rgb=self._use_rgb,
            use_resize=False if self._enable_aipp else True,
            resize_type=self._resize_type,
            interpolation=cv2.INTER_LINEAR,
            padding_value=self._padding_value,
            padding_mode=self._padding_mode
        )

    def _postprocess(self, outputs, cv_image=None):
        if len(outputs) != 1:
            logger.error("only one output, please check")
            exit(-1)
        outputs = outputs[0]  # bs=1
        return outputs

    @property
    def ave_latency_ms(self):
        return self._infer.ave_latency_ms

    @property
    def end2end_latency_ms(self):
        if self._total == 0:
            return 0
        return self._end2end_latency_ms / self._total

    def inference(self, cv_image):
        t_start = time.time()
        data = self._preprocess(cv_image)
        outputs = self._infer.run([data], self._input_enable_aipps)
        output = self._postprocess(outputs, cv_image)
        end2end_cost = time.time() - t_start
        self._end2end_latency_ms += (end2end_cost * 1000)
        self._total += 1
        return output

    def evaluate(self):
        """ top-k
        """
        if not self._dataset:
            logger.error("The dataset is null")
            exit(-1)

        img_paths, labels = self._dataset.get_datas(num=self._test_num)

        k = 5
        top1, top5 = 0, 0
        total_num = len(img_paths)
        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning("Failed to decode img by opencv -> {}".format(img_path))
                continue

            chip_output = self.inference(cv_image)
            idxes = np.argsort(-chip_output, axis=1, kind="quicksort").flatten()[0:k]  # 降序
            # logger.info("pred = {}, gt = {}".format(idxes, labels[idx]))
            if labels[idx] == idxes[0]:
                top1 += 1
                top5 += 1
                continue
            if labels[idx] in idxes:
                top5 += 1
        top1, top5 = float(top1)/total_num, float(top5)/total_num
        return {
            "input_size": "{}x{}x{}x{}".format(1, 3, self._input_size[1], self._input_size[0]),
            "dataset": self._dataset.dataset_name,
            "num": total_num,
            "top1": "{:.6f}".format(top1),
            "top5": "{:.6f}".format(top5),
            "latency": "{:.6f}".format(self.ave_latency_ms)
        }

    def demo(self, img_path):
        if not os.path.exists(img_path):
            logger.error("The img path not exist -> {}".format(img_path))
            exit(-1)
        logger.info("process: {}".format(img_path))
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            logger.error("Failed to decode img by opencv -> {}".format(img_path))
            exit(-1)

        chip_output = self.inference(cv_image)
        max_idx = np.argmax(chip_output, axis=1).flatten()[0]
        max_prob = chip_output[:, max_idx].flatten()[0]
        logger.info("predict cls = {}, prob = {:.6f}".format(max_idx, max_prob))
