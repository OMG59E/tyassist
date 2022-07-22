#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : classification.py
@Time    : 2022/7/21 上午10:23
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import numpy as np
import os
import cv2
from .model_base import ModelBase
from src.infer import Infer
from utils import logger
from utils.preprocess import default_preprocess
from utils.enum_type import PaddingMode


class Classifier(ModelBase):
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
        self._resize_type =resize_type
        self._padding_value = padding_value
        self._padding_mode = padding_mode

    def load(self, model_dir: str, net_cfg_file="/DEngine/tyhcp/net.cfg",
             sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg", ip="127.0.0.1", port=9090,
             enable_dump=False, max_batch=1):
        self._infer = Infer(
            net_cfg_file=net_cfg_file,
            sdk_cfg_file=sdk_cfg_file,
            ip=ip,
            port=port,
            enable_dump=enable_dump,
            max_batch=max_batch
        )
        self._infer.load(model_dir)

    def _preprocess(self, cv_image):
        return default_preprocess(
            cv_image,
            self._input_size,
            mean=self._mean,
            std=self._std,
            use_norm=self._use_norm,
            use_rgb=self._use_rgb,
            resize_type=self._resize_type,
            interpolation=cv2.INTER_LINEAR,
            padding_value=self._padding_value,
            padding_mode=self._padding_mode
        )

    def _postprocess(self, outputs):
        return outputs

    def inference(self, cv_image):
        data = self._preprocess(cv_image)
        chip_outputs = self._infer.run([data])
        if len(chip_outputs) != 1:
            logger.error("Squeezenet only one output, please check")
            exit(-1)
        chip_output = chip_outputs[0]
        chip_output = self._postprocess(chip_output)
        return chip_output

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
        for idx, img_path in enumerate(img_paths):
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning("Failed to decode img by opencv -> {}".format(img_path))
                continue

            chip_output = self.inference(cv_image)
            idxes = np.argsort(-chip_output, axis=1, kind="quicksort").flatten()[0:k]  # 降序
            logger.info("pred = {}, gt = {}".format(idxes, labels[idx]))
            if labels[idx] == idxes[0]:
                top1 += 1
                top5 += 1
                continue
            if labels[idx] in idxes:
                top5 += 1
        logger.info("accuracy top-1 = {:.6f}, top-5 = {:.6f}".format(float(top1)/total_num, float(top5)/total_num))

    def demo(self, img_path):
        if not os.path.exists(img_path):
            logger.error("The img path not exist -> {}".format(img_path))
            exit(-1)
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            logger.error("Failed to decode img by opencv -> {}".format(img_path))
            exit(-1)

        chip_output = self.inference(cv_image)
        max_idx = np.argmax(chip_output, axis=1).flatten()[0]
        max_prob = chip_output[:, max_idx].flatten()[0]
        logger.info("predict cls = {}, prob = {}".format(max_idx, max_prob))