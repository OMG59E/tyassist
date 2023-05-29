#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : classification.py
@Time    : 2022/7/21 上午10:23
@Author  : xingwg
@Software: PyCharm
"""
import time
from abc import ABC

import numpy as np
import os
import cv2
import tqdm

from .base_model import BaseModel
from utils import logger
from utils.preprocess import default_preprocess


class Classifier(BaseModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n, c, h, w = self.inputs[0]["shape"]
        if self.inputs[0]["layout"] == "NHWC":
            n, h, w, c = self.inputs[0]["shape"]
        self._input_size = (w, h)
        self._end2end_latency_ms = 0

    def load(self, model_path):
        self.infer.load(model_path)

    def _preprocess(self, cv_image):
        return default_preprocess(
            cv_image,
            self._input_size,
            mean=self.inputs[0]["mean"],
            std=self.inputs[0]["std"],
            use_norm=self.use_norm,
            use_rgb=True if self.inputs[0]["pixel_format"] == "RGB" else False,
            use_resize=True if not self.enable_aipp else False,
            resize_type=self.inputs[0]["resize_type"],
            interpolation=cv2.INTER_LINEAR,
            padding_value=self.inputs[0]["padding_value"],
            padding_mode=self.inputs[0]["padding_mode"]
        )

    def _postprocess(self, outputs, cv_image=None):
        if len(outputs) != 1:
            logger.error("only support signal output, please check")
            exit(-1)
        outputs = outputs[0]  # [bs, num_cls] or [num_cls] for 4xx iss
        bs = outputs.shape[0]
        if bs != 1:
            logger.error("only support bs=1, please check")
            exit(-1)
        return outputs

    @property
    def ave_latency_ms(self):
        if self.backend != "tvm":
            self.infer.unload()
        return self.infer.ave_latency_ms

    @property
    def end2end_latency_ms(self):
        if self.total == 0:
            return 0
        return self._end2end_latency_ms / self.total

    def inference(self, cv_image):
        t_start = time.time()
        data = self._preprocess(cv_image)
        outputs = self.infer.run([data])
        output = self._postprocess(outputs, cv_image)
        end2end_cost = time.time() - t_start
        self._end2end_latency_ms += (end2end_cost * 1000)
        self.total += 1
        return output

    def evaluate(self):
        """ top-k
        """
        if not self.dataset:
            logger.error("The dataset is null")
            exit(-1)

        img_paths, labels = self.dataset.get_datas(num=self.test_num)

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
            "dataset": self.dataset.dataset_name,
            "num": total_num,
            "top1": "{:.6f}".format(top1),
            "top5": "{:.6f}".format(top5),
            "latency": "{:.6f}".format(self.ave_latency_ms)
        }

    def demo(self, img_path):
        if not os.path.exists(img_path):
            logger.error("The img path not exist -> {}".format(img_path))
            exit(-1)
        save_results = "vis_{}_{}".format(self.backend, self.dtype)
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        logger.info("process: {}".format(img_path))
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            logger.error("Failed to decode img by opencv -> {}".format(img_path))
            exit(-1)

        chip_output = self.inference(cv_image)
        max_idx = np.argmax(chip_output, axis=1).flatten()[0]
        max_prob = chip_output[0][max_idx].flatten()[0]
        logger.info("predict cls = {}, prob = {:.6f}, cls_name = {}".format(
            max_idx, max_prob, self.dataset.get_class_name(max_idx)))
        cv2.imwrite("{}/{}.jpg".format(save_results, self.dataset.get_class_name(max_idx)), cv_image)
