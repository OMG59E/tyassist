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
from datasets.imagenet import ILSVRC2012_LABELS
from utils import logger
from utils.preprocess import default_preprocess


class Classifier(BaseModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def _postprocess(self, outputs, cv_images=None):
        if len(outputs) != 1:
            logger.error("only support signal output, please check")
            exit(-1)

        bs = len(cv_images)
        outputs = outputs[0]
        outputs = outputs[:bs, ...]
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

    def inference(self, cv_images: list):
        t_start = time.time()
        datas = np.zeros((self.bs, self.channels, self._input_size[1], self._input_size[0]),
                         dtype=np.uint8 if not self.use_norm else np.float32)
        for idx, cv_image in enumerate(cv_images):
            data = self._preprocess(cv_image)
            datas[idx, ...] = data
        outputs = self.infer.run([datas])
        outputs = self._postprocess(outputs, cv_images)
        end2end_cost = time.time() - t_start
        self._end2end_latency_ms += (end2end_cost * 1000)
        self.total += 1
        return outputs

    @staticmethod
    def top_k(k, labels, offset, outputs, top1, top5):
        idxes = np.argsort(-outputs, axis=1, kind="quicksort")[:, 0:k]  # 降序W
        for b, max_idxes in enumerate(idxes):
            label = labels[offset + b]
            if label == max_idxes[0]:
                top1 += 1
                top5 += 1
                continue
            if label in max_idxes:
                top5 += 1
        return top1, top5

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
        offset = 0
        cv_images = list()
        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning("Failed to decode img by opencv -> {}".format(img_path))
                continue
            cv_images.append(cv_image)

            if (idx + 1) % self.bs != 0:
                continue

            outputs = self.inference(cv_images)
            top1, top5 = self.top_k(k, labels, offset, outputs, top1, top5)
            cv_images.clear()
            offset += self.bs

        if len(cv_images) > 0:
            offset = total_num - total_num % self.bs
            outputs = self.inference(cv_images)
            top1, top5 = self.top_k(k, labels, offset, outputs, top1, top5)

        top1, top5 = float(top1)/total_num, float(top5)/total_num
        return {
            "input_size": "{}x{}x{}x{}".format(self.bs, 3, self._input_size[1], self._input_size[0]),
            "dataset": self.dataset.dataset_name,
            "num": total_num,
            "top1": "{:.6f}".format(top1),
            "top5": "{:.6f}".format(top5),
            "latency": "{:.6f}".format(self.ave_latency_ms)
        }

    @staticmethod
    def show_info(outputs):
        max_idxes = np.argmax(outputs, axis=1)
        for idx, max_idx in enumerate(max_idxes):
            max_prob = outputs[idx, max_idx]
            max_idx = str(max_idx)
            logger.info("predict cls: {}, prob: {:.6f}, cls_name: {}".format(
                max_idx, max_prob, ILSVRC2012_LABELS[max_idx][0]))

    def demo(self, img_paths: list):
        if len(img_paths) != self.bs:
            logger.warning("img_num({}) != batch_size({})".format(len(img_paths), self.bs))

        # save_results = "vis_{}_{}".format(self.backend, self.dtype)
        # if not os.path.exists(save_results):
        #     os.makedirs(save_results)

        cv_images = list()
        for img_path in img_paths:
            if not os.path.exists(img_path):
                logger.error("The img path not exist -> {}".format(img_path))
                exit(-1)
            logger.info("process: {}".format(img_path))
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.error("Failed to decode img by opencv -> {}".format(img_path))
                exit(-1)
            cv_images.append(cv_image)

        outputs = self.inference(cv_images)
        self.show_info(outputs)
