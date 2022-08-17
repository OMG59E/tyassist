#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : detection.py
@Time    : 2022/7/21 上午10:24
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import time
import os
import cv2
import torch
import tqdm

from base.classification import Classifier
from utils.postprocess import (
    non_max_suppression,
    scale_coords,
)
from utils.metrics import (
    coco_eval,
    detection_txt2json,
    detections2txt,
)
from utils.enum_type import PaddingMode
from utils import logger


class Detector(Classifier):
    def __init__(self, input_size: tuple, mean: tuple, std: tuple, use_rgb=False, use_norm=False, resize_type=0,
                 padding_value=114, padding_mode=PaddingMode.CENTER, dataset=None, test_num=0):
        super().__init__(
            input_size,
            mean,
            std,
            use_rgb=use_rgb,
            use_norm=use_norm,
            resize_type=resize_type,
            padding_value=padding_value,
            padding_mode=padding_mode,
            dataset=dataset,
            test_num=test_num
        )
        self._iou_threshold = 0.45
        self._conf_threshold = 0.25

    def set_iou_threshold(self, iou_threshold=0.45):
        self._iou_threshold = iou_threshold

    def set_conf_threshold(self, conf_threshold=0.25):
        self._conf_threshold = conf_threshold

    def _postprocess(self, outputs, cv_image=None):
        if len(outputs) == 4 or len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 3:
            logger.error("Not support yet.")
            exit(-1)
        else:
            logger.error("Output num error -> {}".format(len(outputs)))
            exit(-1)
        outputs = torch.from_numpy(outputs)
        outputs = non_max_suppression(outputs, self._conf_threshold, self._iou_threshold)
        outputs = outputs[0]  # bs=1
        outputs[:, :4] = scale_coords(self._input_size, outputs[:, :4], cv_image.shape).round()
        return outputs.numpy()

    def evaluate(self):
        if not self._dataset:
            logger.error("The dataset is null")
            exit(-1)

        img_paths = self._dataset.get_datas(num=self._test_num)

        save_results = "results" if self.is_fixed else "results_fp32"
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)
            label_path = os.path.join(save_results, "{}.txt".format(filename))
            if os.path.exists(label_path):
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning("Failed to decode img by opencv -> {}".format(img_path))
                continue
            detections = self.inference(cv_image)
            detections2txt(detections, label_path)
        pred_json = "pred.json"
        detection_txt2json(save_results, pred_json)
        _map, map50 = coco_eval(pred_json, self._dataset.annotations_file, self._dataset.image_ids)
        return {
            "input_size": "{}x{}x{}x{}".format(1, 3, self._input_size[1], self._input_size[0]),
            "dataset": self._dataset.dataset_name,
            "num": len(img_paths),
            "map": "{:.6f}".format(_map),
            "map50": "{:.6f}".format(map50),
            "latency": "{:.6f}".format(self.ave_latency_ms)
        }

    def demo(self, img_path):
        if not os.path.exists(img_path):
            logger.error("The img path not exist -> {}".format(img_path))
            exit(-1)
        filename = os.path.basename(img_path)
        logger.info("process: {}".format(img_path))
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            logger.error("Failed to decode img by opencv -> {}".format(img_path))
            exit(-1)

        detections = self.inference(cv_image)

        save_results = "vis" if self.is_fixed else "vis_fp32"
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        for det in detections:
            (x1, y1, x2, y2), conf, cls = list(map(int, det[0:4])), det[4], int(det[5])
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2, 8)
            cv2.imwrite(os.path.join(save_results, filename), cv_image)
            logger.info("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, int(cls)))
