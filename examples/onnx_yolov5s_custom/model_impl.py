#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : model_impl.py
@Time    : 2022/7/21 下午2:02
@Author  : xingwg
@Software: PyCharm
"""
from base.detection import Detector
from utils.postprocess import scale_coords


class YoloV5s(Detector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iou_threshold = 0.45
        self._conf_threshold = 0.25

    def _postprocess(self, outputs, cv_images=None):
        outputs = outputs[0]
        detections = list()
        for idx, cv_image in enumerate(cv_images):
            output = outputs[idx]
            output = output[output[:, 4] > 0]
            output[:, :4] = scale_coords(self._input_size, output[:, :4], cv_image.shape).round()
            detections.append(output)
        return detections
