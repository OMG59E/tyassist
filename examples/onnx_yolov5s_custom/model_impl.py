#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : model_impl.py
@Time    : 2022/7/21 下午2:02
@Author  : xingwg
@Software: PyCharm
"""
from utils.enum_type import PaddingMode
from base.detection import Detector
from utils.postprocess import scale_coords


class YoloV5s(Detector):
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
        self._conf_threshold = 0.0

    def _postprocess(self, outputs, cv_image=None):
        outputs = outputs[0]
        conf = outputs[:, :, 4]
        outputs = outputs[conf > self._conf_threshold]
        outputs[:, :4] = scale_coords(self._input_size, outputs[:, :4], cv_image.shape).round()
        return outputs


