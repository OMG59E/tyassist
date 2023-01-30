#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : model_impl.py
@Time    : 2022/7/21 下午2:02
@Author  : xingwg
@Software: PyCharm
"""
from base.detection import Detector


class YoloV5s(Detector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iou_threshold = 0.45
        self._conf_threshold = 0.25
