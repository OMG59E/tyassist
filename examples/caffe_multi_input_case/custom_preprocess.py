#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : check.py
@Time    : 2022/7/13 上午10:02
@Author  : xingwg
@Software: PyCharm
"""
import os
import cv2
import numpy as np
from utils import logger
from utils.preprocess import default_preprocess
from base.base_custom_preprocess import BaseCustomPreprocess


class CustomMultiInputModel(BaseCustomPreprocess):
    """自定义预处理模块，可以自由定义，成员函数get_data为固定入口"""
    def __init__(self, inputs, calib_num, data_dir):
        """init
        :param input_infos: 模型输入信息
        :param calib_num:  校准数据量
        :param data_dir:  校准数据目录
        """
        self._inputs = inputs
        self._calib_num = calib_num
        self._data_dir = data_dir

        self._img_lists = os.listdir(data_dir)
        self._inputs = inputs

        if self._calib_num > len(self._img_lists):
            self._calib_num = len(self._img_lists)

    def get_single_data(self, filepath, idx):
        """ 非图像数据生成预处理数据
        :param filepath:  指定数据路径
        :param idx:  输入索引
        :return:
        """
        pass

    def get_data(self):
        """生成量化数据"""
        for i in range(self._calib_num):
            filename = self._img_lists[i]
            _, ext = os.path.splitext(filename)
            if ext not in [".jpg", ".jpeg", ".JPEG", ".PNG", ".png", ".bmp"]:
                continue

            img_path = os.path.join(self._data_dir, self._img_lists[i])

            datas = dict()
            for idx, _input in enumerate(self._inputs):
                shape = _input["shape"]
                pixel_format = _input["pixel_format"]
                dtype = _input["dtype"]
                n, c, h, w = shape
                if "NHWC" == _input["layout"]:
                    n, h, w, c = shape
                cv_image = cv2.imread(img_path)  # 此处作为示例，复用多个输入
                datas[_input["name"]] = default_preprocess(
                    cv_image,
                    (w, h),
                    mean=_input["mean"],
                    std=_input["std"],
                    use_resize=True,
                    use_rgb=True if "RGB" == pixel_format else False,
                    use_norm=False,
                    resize_type=0,
                ).astype(dtype=dtype)
            yield datas
