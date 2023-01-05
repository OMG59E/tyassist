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
        """用于处理模型输入图片的预处理
        :param filepath:
        :param idx:
        :return:
        """
        _input = self._inputs[idx]
        cv_image = cv2.imread(filepath)
        # 预处理
        shape = _input["shape"]
        n, c, h, w = shape
        if "NHWC" == _input["layout"]:
            n, h, w, c = shape

        # TODO 不支持输入在此处自定义实现，此处以默认预处理作为示例
        im = cv2.resize(cv_image, (w, h))
        if "RGB" == _input["pixel_format"]:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.expand_dims(im.transpose((2, 0, 1)), axis=0)
        return im.copy()

    def get_data(self):
        """工具链内部调用预处理来校准的函数
        :return:
        """
        for i in range(self._calib_num):
            filename = self._img_lists[i]
            _, ext = os.path.splitext(filename)
            if ext not in [".jpg", ".jpeg", ".JPEG", ".PNG", ".png", ".bmp"]:
                continue

            img_path = os.path.join(self._data_dir, self._img_lists[i])

            datas = dict()
            for idx, _input in enumerate(self._inputs):
                shape = _input["shape"]
                n, c, h, w = shape
                if "NHWC" == _input["layout"]:
                    n, h, w, c = shape

                if _input["support"]:
                    cv_image = cv2.imread(img_path)  # 此处作为示例，复用多个输入
                    datas[_input["name"]] = default_preprocess(
                        cv_image,
                        (w, h),
                        mean=_input["mean"],
                        std=_input["std"],
                        use_resize=True,
                        use_rgb=True if "RGB" == _input["pixel_format"] else False,
                        use_norm=False,
                        resize_type=0,
                    )
                else:
                    datas[_input["name"]] = self.get_single_data(img_path, idx)

            yield datas
