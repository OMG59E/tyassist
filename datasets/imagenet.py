#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : imagenet.py
@Time    : 2022/7/15 下午4:32
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import os
from utils import logger
from base.dataset_base import DatasetBase


class ILSVRC2012(DatasetBase):
    """
    imagenet dataset
    """
    def __init__(self, root_path, batch_size=1):
        """
        :param root_path:
        :param batch_size:
        """
        if not os.path.exists(root_path):
            logger.error("ILSVRC2012 dataset path not exist -> {}".format(root_path))
            exit(-1)
        self._batch_idx = 0
        self._batch_size = batch_size
        self._data_root_path = root_path
        self._val_file = os.path.join(self._data_root_path, "..", "val.txt")
        if not os.path.exists(self._val_file):
            logger.error("Not found val file -> {}".format(self._val_file))
            exit(-1)
        self._filepaths = list()
        self._labels = list()
        with open(self._val_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                filename, cls = line.strip().split()
                _, ext = os.path.splitext(filename)
                if ext not in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
                    continue
                filepath = os.path.join(self._data_root_path, filename)
                if not os.path.exists(filepath):
                    continue
                self._filepaths.append(filepath)
                self._labels.append(int(cls))
        self._total_num = len(self._filepaths)

    def get_next_batch(self):
        """Generate a batch of image path.
        """
        filepath = self._filepaths[self._batch_idx]
        self._batch_idx += 1
        return filepath

    def get_datas(self, num=0):
        """
        :param num:
        :return:
        """
        if num == 0:
            num = self._total_num
        elif num > self._total_num:
            num = self._total_num

        img_paths = self._filepaths[0:num]
        labels = self._labels[0:num]
        return img_paths, labels
