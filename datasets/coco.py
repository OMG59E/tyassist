#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : coco.py
@Time    : 2022/7/25 下午7:35
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import os
from base.dataset_base import DatasetBase
from utils import logger


class COCO2017Val(DatasetBase):
    """提供图片path和label
    """
    def __init__(self, root_path, batch_size=1):
        self._root_path = root_path
        self._batch_size = batch_size
        if not os.path.exists(self._root_path):
            logger.error("root_path not exits -> {}".format(self._root_path))
            exit(-1)
        # self._label_path = os.path.join(self._root_path, "labels", "val2017")

        self._filepath = os.path.join(self._root_path, "..", "val2017.txt")
        if not os.path.exists(self._filepath):
            logger.error("filepath not exist -> {}".format(self._filepath))
            exit(-1)
        with open(self._filepath, "r") as f:
            lines = f.readlines()

        self._label_files = list()
        self._img_files = list()
        for line in lines:
            sub_path = line.strip()
            basename = os.path.basename(sub_path)
            filename, ext = os.path.splitext(basename)
            img_path = os.path.join(self._root_path, "..", sub_path)
            if not os.path.exists(img_path):
                logger.warning("img_path not exist -> {}".format(img_path))
                continue
            # label_path = os.path.join(self._root_path, "labels", "val2017", "{}.txt".format(filename))
            # if not os.path.exists(label_path):
            #     logger.warning("label_path not exist -> {}".format(label_path))
            #     continue
            self._img_files.append(img_path)
            # self._label_files.append(label_path)
        self._total_num = len(self._img_files)
        # if len(self._img_files) != len(self._label_files):
        #     logger.error("img_files_num must be equal label_files_num -> {} vs {}".format(
        #         len(self._img_files), len(self._label_files)))
        #     exit(-1)

    def get_next_batch(self):
        """获取下一批数据
        """
        pass

    def get_datas(self, num: int):
        """截取部分数据
        :param num: 0表示使用全部数据，否则按num截取，超出全部则按全部截取
        :return:
        """
        if num == 0:
            num = self._total_num
        elif num > self._total_num:
            num = self._total_num

        img_paths = self._img_files[0:num]
        # labels = self._labels[0:num]
        return img_paths
