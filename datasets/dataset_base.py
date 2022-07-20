#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : dataset_base.py
@Time    : 2022/7/15 下午4:15
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import abc


class DatasetBase(object, metaclass=abc.ABCMeta):
    """提供图片path和label
    """
    @abc.abstractmethod
    def __init__(self, root_path, batch_size=1):
        pass

    @abc.abstractmethod
    def get_next_batch(self):
        """获取下一批数据
        """
        pass

    @abc.abstractmethod
    def get_datas(self, num: int):
        """截取部分数据
        :param num: 0表示使用全部数据，否则按num截取，超出全部则按全部截取
        :return:
        """
        pass
