#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : enum_type.py
@Time    : 2022/7/14 上午10:54
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
from enum import Enum


class PaddingMode(Enum):
    LEFT_TOP = 0
    CENTER = 1


class PixelFormat(Enum):
    NONE = -1
    BGR = 0
    RGB = 1
    GRAY = 2
    YUV420P = 3
    YUV420SP_NV12 = 4


class DataLayout(Enum):
    NCHW = 0
    NHWC = 1


class DataType(Enum):
    UINT8 = 0
    FLOAT32 = 1
