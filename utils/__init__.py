#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : __init__.py.py
@Time    : 2022/7/1 下午4:21
@Author  : xingwg
@Software: PyCharm
"""
import logging
from utils.glog_format import GLogFormatterWithColor


logger = logging.getLogger()

console_handler = logging.StreamHandler()
console_handler.setFormatter(GLogFormatterWithColor())

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
