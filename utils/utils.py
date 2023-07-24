#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2022/7/12 下午2:17
@Author  : xingwg
@Software: PyCharm
"""
import socket
import hashlib
import importlib


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def get_md5(data):
    md5 = hashlib.md5()
    md5.update(data)
    return md5.hexdigest()


def get_method(module, method):
    m = importlib.import_module(module)
    assert hasattr(m, method)
    return getattr(m, method)
