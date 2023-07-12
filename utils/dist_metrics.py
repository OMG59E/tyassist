#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : dist_metrics.py
@Time    : 2022/7/8 下午3:20
@Author  : xingwg
@Software: PyCharm
"""
import numpy as np


def cosine_distance(v1, v2):
    """余弦距离
    """
    v1_d = v1.flatten().astype("float64")
    v2_d = v2.flatten().astype("float64")
    assert len(v1_d) == len(v2_d), "v1 dim must be == v2 dim"
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    return v1_d.dot(v2_d) / np.maximum(np.linalg.norm(v1_d) * np.linalg.norm(v2_d), np.finfo(np.float32).eps)


def numerical_distance(v1, v2, epsilon=1e-3):
    v1_d = v1.flatten().astype("float64")
    v2_d = v2.flatten().astype("float64")
    assert len(v1_d) == len(v2_d), "v1 dim must be == v2 dim"
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    return len(np.where(np.abs(v1_d - v2_d) < epsilon)) / len(v2_d)
