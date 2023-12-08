#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : dist_metrics.py
@Time    : 2022/7/8 下午3:20
@Author  : xingwg
@Software: PyCharm
"""
import functools
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


def get_cos_similarity(data1, data2, mask=None):
    """cal cos_similarity"""
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    assert len(v1_d) == len(v2_d), "calc cos_similarity must have same length!"
    if np.all(v1_d == v2_d):
        return 1.0
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    if mask is not None:
        v1_d = v1_d[mask.flatten()]
        v2_d = v2_d[mask.flatten()]
    v1_norm = v1_d / (1e-30 + np.linalg.norm(v1_d))
    v2_norm = v2_d / (1e-30 + np.linalg.norm(v2_d))
    sim = np.dot(v1_norm, v2_norm)
    if np.linalg.norm(v1_d) * np.linalg.norm(v2_d) == 0:
        return 1.0
    return sim


def get_cos_similarity_per_channel_average(expect, actual, layout="NCHW"):
    """Compute cosine similarity per channel if the shape match assumed layout.

    Also implement following smart rules if the actual shape mismatch:
    - If the expect's ndim is 4 and the actual is 5, assume the actual is NCHWc
    - If the different dimension size is 1, skip them

    Return None if the shape mismatch could not get handled.
    """

    if expect.shape != actual.shape:
        expect_volume = functools.reduce(lambda a, b: a * b, expect.shape, 1)
        actual_volume = functools.reduce(lambda a, b: a * b, actual.shape, 1)
        if expect_volume != actual_volume:
            return None

        # NCHW - NCHWc
        if layout == "NCHW" and len(expect.shape) == 4 and len(actual.shape) == 5:
            y = np.transpose(actual, [0, 1, 4, 2, 3])
            y = np.reshape(y, [actual.shape[0], -1, actual.shape[2], actual.shape[3]])
            if y.shape != expect.shape:
                return None
            actual = y

        # NCHW - NCHWc
        if layout == "NCHW" and len(expect.shape) == 5 and len(actual.shape) == 4:
            y = np.transpose(expect, [0, 1, 4, 2, 3])
            y = np.reshape(y, [expect.shape[0], -1, expect.shape[2], expect.shape[3]])
            if y.shape != actual.shape:
                return None
            expect = y

        # only diff at expand dims
        elif [_ for _ in actual.shape if _ > 0] == [_ for _ in expect.shape if _ > 0]:
            actual = np.reshape(actual, expect.shape)
        else:
            return None

    if len(expect.shape) == len(layout):
        cidx = layout.find("C")
        if cidx >= 0:
            cosine_list = []
            for c in range(expect.shape[cidx]):
                per_channel_expect = expect.take(indices=c, axis=cidx)
                per_channel_actual = actual.take(indices=c, axis=cidx)
                cosine_list.append(get_cos_similarity(per_channel_expect, per_channel_actual))
            return np.mean(cosine_list)

    return get_cos_similarity(expect, actual)
