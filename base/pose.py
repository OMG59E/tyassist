#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tqdm
import torch

from utils.postprocess import non_max_suppression_kpt, \
    scale_coords_kpt, output_to_keypoint, plot_skeleton_kpts
from utils.metrics import coco_eval, merge_json, detections_kpt2json
from base.detection import Detector
from utils import logger


class Pose(Detector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iou_threshold = 0.45
        self._conf_threshold = 0.25

    def _postprocess(self, outputs, cv_images=None):
        outputs = torch.from_numpy(outputs[0])  # bsx25500x57
        outputs = non_max_suppression_kpt(
            outputs, self._conf_threshold, self._iou_threshold, nc=1, nkpt=17, kpt_label=True)
        results = list()
        for idx, output in enumerate(outputs):
            results.append(scale_coords_kpt(self._input_size, output, cv_images[idx].shape))
        return results

    def evaluate(self):
        if not self.dataset:
            logger.error("The dataset is null")
            exit(-1)

        self._iou_threshold = 0.65
        self._conf_threshold = 0.01

        img_paths = self.dataset.get_datas(num=self.test_num)

        save_results = "results_{}_{}".format(self.backend, self.dtype)
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        cv_images = list()
        label_paths = list()
        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)
            label_path = os.path.join(save_results, "{}.json".format(filename))
            if os.path.exists(label_path):
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning("Failed to decode img by opencv -> {}".format(img_path))
                continue
            cv_images.append(cv_image)
            label_paths.append(label_path)

            if (idx + 1) % self.bs != 0:
                continue

            outputs = self.inference(cv_images)
            for b in range(len(cv_images)):
                detections_kpt2json(outputs[b], label_paths[b])
            cv_images.clear()
            label_paths.clear()

        if len(cv_images) > 1:
            outputs = self.inference(cv_images)
            for b in range(len(cv_images)):
                detections_kpt2json(outputs[b], label_paths[b])

        pred_json = "pred.json"
        merge_json(save_results, pred_json)
        _map, map50 = coco_eval(pred_json, self.dataset.annotations_kpt, self.dataset.image_ids, "keypoints")
        return {
            "input_size": "{}x{}x{}x{}".format(self.bs, 3, self._input_size[1], self._input_size[0]),
            "dataset": self.dataset.dataset_name,
            "num": len(img_paths),
            "map": "{:.6f}".format(_map),
            "map50": "{:.6f}".format(map50),
            "latency": "{:.6f}".format(self.ave_latency_ms)
        }

    def demo(self, img_paths):
        save_results = "vis_{}_{}".format(self.backend, self.dtype)
        if not os.path.exists(save_results):
            os.makedirs(save_results)

        filenames = list()
        cv_images = list()
        for img_path in img_paths:
            if not os.path.exists(img_path):
                logger.error("The img path not exist -> {}".format(img_path))
                exit(-1)
            filename = os.path.basename(img_path)
            logger.info("process: {}".format(img_path))
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.error("Failed to decode img by opencv -> {}".format(img_path))
                exit(-1)
            cv_images.append(cv_image)
            filenames.append(filename)

        outputs = self.inference(cv_images)

        for idx, cv_image in enumerate(cv_images):
            for output in outputs[idx]:
                plot_skeleton_kpts(cv_images[idx], output[6:].T, 3)
                x1, y1, x2, y2 = output[0:4]
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, 8)
            cv2.imwrite(os.path.join(save_results, filenames[idx]), cv_images[idx])
