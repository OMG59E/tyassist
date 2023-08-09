#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tqdm
import torch

from utils.postprocess import non_max_suppression, scale_coords, process_mask, scale_coords_mask
from utils.metrics import coco_eval, merge_json, detections_mask2json
from base.detection import Detector
from utils import logger


COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (169, 169, 169), (0, 0, 139), (0, 69, 255), (30, 105, 210),
          (10, 215, 255), (0, 255, 255), (0, 128, 128), (144, 238, 144), (139, 139, 0), (230, 216, 173), (130, 0, 75),
          (128, 0, 128), (203, 192, 255), (147, 20, 255), (238, 130, 238)]


class Segmentation(Detector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iou_threshold = 0.45
        self._conf_threshold = 0.25
        self._nm = 32

    def _postprocess(self, outputs, cv_images=None):
        pred, proto = outputs
        pred = torch.from_numpy(pred)  # bsx25200x117
        proto = torch.from_numpy(proto)  # bsx32x160x160
        pred = non_max_suppression(pred, self._conf_threshold, self._iou_threshold, nm=self._nm)
        _masks, _contours = list(), list()
        for idx, cv_image in enumerate(cv_images):
            tmp_contours = list()
            tmp_masks = list()
            if pred[idx].shape[0] > 0:
                masks = process_mask(proto[idx], pred[idx][:, 6:], pred[idx][:, :4], self._input_size, upsample=True)  # HWC
                pred[idx][:, :4] = scale_coords(self._input_size, pred[idx][:, :4], cv_image.shape).round()
                masks = masks.numpy()
                h, w, c = cv_image.shape
                for _, mask in enumerate(masks):
                    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if isinstance(contours, tuple):
                        contours = list(contours)
                    contours = scale_coords_mask(self._input_size, contours, cv_image.shape)
                    tmp_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(tmp_mask, contours, 255)
                    tmp_masks.append(tmp_mask)
                    tmp_contours.append(contours)
            _masks.append(tmp_masks)
            _contours.append(tmp_contours)
        return pred, _masks, _contours

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

            detections, masks, contours = self.inference(cv_images)
            for b in range(len(masks)):
                detections_mask2json(detections[b], contours[b], label_paths[b])
            cv_images.clear()
            label_paths.clear()

        if len(cv_images) > 0:
            detections, masks, contours = self.inference(cv_images)
            for b in range(len(masks)):
                detections_mask2json(detections[b], contours[b], label_paths[b])

        pred_json = "pred.json"
        merge_json(save_results, pred_json)
        _map, map50 = coco_eval(pred_json, self.dataset.annotations_file, self.dataset.image_ids, "segm")
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

        detections, masks, contours = self.inference(cv_images)

        # draw mask
        for idx, cv_image in enumerate(cv_images):
            _detections, _masks, _contours = detections[idx].numpy(), masks[idx], contours[idx]
            for k, detection in enumerate(_detections):
                contour = _contours[k]
                mask = _masks[k]
                new_masks = np.array([mask, mask, mask]).transpose((1, 2, 0))
                np.random.shuffle(COLORS)
                color = np.array(COLORS[0])
                cv_image = np.where(new_masks == 255, cv_image * 0.5 + color * 0.5, cv_image)
                cv2.drawContours(cv_image, contour, -1, color.tolist(), 2)
                (x1, y1, x2, y2), conf, cls = list(map(int, detection[0:4])), detection[4], int(detection[5])
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color.tolist(), 2, 8)
                logger.info("x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(x1, y1, x2, y2, conf, int(cls)))
            cv2.imwrite(os.path.join(save_results, filenames[idx]), cv_image)
