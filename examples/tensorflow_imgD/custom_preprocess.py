
import os
import cv2
import numpy as np
from base.base_custom_preprocess import BaseCustomPreprocess


class CustomTensorFlowImgD(BaseCustomPreprocess):
    """自定义预处理模块，可以自由定义，成员函数get_data为固定入口
    """
    def __init__(self, inputs: list, calib_num: int, data_dir: str):
        """
        :param inputs: 模型输入信息
        :param calib_num:  校准数据量
        :param data_dir:  校准数据目录
        """
        assert os.path.exists(data_dir), "calib data path not exist"

        self._inputs = inputs
        self._calib_num = calib_num
        self._data_dir = data_dir

        img_lists = os.listdir(os.path.join(data_dir, "rgb"))
        depth_lists = os.listdir(os.path.join(data_dir, "depth"))

        assert len(img_lists) == len(depth_lists), "img num size must be equal depth num size"

        self._input_h = self._inputs[0]["shape"][1] if self._inputs[0]["layout"] == "NHWC" else self._inputs[0]["shape"][2]
        self._input_w = self._inputs[0]["shape"][2] if self._inputs[0]["layout"] == "NHWC" else self._inputs[0]["shape"][3]
        # self._dtype = np.uint8 if self._inputs[0]["dtype"] == "uint8" else np.float32
        self._calib_num = calib_num if calib_num < len(img_lists) else len(img_lists)

        self._data_lists = list()
        for img_name in img_lists:
            name, ext = os.path.splitext(img_name)
            if ext not in [".png", ".jpg", ".bmp"]:
                continue
            img_path = os.path.join(self._data_dir, "rgb", img_name)
            assert os.path.exists(img_path)
            depth_path = os.path.join(self._data_dir, "depth", "{}_depth{}".format(name, ext))
            assert os.path.exists(depth_path)
            self._data_lists.append((img_path, depth_path))

    def _preprocess(self, filepath1, filepath2, use_norm=True):
        """预处理 仅需支持resize和cvtColor
        :param filepath1: BGR图像
        :param filepath2: 深度图
        :param use_norm:
        :return:
        """
        # -- step1 get 3-ch data
        ch3_img = cv2.imread(filepath1)
        ch3_img = cv2.resize(ch3_img, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)  # BGR NHWC
        ch3_img = ch3_img[:, :, ::-1]  # to RGB

        # -- step2 get 1-ch data
        ch1_img = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)
        ch1_img = cv2.resize(ch1_img, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)
        ch1_img = np.expand_dims(ch1_img, axis=-1)

        # -- step3 concat
        out = np.concatenate((ch3_img, ch1_img), axis=2)
        if use_norm:
            mean = self._inputs[0]["mean"] if self._inputs[0]["mean"] else [0.0 for _ in range(len(self._inputs[0]["shape"]))]
            std = self._inputs[0]["std"] if self._inputs[0]["std"] else [1.0 for _ in range(len(self._inputs[0]["shape"]))]
            out = out.astype(dtype=np.float32)
            out -= np.array(mean, dtype=np.float32)
            out /= np.array(std, dtype=np.float32)

        out = np.expand_dims(out.transpose((2, 0, 1)), axis=0)
        return out

    def get_single_data(self, filepath):
        """用于处理指定输入图片的预处理，一般用于推理计算相似度
            不需要norm归一化，工具链会根据配置文件，内部进行norm
        :param filepath:
        :return:
        """
        not_ext_path, ext = os.path.splitext(filepath)
        name = not_ext_path.split("/")[-1]
        depth_path = os.path.join(os.path.dirname(filepath), "..", "depth", "{}_depth{}".format(name, ext))
        return self._preprocess(filepath, depth_path, use_norm=False)

    def get_data(self):
        """工具链量化时内部调用预处理的函数，获取校准数据
            不需要norm归一化，工具链会根据配置文件，内部进行norm
        :return:
        """
        for i in range(self._calib_num):
            in_data = dict()
            for _input in self._inputs:
                in_data[_input["name"]] = self._preprocess(
                    self._data_lists[i][0], self._data_lists[i][1], use_norm=False)
            yield in_data
