
import os
import cv2
import numpy as np
from base.base_custom_preprocess import BaseCustomPreprocess


class CustomTensorFlowInput4(BaseCustomPreprocess):
    """自定义预处理模块，可以自由定义，成员函数get_data为固定入口
    """
    def __init__(self, inputs: list, calib_num: int, data_dir: str):
        """
        :param inputs: 模型输入信息
        :param calib_num:  校准数据量
        :param data_dir:  校准数据目录
        """
        self._inputs = inputs
        self._calib_num = calib_num
        self._data_dir = data_dir

        self._dtypes = [np.int16, np.float32, np.float32, np.uint8]

        input_path0 = "data/iss_in0_1x256x256"
        input_path1 = "data/iss_in1_1x64x64"
        input_path2 = "data/iss_in2_1x64x64"
        input_path3 = "data/iss_in3_4x1x1"

        input0 = np.fromfile(input_path0, dtype=np.int16).reshape(1, 1, 256, 256)
        input3 = np.fromfile(input_path3, dtype=np.float32).reshape(1, 4, 1, 1)
        input2 = np.fromfile(input_path2, dtype=np.float32).reshape(1, 1, 64, 64)
        input1 = np.fromfile(input_path1, dtype=np.uint8).reshape(1, 1, 64, 64)

        self._data_lists = [input3, input0, input1, input2]

    def get_single_data(self, filepath, idx):
        return self._data_lists[idx]

    def get_data(self):
        dtypes = [np.int16, np.float32, np.float32, np.uint8]
        for i in range(self._calib_num):
            datas = dict()
            for d, _input in enumerate(self._inputs):
                n, c, h, w = _input["shape"]
                if _input["layout"] == "NHWC":
                    n, h, w, c = _input["shape"]
                if d == 0 or d == 3:
                    datas[_input["name"]] = np.random.randint(low=5, high=255, size=(n, c, h, w)).astype(dtype=self._dtypes[d])
                else:
                    datas[_input["name"]] = np.random.rand(n, c, h, w).astype(dtype=self._dtypes[d])
            yield datas
