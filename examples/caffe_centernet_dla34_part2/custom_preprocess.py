
import numpy as np
from base.base_custom_preprocess import BaseCustomPreprocess


class CustomCenterNetDLA34Part2(BaseCustomPreprocess):
    """自定义预处理模块，可以自由定义，成员函数get_data为固定入口"""
    def __init__(self, inputs, calib_num, data_dir):
        """init
        :param input_infos: 模型输入信息
        :param calib_num:  校准数据量
        :param data_dir:  校准数据目录
        """
        self._inputs = inputs
        self._calib_num = calib_num
        self._data_dir = data_dir
        self._dtype = np.uint8 if self._inputs[0]["dtype"] == "uint8" else np.float32

    def get_single_data(self, filepath):
        """用于处理模型输入图片的预处理
        :param filepath:
        :return:
        """
        for _input in self._inputs:
            if _input["img_path"] != filepath:
                continue
            # 预处理
            n, c, h, w = _input["shape"]
            if _input["layout"] == "NHWC":
                n, c, h, w = _input["shape"][0], _input["shape"][3], _input["shape"][1], _input["shape"][2]
            return np.random.randn(n, c, h, w).astype(self._dtype)

    def get_data(self):
        """工具链内部调用预处理来校准的函数
        :return:
        """
        for i in range(self._calib_num):
            datas = dict()
            for _input in self._inputs:
                n, c, h, w = _input["shape"]
                if _input["layout"] == "NHWC":
                    n, c, h, w = _input["shape"][0], _input["shape"][3], _input["shape"][1], _input["shape"][2]
                datas[_input["name"]] = np.random.randn(n, c, h, w).astype(self._dtype)
            yield datas
