
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

    def get_single_data(self, filepath, idx):
        """用于处理模型输入图片的预处理
        :param filepath:
        :param idx:  不支持输入的索引
        :return:
        """
        n, c, h, w = self._inputs[idx]["shape"]
        if self._inputs[idx]["layout"] == "NHWC":
            n, h, w, c = self._inputs[idx]["shape"]
        return np.random.randn(n, c, h, w).astype(dtype=np.float32)

    def get_data(self):
        """工具链内部调用预处理来校准的函数
        :return:
        """
        for i in range(self._calib_num):
            datas = dict()
            for idx, _input in enumerate(self._inputs):
                datas[_input["name"]] = self.get_single_data("", idx)
            yield datas
