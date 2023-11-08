
import numpy as np
from base.base_custom_preprocess import BaseCustomPreprocess


class CustomPreprocessMultiDims(BaseCustomPreprocess):
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

        self._data_lists = list()
        for _, _input in enumerate(self._inputs):
            shape = _input["shape"]
            self._data_lists.append(np.random.random(shape).astype(np.float32))

    def get_single_data(self, filepaths: list, idx, use_norm):
        return self._data_lists[idx]

    def get_data(self):
        for i in range(self._calib_num):
            datas = dict()
            for d, _input in enumerate(self._inputs):
                shape = _input["shape"]
                input_name = _input["name"]
                datas[input_name] = np.random.random(shape).astype(np.float32)
            yield datas
