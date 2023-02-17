#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: base_tyexec.py
@time: 2022/12/14
@Author  : xingwg
@software: PyCharm 
"""
import os
import abc
import cv2
import importlib
import numpy as np
from utils import logger
from utils.enum_type import PaddingMode
from utils.preprocess import calc_padding_size, default_preprocess
from collections import namedtuple, OrderedDict

# Input = namedtuple("Input", ["idx", "name", "shape", "mean", "std", "layout", "pixel_format", "resize_type",
#                              "padding_mode", "padding_value", "enable_aipp", "support"])


class BaseTyExec(object, metaclass=abc.ABCMeta):
    """base tyexec"""

    def __init__(self, cfg: dict):
        """init"""
        self.cfg = cfg
        self.model_dir = ""
        self.result_dir = ""
        self.quant_cfg = cfg["build"]["quant"]
        self.target = cfg["build"]["target"]
        self.enable_quant = cfg["build"]["enable_quant"]
        self.enable_build = cfg["build"]["enable_build"]
        self.enable_dump = cfg["build"]["enable_dump"]
        self.framework = cfg["model"]["framework"]
        self.custom_preprocess_module = self.quant_cfg["custom_preprocess_module"]
        self.custom_preprocess_cls = self.quant_cfg["custom_preprocess_cls"]
        self.graph = cfg["model"]["graph"]
        self.weight = cfg["model"]["weight"]
        self.inputs = cfg["model"]["inputs"]
        # self.inputs_ = list()
        self.outputs = cfg["model"]["outputs"]
        self.num_inputs = len(self.inputs)
        self.num_outputs = len(self.outputs)
        self.relay_quant = None
        self.params_quant = None
        self.relay = None
        self.params = None
        self.model_name = "net_combine"  # default

        self.model_dir = os.path.join(self.cfg["model"]["save_dir"], self.target)
        self.result_dir = os.path.join(self.model_dir, "result")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        logger.info("model output dir -> {}".format(self.model_dir))
        self.quant_json_path = os.path.join(self.result_dir, "model_quant.json")
        self.quant_model_path = os.path.join(self.result_dir, "model_quant.model")
        self.original_json_path = os.path.join(self.result_dir, "model_original.json")
        self.original_model_path = os.path.join(self.result_dir, "model_original.model")
        self.cpu_model_float_path = os.path.join(self.model_dir, "cpu_model_float.ty")
        self.cpu_model_fixed_path = os.path.join(self.model_dir, "cpu_model_fixed.ty")

        self.shape_dict = dict()
        self.dtype_dict = dict()

        self.set_suppress_long_func()  # 限制每个融合的最大算子个数
        self.set_model_name()
        self.set_input_infos()
        self.set_custom_preprocess()

        self.backend = "chip"

    @staticmethod
    def set_env():
        raise NotImplementedError

    def get_dataset(self):
        quant_data_dir = self.quant_cfg["data_dir"]
        dataset = quant_data_dir
        if not quant_data_dir:  # 未配置量化路径使用随机数据情况
            dataset = self.gen_random_quant_data
        else:
            if self.has_custom_preprocess:  # 配置量化数据目录情况下存在自定义预处理
                dataset = self.custom_preprocess_cls.get_data
        return dataset

    def set_suppress_long_func(self):
        if self.target.startswith("nnp4"):
            logger.warning("Nnp4xx not support set_suppress_long_func")
            return

        if "suppress_long_func" not in self.cfg["build"]:
            self.cfg["build"]["suppress_long_func"] = False
        else:
            if self.cfg["build"]["suppress_long_func"] is None:
                self.cfg["build"]["suppress_long_func"] = False

    def set_model_name(self):
        if "name" in self.cfg["model"]:
            if self.cfg["model"]["name"]:
                self.model_name = self.cfg["model"]["name"]
                return
        logger.warning("Not set model name, default name -> {}".format(self.model_name))

    def set_input_infos(self):
        for idx, _input in enumerate(self.inputs):
            shape = _input["shape"]
            n, c, h, w = shape
            if _input["layout"] == "NHWC":
                n, h, w, c = shape

            if "dtype" not in _input:
                if _input["pixel_format"] == "None":
                    _input["dtype"] = "float32"
                else:
                    _input["dtype"] = "uint8"

            self.shape_dict[_input["name"]] = (n, c, h, w)
            self.dtype_dict[_input["name"]] = _input["dtype"]

            _input["support"] = False if "None" == _input["pixel_format"] else True
            if _input["support"]:
                if "enable_aipp" not in _input:
                    _input["enable_aipp"] = True
                elif _input["enable_aipp"] is None:
                    _input["enable_aipp"] = True

                if "uint8" != _input["dtype"]:  # AIPP目前限制输出必须是uint8
                    logger.warning("input[{}] cannot enable aipp -> pixel_format: {}, dtype: {}".format(
                        _input["name"], _input["pixel_format"], _input["dtype"]))
                    _input["enable_aipp"] = False
            else:
                _input["enable_aipp"] = False

            if self.target.startswith("nnp4"):  # 4xx不支持aipp
                _input["enable_aipp"] = False

            if not _input["mean"]:
                _input["mean"] = [0.0 for _ in range(c)]
            if not _input["std"]:
                _input["std"] = [1.0 for _ in range(c)]

            _input["padding_mode"] = PaddingMode.LEFT_TOP if _input["padding_mode"] == 0 else PaddingMode.CENTER

        # 检查多输入配置是否正确，uint8图像必须排列在最前面
        if len(self.inputs) >= 2:
            for idx, _input in enumerate(self.inputs):
                if _input["enable_aipp"] and idx > 0:
                    # 检查之前的输入是否存在非图像数据
                    if not self.inputs[idx-1]["enable_aipp"]:
                        logger.error("Not support input(disable_aipp) in front of input(enable_aipp)")
                        exit(-1)

    @property
    def has_custom_preprocess(self):
        return True if self.custom_preprocess_cls else False

    def set_custom_preprocess(self):
        """检查是否存在自定义预处理
         1.多输入情况需要自定义
         2.默认预处理不能满足的情况
        """
        # 自定义预处理
        if self.custom_preprocess_cls:
            m = importlib.import_module(self.custom_preprocess_module)
            if hasattr(m, self.custom_preprocess_cls):
                # 实例化预处理对象
                self.custom_preprocess_cls = getattr(m, self.custom_preprocess_cls)(
                    self.inputs, self.quant_cfg["prof_img_num"], self.quant_cfg["data_dir"])
            else:
                logger.error("{}.py has no class named {}".format(
                    self.custom_preprocess_module, self.custom_preprocess_cls))
                exit(-1)

    def gen_random_quant_data(self):
        prof_img_num = self.quant_cfg["prof_img_num"]
        for _ in range(prof_img_num):
            yield self.get_datas(force_random=True)

    @staticmethod
    def check_not_exist(filepath):
        if not os.path.exists(filepath):
            logger.error("Not found filepath -> {}".format(filepath))
            exit(-1)

    @staticmethod
    def check_dtype(name, data, target_dtype):
        if data.dtype != target_dtype:
            logger.error("input({}) dtype mismatch {} vs {}".format(name, data.dtype, target_dtype))
            exit(-1)

    def get_datas(self, filepath="", force_float=False, force_cr=True, force_random=False, to_file=False):
        """ 生成模型输入数据
        @param filepath:  外部指定数据
        @param force_float:  强制输出float数据
        @param force_cr:　是否强制使能CR
        @param force_random:  是否强制使用随机数据，主要用于生成量化数据
        @param to_file:
        @return:
        """
        in_datas = OrderedDict()  # 保证输入顺序一致
        for idx, _input in enumerate(self.inputs):
            name = _input["name"]
            pixel_format = _input["pixel_format"]
            dtype = _input["dtype"]
            if _input["enable_aipp"]:
                dtype = "uint8"
            if force_float:
                dtype = "float32"
            data_path = _input["data_path"] if not filepath else filepath
            data_npy_path = os.path.join(self.result_dir, "{}_{}_{}.npy".format(idx, name, dtype))
            data_bin_path = os.path.join(self.result_dir, "{}_{}_{}.bin".format(idx, name, dtype))
            data_txt_path = os.path.join(self.result_dir, "{}_{}_{}.txt".format(idx, name, dtype))
            n, c, h, w = self.shape_dict[name]
            if _input["support"]:   # 图像数据，工具内部处理
                if data_path:   # 指定输入数据
                    im = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE if pixel_format == "GRAY" else cv2.IMREAD_COLOR)
                else:   # 未指定输入数据，生成随机图像
                    logger.warning("input[{}] will use random image".format(name))
                    if force_random:  # 用于量化和统计含零情况
                        im = np.random.randint(low=0, high=255, size=(h, w, c), dtype="uint8")
                    else:
                        random_im_path = os.path.join(self.result_dir, "{}_{}_random.jpg".format(idx, name))
                        if os.path.exists(random_im_path):
                            im = cv2.imread(random_im_path, cv2.IMREAD_GRAYSCALE if pixel_format == "GRAY" else cv2.IMREAD_COLOR)
                        else:
                            im = np.random.randint(low=0, high=255, size=(h, w, c), dtype="uint8")
                            cv2.imwrite(random_im_path, im)
                if not _input["enable_aipp"] or force_cr:  # 兼容芯片orISS使能AIPP情况
                    _input["padding_size"], _ = calc_padding_size(im, (w, h), _input["padding_mode"])
                    in_datas[name] = default_preprocess(
                        im,
                        (w, h),
                        mean=_input["mean"],
                        std=_input["std"],
                        use_norm=False if not force_float else True,  # 量化前relay_func需要norm
                        use_rgb=True if pixel_format == "RGB" else False,
                        use_resize=True,
                        resize_type=_input["resize_type"],
                        padding_value=_input["padding_value"],
                        padding_mode=_input["padding_mode"]
                    ).astype(dtype=dtype if not force_float else "float32")  # 量化前relay_func需要float输入，也可不强转由tvm自定转换
                else:
                    if pixel_format == "RGB":
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # AIPP不支持BGR -> RGB，提前转换
                    if len(im.shape) not in [2, 3]:
                        logger.error("Not support image shape -> {}".format(im.shape))
                        exit(-1)
                    if len(im.shape) == 2:  # gray
                        im = np.expand_dims(im, axis=2)
                        im = np.expand_dims(im, axis=0)
                    else:
                        im = np.expand_dims(im, axis=0)
                    in_datas[name] = im.transpose((0, 3, 1, 2))  # nhwc -> nchw, BGR888  uint8
            else:  # 非图像数据，由自定义模块处理或随机生成
                assert not _input["enable_aipp"], "non-image cannot enable AIPP"
                if data_path:   # 指定输入数据
                    if not self.has_custom_preprocess:
                        logger.error("Not set custom preprocess")
                        exit(-1)
                    in_datas[name] = self.custom_preprocess_cls.get_single_data(data_path, idx)
                    self.check_dtype(name, in_datas[name], dtype)
                else:   # 未指定输入数据
                    logger.warning("input[{}] will use random data".format(name))
                    if not force_random and os.path.exists(data_npy_path):
                        in_datas[name] = np.load(data_npy_path)
                        logger.warning("load random data -> {}".format(data_npy_path))
                    else:
                        if dtype == "float32":
                            in_datas[name] = np.random.rand(n, c, h, w).astype(dtype=dtype)   # 数值范围[0, 1)
                        elif dtype == "float16":
                            in_datas[name] = np.random.rand(n, c, h, w).astype(dtype=dtype)   # 数值范围[0, 1)
                        elif dtype == "int16":
                            in_datas[name] = np.random.randint(low=-(2**15), high=2**15-1, size=(n, c, h, w), dtype=dtype)
                        elif dtype == "uint8":
                            in_datas[name] = np.random.randint(low=0, high=255, size=(n, c, h, w), dtype=dtype)
            if to_file:
                data = in_datas[name].copy()
                np.save(data_npy_path, data)
                data.tofile(data_bin_path)
                data.tofile(data_txt_path, sep="\n")
                logger.info("save data -> {}".format(data_npy_path))
        return in_datas

    def set_quantization_cfg(self, in_datas):
        """ quantization config
        @param in_datas:
        @return: quantize_config
        """
        in_dtypes, norm = dict(), dict()
        for idx, _input in enumerate(self.inputs):
            name = _input["name"]
            if in_datas[name].dtype == np.uint8:
                data_type = "uint8"
            elif in_datas[name].dtype == np.int16:
                data_type = "int16"
            elif in_datas[name].dtype == np.float16:
                data_type = "float16"
            elif in_datas[name].dtype == np.float32:
                data_type = "float32"
            else:
                logger.error("Not support input dtype -> {}".format(in_datas[name].dtype))
                exit(-1)
            # 与最终量化后的模型输入数据类型相对应
            in_dtypes[name] = data_type
            norm[name] = {"mean": _input["mean"], "std": _input["std"], "axis": 1}
            logger.info("Input({}) dtype -> {}".format(name, in_dtypes[name]))
            logger.info("Input({}) mean/std -> {}".format(name, norm[name]))

        import tvm
        from tvm import relay
        quantize_config = tvm.relay.quantization.get_quantize_config(self.target, in_dtypes)
        quantize_config["calib_method"] = self.quant_cfg["calib_method"]

        quantize_config["float_list"] = list()
        if self.quant_cfg["skip_layer_idxes"]:
            quantize_config["float_list"].extend(self.quant_cfg["skip_layer_idxes"])
        if self.quant_cfg["skip_layer_types"]:
            quantize_config["float_list"].extend(self.quant_cfg["skip_layer_types"])
        return quantize_config, norm

    @staticmethod
    def save_relay_to_json(quant_json_path, relay_func, params):
        """ save relay to json file
        @param quant_json_path:
        @param relay_func:
        @param params:
        @return:
        """
        import tvm
        from tvm import relay
        tvm.relay.quantization.save_ir_to_json(quant_json_path, relay_func, params)
        logger.info("save relay to {}".format(quant_json_path))

    @staticmethod
    def save_relay_to_model(quant_model_path, relay_func, params):
        """ save relay to model, can be visualized by netron
        @param quant_model_path:
        @param relay_func:
        @param params:
        @return:
        """
        raise NotImplementedError

    @staticmethod
    def load_relay_from_json(json_path):
        """ load relay from json file
        @param json_path:
        @return:
        """
        import tvm
        from tvm import relay
        relay_func = tvm.relay.quantization.get_ir_from_json(json_path)
        return relay_func

    def iss_dump_output(self, in_datas):
        """ dump result layer by layer
        @param in_datas:
        @return:
        """
        raise NotImplementedError

    def save_compare_layer_outputs(self):
        """tvm float vs fixed """
        raise NotImplementedError

    @abc.abstractmethod
    def quantization(self, in_datas):
        """relay quantization"""
        raise NotImplementedError

    @abc.abstractmethod
    def build(self, in_datas):
        """relay build"""
        raise NotImplementedError

    @property
    def target_ops(self):
        return ["nn.conv2d", "nn.dense", "nn.batch_matmul"]

    def compress_analysis(self):
        raise NotImplementedError

    def model_analysis(self):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def build_x86_64(relay_func, params, save_path=""):
        """compile cpu model"""
        raise NotImplementedError

    @abc.abstractmethod
    def iss_fixed_inference(self, in_datas, to_file=False):
        """tvm iss fixed"""
        raise NotImplementedError

    def sdk_iss_fixed_inference(self):
        raise NotImplementedError

    @staticmethod
    def tvm_inference(engine: object, in_datas: dict) -> list:
        """
        @param engine: tvm.contrib.graph_executor
        @param in_datas:
        @return:
        """
        try:
            from tvm.contrib.graph_runtime import GraphModule
        except:
            from tvm.contrib.graph_executor import GraphModule
        if not isinstance(engine, GraphModule):
            logger.error("Infer engine{} not match -> {}".format(
                type(engine), "tvm.contrib.graph_executor.GraphModule"))
            exit(-1)
        engine.set_input(**in_datas)
        engine.run()
        outputs = list()
        for idx in range(engine.get_num_outputs()):
            outputs.append(engine.get_output(idx).asnumpy())
        return outputs

    @abc.abstractmethod
    def tvm_fixed_inference(self, in_datas, to_file=False):
        """tvm cpu fixed"""
        raise NotImplementedError

    def infer(self):
        """ inference on chip/sdk_iss """
        raise NotImplementedError

    def profile(self):
        raise NotImplementedError

    @abc.abstractmethod
    def tvm_float_inference(self, in_datas, to_file=False):
        """tvm cpu float"""
        raise NotImplementedError

    def get_relay_mac(self):
        """get relay func MAC count"""
        raise NotImplementedError

    def get_device_type(self):
        """get op run device"""
        raise NotImplementedError

    def get_version(self):
        """get tytvm version"""
        raise NotImplementedError

    def get_profile_info(self):
        """get tvm profile info"""
        raise NotImplementedError

    def x2relay(self):
        """caffe/onnx/tensorflow/pytorch/mxnet to relay func"""
        if self.framework == "caffe":
            self.caffe2relay()
        elif self.framework == "onnx":
            self.onnx2relay()
        elif self.framework == "pytorch":
            self.pytorch2relay()
        elif self.framework == "mxnet":
            self.mxnet2relay()
        elif self.framework == "tensorflow":
            self.tensorflow2relay()
        elif self.framework == "tflite":
            self.tflite2relay()
        elif self.framework == "tflite-qnn":
            self.tflite_qnn2relay()
        else:
            logger.error("Not support framework -> {}".format(self.framework))
            exit(-1)

        self.save_relay_to_json(self.original_json_path, self.relay, self.params)
        self.save_relay_to_model(self.original_model_path, self.relay, self.params)

    def caffe2relay(self):
        raise NotImplementedError

    @abc.abstractmethod
    def onnx2relay(self):
        raise NotImplementedError

    def pytorch2relay(self):
        raise NotImplementedError

    def mxnet2relay(self):
        raise NotImplementedError

    def tensorflow2relay(self):
        raise NotImplementedError

    def tflite2relay(self):
        raise NotImplementedError

    def tflite_qnn2relay(self):
        raise NotImplementedError
