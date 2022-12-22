#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: base_tyexec.py
@time: 2022/12/14
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
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

        self.set_model_name()
        self.set_input_infos()
        self.set_custom_preprocess()
        self.model_path = os.path.join(self.model_dir, "{}.ty".format(self.model_name))
        self.model_path_aarch64 = os.path.join(self.model_dir, "{}_aarch64.ty".format(self.model_name))

        # sdk
        # self.sdk = None
        # self.sdk_enable_aipp = False
        # self.sdk_enable_dump = False
        # self.engine = None
        # self.prefix = "chip"
        # self.sdk_dump_root_path = ""
        # self.total = 0
        # self.time_span = 0
        #
        # self.phase = "build"

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

            _input["shape"] = (n, c, h, w)
            self.shape_dict[_input["name"]] = (n, c, h, w)
            self.dtype_dict[_input["name"]] = "float32"

            _input["support"] = False if "None" == _input["pixel_format"] or "None" == _input["layout"] else True
            if _input["support"]:
                if "enable_aipp" not in _input:
                    _input["enable_aipp"] = True
            else:
                _input["enable_aipp"] = False

            if not _input["mean"]:
                _input["mean"] = [0.0 for _ in range(c)]
            if not _input["std"]:
                _input["std"] = [0.0 for _ in range(c)]

            _input["padding_mode"] = PaddingMode.LEFT_TOP if _input["padding_mode"] == 0 else PaddingMode.CENTER

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

    def get_datas(self, filepath="", use_norm=False, force_cr=False, to_file=True):
        """获取tvm float/tvm fixed/iss fixed仿真的数据
        @param filepath: 可指定图片输入
        @param use_norm: 是否归一化，仅tvm float
        @param force_cr: 是否强制进行cvtColor、resize，仅disable_aipp、tvm fixed
        @param to_file:  是否保存数据
        @return:
        """
        in_datas = OrderedDict()  # 保证输入顺序一致
        for idx, _input in enumerate(self.inputs):
            data_path = _input["data_path"] if not filepath else filepath
            n, c, h, w = _input["shape"]
            if data_path:
                if not os.path.exists(data_path):
                    logger.error("Not found data_path -> {}".format(data_path))
                    return None
                if not _input["support"]:
                    # not support will use custom preprocess
                    in_datas[_input["name"]] = self.custom_preprocess_cls.get_single_data(data_path, idx)
                else:
                    # default preprocess，only image channel 1 or 3
                    im = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE if _input["pixel_format"] == "GRAY" else cv2.IMREAD_COLOR)
                    if (not _input["enable_aipp"]) or force_cr:
                        _input["padding_size"], _ = calc_padding_size(im, (w, h), _input["padding_mode"])
                        in_datas[_input["name"]] = default_preprocess(
                            im,
                            (w, h),
                            mean=_input["mean"],
                            std=_input["std"],
                            use_norm=use_norm,
                            use_rgb=True if _input["pixel_format"] == "RGB" else False,
                            use_resize=True,
                            resize_type=_input["resize_type"],
                            padding_value=_input["padding_value"],
                            padding_mode=_input["padding_mode"]
                        )
                    else:
                        if _input["pixel_format"] == "RGB":
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # AIPP不支持BGR->RGB，需提前转换

                        if len(im.shape) not in [2, 3]:
                            logger.error("Not support image shape -> {}".format(im.shape))
                            exit(-1)
                        if len(im.shape) == 2:  # gray
                            im = np.expand_dims(im, axis=2)
                            im = np.expand_dims(im, axis=0)
                        else:
                            im = np.expand_dims(im, axis=0)
                        in_datas[_input["name"]] = im.transpose((0, 3, 1, 2))  # hwc -> chw, BGR888
            else:
                # random data
                logger.warning("Not set data_path, will use random data")
                if not _input["support"]:
                    in_datas[_input["name"]] = self.custom_preprocess_cls.get_single_data("", idx)
                else:
                    if use_norm:
                        in_datas[_input["name"]] = np.random.randn(n, c, h, w).astype(dtype=np.float32)
                    else:
                        in_datas[_input["name"]] = np.random.randint(0, 255, (n, c, h, w)).astype(dtype=np.uint8)

                _input["padding_size"] = None  #

            if to_file:
                # save data
                in_datas[_input["name"]].tofile(os.path.join(self.result_dir, "data_{}_CRN.bin".format(idx)))
                in_datas[_input["name"]].tofile(os.path.join(self.result_dir, "data_{}_CRN.txt".format(idx)), sep="\n")

        return in_datas

    def set_quantization_cfg(self, in_datas):
        """ quantization config
        @param in_datas:
        @return: quantize_config
        """
        in_dtypes, norm = dict(), dict()
        for idx, _input in enumerate(self.inputs):
            name = _input["name"]
            data_type = "uint8"
            if in_datas[name].dtype == np.uint8:
                pass
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
            in_dtypes[name] = data_type if self.has_custom_preprocess else "uint8"
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
    def load_relay_from_json(quant_json_path):
        """ load relay from json file
        @param quant_json_path:
        @return:
        """
        import tvm
        from tvm import relay
        relay_func = tvm.relay.quantization.get_ir_from_json(quant_json_path)
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

    def profile(self):
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

    # def load(self, net_cfg_file, sdk_cfg_file, model_path, enable_aipp, enable_dump):
    #     """sdk load netbin for chip/iss inference
    #     @param net_cfg_file: ip config
    #     @param sdk_cfg_file: sdk init config
    #     @param model_path:  netbin path
    #     @param enable_aipp:
    #     @param enable_dump:
    #     @return:
    #     """
    #     raise NotImplementedError
    #
    # def infer(self, in_datas, to_file=False):
    #     """sdk chip/iss inference
    #     @param in_datas:
    #     @param to_file:
    #     @return:
    #     """
    #     raise NotImplementedError
    #
    # def unload(self):
    #     """unload netbin"""
    #     raise NotImplementedError
    #
    # def dump_profile(self):
    #     raise NotImplementedError
    #
    # def compare_layer_out(self):
    #     raise NotImplementedError
