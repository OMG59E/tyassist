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
import time
import importlib
import numpy as np
from utils import logger
from utils.enum_type import PaddingMode
from utils.preprocess import calc_padding_size, default_preprocess
from collections import OrderedDict


class BaseTyExec(object, metaclass=abc.ABCMeta):
    """base tyexec"""

    def __init__(self, cfg: dict):
        """init"""
        self.model_dir = ""
        self.result_dir = ""
        
        self.cfg = cfg
        self.build_cfg = cfg["build"]
        self.quant_cfg = cfg["build"]["quant"]
        self.model_cfg = cfg["model"]
        self.inputs = cfg["model"]["inputs"]
        self.outputs = self.model_cfg.get("outputs")
        self.num_inputs = len(self.inputs)
        # self.num_outputs = len(self.outputs)
        
        self.target = self.build_cfg["target"]
        self.enable_quant = self.build_cfg.get("enable_quant", True)
        self.enable_build = self.build_cfg.get("enable_build", True)
        self.enable_dump = self.build_cfg.get("enable_dump", 0)

        self.custom_preprocess_module = self.quant_cfg.get("custom_preprocess_module")
        self.custom_preprocess_cls = self.quant_cfg.get("custom_preprocess_cls")
        
        self.framework = self.model_cfg["framework"]
        self.weight = self.model_cfg["weight"]
        self.graph = self.model_cfg.get("graph")
                
        self.quant_data_dir = self.quant_cfg.get("data_dir")
        self.prof_img_num = self.quant_cfg["prof_img_num"]
        self.build_opt_level = self.build_cfg.get("opt_level", 0)
        self.quant_opt_level = self.quant_cfg.get("opt_level", 0)
        self.quant_debug_level = self.quant_cfg.get("debug_level", -1)
        self.quant_calib_method = self.quant_cfg.get("calib_method", "l2norm")
        self.disable_pass = self.quant_cfg.get("disable_pass")
        self.similarity_img_num = self.quant_cfg.get("similarity_img_num", 1)
        self.similarity_dataset = self.quant_cfg.get("similarity_dataset")
        
        self.relay_quant = None
        self.params_quant = None
        self.relay = None
        self.params = None
        
        self.model_name = self.model_cfg.get("name")
        if not self.model_name:
            self.model_name = "net_combine"
            logger.warning("Not set model name, default name -> {}".format(self.model_name))
            
        self.save_dir = self.model_cfg.get("save_dir")
        if not self.save_dir:
            self.save_dir = "outputs"
        self.model_dir = os.path.join(self.save_dir, self.target)
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
        
        self.bs = 1  # 默认bs为1，TODO 需要考虑多输入不同batch情况

        self.set_custom_preprocess()
        self.set_input_infos()
        
        self.backend = "chip"   # 默认后端为chip

        self.quantization_span = 0
        self.build_span = 0
        self.iss_simu_span = 0
        self.tvm_layerwise_dump_span = 0
        self.iss_layerwise_dump_span = 0
        self.tvm_float_simu_span = 0
        self.tvm_fixed_simu_span = 0
        self.x2relay_span = 0

        self.is_qnn = True if self.model_cfg["framework"] == "onnx-qnn" else False

        if not self.quant_data_dir:
            # 不配置量化数据目录的情况下，将使用随机数据，这时prof_img_num必须指定具体数值
            assert self.prof_img_num > 0, "Random data mode, prof_img_num must be > 0"
        else:
            # 单输入且可被内置预处理支持
            if len(self.inputs) == 1 and self.inputs[0]["support"]:
                assert self.prof_img_num >= 0, "custom/data_dir mode, prof_img_num must be >= 0"
                if isinstance(self.quant_data_dir, str):
                    img_lists = os.listdir(self.quant_data_dir)
                    prof_img_num = len(img_lists) // self.bs
                    # 根据实际数据量，更新prof_img_num
                    if self.prof_img_num == 0:
                        self.prof_img_num = prof_img_num
                    elif prof_img_num < self.prof_img_num:
                        logger.warning("Quant data not enough, and update prof_img_num {} -> {}".format(
                            self.prof_img_num, prof_img_num))
                        self.prof_img_num = prof_img_num

    def set_input_infos(self):
        # 更新输入配置信息
        for idx, _input in enumerate(self.inputs):
            input_name = _input["name"]
            shape = _input["shape"]
            mean = _input.get("mean")
            if mean:
                norm_axis = _input.get("norm_axis")
                if not norm_axis:
                    _input["norm_axis"] = 1
                norm_axis = _input["norm_axis"]
                dim = shape[norm_axis]
                if len(_input.get("mean")) == 1:  # broadcast
                    val = _input.get("mean")[0]
                    _input["mean"] = [val for _ in range(dim)]
                if len(_input.get("std")) == 1:   # broadcast
                    val = _input.get("std")[0]
                    _input["std"] = [val for _ in range(dim)]
            else:
                _input["mean"] = None
                _input["std"] = None

            layout = _input["layout"]
            if layout in ["NCHW", "NHWC"]:
                n, c, h, w = shape
                if layout == "NHWC":
                    n, h, w, c = shape
                shape = (n, c, h, w)
                
            self.shape_dict[input_name] = shape
            self.bs = shape[0]
            if idx > 0:
                assert self.bs == self.inputs[idx - 1]["shape"][0], "all input batch size must be same"
            
            pixel_format = _input["pixel_format"]                
            if not _input.get("dtype"):
                # 用户未指定输入数据类型的情况下，非图像默认为float32，图像默认uint8
                if pixel_format == "None":
                    _input["dtype"] = "float32"
                else:
                    _input["dtype"] = "uint8"

            self.dtype_dict[input_name] = _input["dtype"]
            
            # 判断输入是否能够被支持
            _input["support"] = True
            # 存在自定义优先走自定义处理          
            if self.has_custom_preprocess or pixel_format in ["None"] or len(self.inputs) > 1:
                _input["support"] = False
            
            if _input["support"]:
                # 检查预处理参数配置
                resize_type = _input.get("resize_type", 0)
                padding_value = _input.get("padding_value", 128)
                padding_mode = _input.get("padding_mode", 0)
                _input["resize_type"] = resize_type
                _input["padding_value"] = padding_value
                _input["padding_mode"] = PaddingMode.LEFT_TOP if padding_mode == 0 else PaddingMode.CENTER
            
            _input["enable_aipp"] = _input.get("enable_aipp", False)
            # 该输入内置预处理支持，则可使能AIPP
            if _input["support"] and _input["dtype"] == "uint8":
                _input["enable_aipp"] = True
                
            # 4xx没有CR模块
            if self.target.startswith("nnp4"):  
                _input["enable_aipp"] = False

        # 3xx检查多输入配置是否正确，多输入使能AIPP的情况下，图像数据必须排列在最前面
        if len(self.inputs) >= 2 and self.target.startswith("nnp3"):
            for idx, _input in enumerate(self.inputs):
                if _input["enable_aipp"] and idx > 0:
                    # 检查之前的输入是否存在非图像数据
                    if not self.inputs[idx - 1]["enable_aipp"]:
                        logger.error("Not support input(tensor) in front of input(image)")
                        exit(-1)

    @property
    def has_custom_preprocess(self):
        return True if self.custom_preprocess_cls else False

    def set_custom_preprocess(self):
        """import自定义预处理模块
        需要自定义预处理的情况
        1.pixel_format为None, 表示输入为tensor数据
        2.输入为图像数据, 但内置预处理不支持
        """
        # 自定义预处理
        if self.custom_preprocess_cls:
            m = importlib.import_module(self.custom_preprocess_module)
            if hasattr(m, self.custom_preprocess_cls):
                # 实例化预处理对象
                self.custom_preprocess_cls = getattr(m, self.custom_preprocess_cls)(
                    self.inputs, self.prof_img_num, self.quant_data_dir)
            else:
                logger.error("{}.py has no class named {}".format(
                    self.custom_preprocess_module, self.custom_preprocess_cls))
                exit(-1)

    @staticmethod
    def set_env():
        raise NotImplementedError

    def _batch_preprocess(self):
        assert len(self.inputs) == 1, "input_num must be = 1"
        data_dir = self.quant_data_dir
        img_lists = os.listdir(data_dir)
        for idx in range(self.prof_img_num):
            batch_img_lists = [
                os.path.join(data_dir, img_lists[i]) for i in range(idx * self.bs, (idx + 1) * self.bs)]
            yield self.get_datas(
                batch_img_lists, use_norm=False, force_cr=True, force_random=False, to_file=False)

    def _gen_random_quant_data(self):
        np.random.seed(10086)
        for _ in range(self.prof_img_num):
            yield self.get_datas(force_random=True)
            
    def get_dataset(self):
        if not self.quant_data_dir:
            # 未配置量化路径使用随机数据情况
            dataset = self._gen_random_quant_data
        else:
            # 存在自定义预处理
            if self.has_custom_preprocess:
                # 自定义处理
                logger.info("There is a custom preprocess, the custom preprocess will be used")
                dataset = self.custom_preprocess_cls.get_data
            else:
                # 内置处理
                dataset = self._batch_preprocess
        return dataset

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
            
    @staticmethod
    def _gen_random_data(shape, layout, dtype):
        if dtype == "float32":
            _data = np.random.uniform(low=0, high=128, size=shape).astype(dtype=dtype)  # 数值范围[0, 128)
        elif dtype == "float16":
            _data = np.random.uniform(low=0, high=128, size=shape).astype(dtype=dtype)  # 数值范围[0, 128)
        elif dtype == "int16":
            _data = np.random.randint(low=-(2 ** 15), high=2 ** 15 - 1, size=shape, dtype=dtype)
        elif dtype == "uint8":
            _data = np.random.randint(low=0, high=255, size=shape, dtype=dtype)
        else:
            logger.error("Not support dtype -> {}".format(dtype))
            exit(-1)
        if layout in ["NHWC"]:
            _data = np.transpose(_data, (0, 3, 1, 2))   # NHWC -> NCHW
        return _data
                    
    def get_datas(self, filepath: str or list = "",
                  use_norm=False, force_cr=True, force_random=False, to_file=False):
        """ 生成模型输入数据
        @param filepath: 外部指定数据
        @param use_norm: 是否norm
        @param force_cr: 是否强制使能CR
        @param force_random: 是否强制使用随机数据，主要用于生成量化数据
        @param to_file:  是否保存数据
        @return:
        """
        in_datas = OrderedDict()  # 保证输入顺序一致
        for idx, _input in enumerate(self.inputs):
            name = _input["name"]
            pixel_format = _input["pixel_format"]
            layout = _input["layout"]
            dtype = _input["dtype"]
            shape = _input["shape"]
            enable_aipp = _input["enable_aipp"]
            support = _input["support"]
            if enable_aipp:
                dtype = "uint8"
                       
            data_paths = _input.get("data_path") if not filepath else filepath
            if isinstance(data_paths, list):
                assert len(data_paths) == self.bs
            else:
                if data_paths is None:
                    data_paths = ""
                assert isinstance(data_paths, str)
                if self.bs > 1:
                    logger.warning("data_path will be reused")
                data_paths = [data_paths for _ in range(self.bs)]

            if support:
                new_shape = shape
                new_shape[0] = 1
                shape_s = "x".join(list(map(str, new_shape)))
            else:
                shape_s = "x".join(list(map(str, shape)))
            data_npy_path = os.path.join(self.result_dir, "{}_{}_{}_{}.npy".format(idx, name.replace("/", "_"), dtype, shape_s))
            data_bin_path = os.path.join(self.result_dir, "{}_{}_{}_{}.bin".format(idx, name.replace("/", "_"), dtype, shape_s))
            data_txt_path = os.path.join(self.result_dir, "{}_{}_{}_{}.txt".format(idx, name.replace("/", "_"), dtype, shape_s))

            if support:  # 图像数据，工具内部处理
                n, c, h, w = shape
                if layout == "NHWC":
                    n, h, w, c = shape
                bs = n
                ims = list()
                for idx, data_path in enumerate(data_paths):
                    if data_path:  # 指定输入数据
                        # 检查data_path是否存在
                        if not os.path.exists(data_path):
                            logger.error("data_path not exist, data_path -> {}".format(data_path))
                            exit(-1)
                        im = cv2.imread(data_path, 
                                cv2.IMREAD_GRAYSCALE if pixel_format == "GRAY" else cv2.IMREAD_COLOR)
                        if im is None:
                            logger.error("data_path imread failed, data_path -> {}".format(data_path))
                            exit(-1)
                        ims.append(im)
                        continue

                    # 未指定输入数据，生成随机图像
                    # TODO 需要处理量化和验证数据为同一批的问题
                    logger.warning("The input[{}] will use random image, recommend make user data!".format(name, idx))
                    if force_random:  # 用于量化和统计含零情况
                        im = np.random.randint(low=0, high=255, size=(h, w, c), dtype="uint8")
                        ims.append(im)
                        continue
                    random_im_path = os.path.join(self.result_dir, "{}_{}_random.jpg".format(idx, name.replace("/", "_")))
                    random_npy_path = os.path.join(self.result_dir, "{}_{}_random.npy".format(idx, name.replace("/", "_")))
                    if os.path.exists(random_npy_path):
                        # 复用随机数据
                        im = np.load(random_npy_path)
                    else:
                        # 保存随机数据以便复用
                        im = np.random.randint(low=0, high=255, size=(h, w, c), dtype="uint8")
                        np.save(random_npy_path, im)
                        cv2.imwrite(random_im_path, im)
                    ims.append(im)

                # preprocess
                datas = list()
                for im in ims:
                    if not enable_aipp or force_cr:  # 兼容芯片orISS使能AIPP情况
                        _input["padding_size"], _ = calc_padding_size(im, (w, h), _input["padding_mode"])
                        in_data = default_preprocess(
                            im,
                            (w, h),
                            mean=_input["mean"],
                            std=_input["std"],
                            use_norm=False if not use_norm else True,  # 量化前relay_func需要norm
                            use_rgb=True if pixel_format == "RGB" else False,
                            use_resize=True,
                            resize_type=_input["resize_type"],
                            padding_value=_input["padding_value"],
                            padding_mode=_input["padding_mode"]
                        ).astype(dtype=dtype if not use_norm else "float32")  # 量化前relay_func需要float输入，也可不强转由tvm自定转换
                        datas.append(np.ascontiguousarray(in_data))
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
                        datas.append(np.ascontiguousarray(im.transpose((0, 3, 1, 2))))  # nhwc -> nchw, BGR888  uint8
                in_datas[name] = np.concatenate(datas, axis=0)
            else:  # 多输入or非图像数据or存在自定义情况
                assert not enable_aipp, "non-image cannot enable AIPP"
                exist = True  # 数据是否存在的标志
                for data_path in data_paths:
                    if not os.path.exists(data_path):
                        exist = False
                        break
                if exist:  # 指定输入数据
                    assert self.has_custom_preprocess, "Not set custom preprocess"
                    in_datas[name] = self.custom_preprocess_cls.get_single_data(data_paths, idx, use_norm)
                    self.check_dtype(name, in_datas[name], "float32" if use_norm else dtype)
                else:  # 未指定输入数据
                    logger.warning("The input[{}] will use random data, recommend make user data!".format(name))
                    if force_random:  # 用于量化和统计含零情况
                        in_datas[name] = self._gen_random_data(shape, layout, dtype)
                    else:
                        if os.path.exists(data_npy_path):
                            in_datas[name] = np.load(data_npy_path)
                        else:
                            in_datas[name] = self._gen_random_data(shape, layout, dtype)     
                    if use_norm:  
                        mean, std = _input["mean"], _input["std"]
                        in_datas[name] = in_datas[name].astype(dtype)
                        if mean:
                            if layout in ["NHWC", "NCHW"]:
                                dim = in_datas[name].shape[1]
                                mean_shape = [1, dim, 1, 1]
                            else:
                                norm_axis = _input["norm_axis"]
                                dim = in_datas[name].shape[norm_axis]
                                mean_shape = [1 for _ in range(len(shape))]
                                mean_shape[norm_axis] = dim
                            mean = np.array(mean, dtype=np.float32).reshape(mean_shape)
                            std = np.array(std, dtype=np.float32).reshape(mean_shape)
                            in_datas[name] = (in_datas[name] - mean) / std
            # 更新保存路径
            if use_norm:
                dtype = "float32"
                data_npy_path = os.path.join(self.result_dir, "{}_{}_{}_{}_norm.npy".format(idx, name.replace("/", "_"), dtype, shape_s))
                data_bin_path = os.path.join(self.result_dir, "{}_{}_{}_{}_norm.bin".format(idx, name.replace("/", "_"), dtype, shape_s))
                data_txt_path = os.path.join(self.result_dir, "{}_{}_{}_{}_norm.txt".format(idx, name.replace("/", "_"), dtype, shape_s))                            
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
            if _input["mean"]:
                norm[name] = {"mean": _input["mean"], "std": _input["std"], "axis": _input["norm_axis"]}
                logger.info("The input({}) mean/std -> {}".format(name, norm[name]))
            logger.info("The input({}) dtype -> {}".format(name, in_dtypes[name]))

        import tvm
        from tvm import relay
        quantize_config = tvm.relay.quantization.get_quantize_config(self.target, in_dtypes)
        quantize_config["calib_method"] = self.quant_calib_method
        quantize_config["level"] = self.quant_opt_level
        if self.disable_pass is not None:
            quantize_config["disable_pass"] = self.disable_pass
        quantize_config["float_list"] = list()
        skip_layer_idxes = self.quant_cfg.get("skip_layer_idxes", list())
        skip_layer_types = self.quant_cfg.get("skip_layer_types", list())
        skip_layer_names = self.quant_cfg.get("skip_layer_names", list())
        if skip_layer_idxes:
            quantize_config["float_list"].extend(skip_layer_idxes)
        if skip_layer_types:
            quantize_config["float_list"].extend(skip_layer_types)
        if skip_layer_names:
            quantize_config["float_list"].extend(skip_layer_names)
        logger.info("quantize_config: {}".format(quantize_config))
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
    def build_x86_64(relay_func, params, target, save_path=""):
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

    def infer(self, device_id, node_id):
        """ inference on chip/sdk_iss """
        raise NotImplementedError

    def profile(self, device_id, node_id):
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
        t_start = time.time()
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
        elif self.framework == "onnx-qnn":
            self.onnx_qnn2relay()
        else:
            logger.error("Not support framework -> {}".format(self.framework))
            exit(-1)
        self.x2relay_span = time.time() - t_start
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

    def onnx_qnn2relay(self):
        raise NotImplementedError
