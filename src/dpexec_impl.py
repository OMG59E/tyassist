#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : dpexec_impl.py
@Time    : 2022/7/14 上午9:40
@Author  : xingwg
@Email   : xing.weiguo@intellif.com
@Software: PyCharm
"""
import os
import importlib
import pickle
import cv2
import numpy as np
import collections
import json
from utils import logger
from utils.preprocess import default_preprocess, calc_padding_size
from utils.enum_type import PaddingMode, PixelFormat, DataLayout
from utils.nnp_func import (
    nnp3xx_load_from_json,
    nnp4xx_load_from_json,
    nnp3xx_build_lib,
    nnp4xx_build_lib,
    nnp4xx_inference
)

import tvm


class DpExec(object):
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._quant_cfg = cfg["build"]["quant"]
        self._target = cfg["build"]["target"]
        self._enable_quant = cfg["build"]["enable_quant"]
        self._enable_dump = cfg["build"]["enable_dump"]
        self._framework = cfg["model"]["framework"]
        self._custom_preprocess_module = self._quant_cfg["custom_preprocess_module"]
        self._custom_preprocess_cls = self._quant_cfg["custom_preprocess_cls"]
        self._graph = cfg["model"]["graph"]
        self._weight = cfg["model"]["weight"]
        self._inputs = cfg["model"]["inputs"]
        self._outputs = cfg["model"]["outputs"]
        self._num_inputs = len(self._inputs)
        self._num_outputs = len(self._outputs)
        self._relay_quant = None
        self._params_quant = None
        self._relay = None
        self._params = None

        self._input_enable_aipps = list()
        self._data_layouts = list()
        self._pixel_formats = list()
        self._resize_types = list()
        self._padding_modes = list()
        self._padding_values = list()
        self._shapes = list()
        self._means = list()
        self._stds = list()
        for idx, _input in enumerate(self._inputs):
            _input["support"] = False if "None" == _input["pixel_format"] or "None" == _input["layout"] else True

            if _input["support"]:
                if "enable_aipp" not in _input:
                    _input["enable_aipp"] = True
            else:
                _input["enable_aipp"] = False

            # if self._enable_dump:
            #     logger.info("enable_dump: true")
            #     _input["enable_aipp"] = False

            self._input_enable_aipps.append(_input["enable_aipp"])

            # data layout
            layout = DataLayout.NCHW
            if _input["layout"] == "NCHW":
                pass
            elif "NHWC" == _input["layout"]:
                layout = DataLayout.NHWC
            else:
                layout = DataLayout.NONE

            self._data_layouts.append(layout)

            # shape
            shape = _input["shape"]
            n, c, h, w = shape
            if layout == DataLayout.NHWC:
                n, h, w, c = shape
            self._shapes.append((n, c, h, w))

            # mean std
            mean = _input["mean"] if _input["mean"] else [0.0 for _ in range(c)]
            std = _input["std"] if _input["std"] else [1.0 for _ in range(c)]
            self._means.append(mean)
            self._stds.append(std)

            # resize padding
            self._resize_types.append(_input["resize_type"])
            self._padding_values.append(_input["padding_value"])
            self._padding_modes.append(PaddingMode.LEFT_TOP if _input["padding_mode"] == 0 else PaddingMode.CENTER)

            # pixel_format
            pixel_format = PixelFormat.NONE
            if _input["pixel_format"] == "RGB":
                pixel_format = PixelFormat.RGB
            elif _input["pixel_format"] == "BGR":
                pixel_format = PixelFormat.BGR
            elif _input["pixel_format"] == "GRAY":
                pixel_format = PixelFormat.GRAY
            elif _input["pixel_format"] == "None":
                pass
            else:
                logger.error("Not support pixel_format -> {}".format(_input["pixel_format"]))
                exit(-1)
            self._pixel_formats.append(pixel_format)

        self._model_dir = cfg["model"]["save_dir"]
        self._result_dir = os.path.join(self._model_dir, "result")
        if not os.path.exists(self._result_dir):
            os.makedirs(self._result_dir)

        # 设置自定义预处理
        self.set_custom_preprocess()

    @property
    def target(self):
        return self._target

    @property
    def has_custom_preprocess(self):
        return True if self._custom_preprocess_cls else False

    @property
    def enable_dump(self):
        return self._enable_dump

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def input_names(self):
        return [_input["name"] for _input in self._inputs]

    @property
    def num_inputs(self):
        return len(self._inputs)

    @property
    def input_enable_aipps(self):
        return self._input_enable_aipps

    # @property
    # def enable_aipp(self):
    #     return not self._enable_dump

    def mean(self, idx):
        return self._means[idx]

    def std(self, idx):
        return self._stds[idx]

    def shape(self, idx):
        return self._shapes[idx]

    def data_layout(self, idx):
        return self._data_layouts[idx]

    def pixel_formats(self, idx):
        return self._pixel_formats[idx]

    def resize_type(self, idx):
        return self._resize_types[idx]

    def padding_value(self, idx):
        return self._padding_values[idx]

    def padding_mode(self, idx):
        return self._padding_modes[idx]

    def print_tvm_version(self):
        try:
            version = ""
            if self._target.startswith("nnp3"):
                import deepeye
                version = deepeye.util.get_version()
            elif self._target.startswith("nnp4"):
                from tvm.contrib.edgex import get_version
                version = get_version()
            else:
                logger.error("Not support target -> {}".format(self._target))
                exit(-1)
            logger.info("TyTVM Version: {}".format(version))
        except Exception as e:
            logger.error("Failed to get tytvm version -> {}".format(e))
            exit(-1)

    def x2relay(self):
        """任意框架转译至relay格式
        """
        from tvm import relay
        shape_dict, dtype_dict = dict(), dict()
        for _, _input in enumerate(self._inputs):
            shape_dict[_input["name"]] = _input["shape"]
            dtype_dict[_input["name"]] = "float32"

        if self._framework == "caffe":
            from google.protobuf import text_format
            from tvm.relay.frontend import caffe_pb2 as pb
            deploy_net, weight_net = pb.NetParameter(), pb.NetParameter()
            with open(self._graph, "r") as f:
                text_format.Merge(f.read(), deploy_net)
            with open(self._weight, "rb") as f:
                weight_net.ParseFromString(f.read())
            self._relay, self._params, _ = relay.frontend.from_caffe(weight_net, deploy_net, shape_dict, dtype_dict)
        elif self._framework == "tensorflow":
            import tvm.relay.testing.tf as tf_testing
            output_names = [item["name"] for item in self._outputs]
            graph_def = tf_testing.get_graph_def_from_pb(self._weight, output_names)
            sym, params = relay.frontend.from_tensorflow(
                graph=graph_def,
                layout="NCHW",
                shape=shape_dict,
                outputs=output_names
            )
            sym = sym["main"]
            for idx, _input in enumerate(self._inputs):
                if self.data_layout(idx) == DataLayout.NHWC:
                    nhwc_shape = _input["shape"]
                    nchw_shape = [nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]]
                    shape_dict[_input["name"]] = nchw_shape

            # 目前不能分别设置多个输入/输出的NHWC->NCHW的转换
            self._relay, self._params = relay.relay_pass.convert_nhwc_to_nchw(
                sym,
                params,
                shape_dict,
                convert_input_as_nchw=True if self.data_layout(0) == DataLayout.NHWC else False,
                convert_output_as_nchw=False,
            )
        elif self._framework == "onnx":
            if self._target.startswith("nnp3"):
                import onnx
                model = onnx.load(self._weight)
                self._relay, self._params = relay.frontend.from_onnx(model, shape_dict, dtype_dict)
            elif self._target.startswith("nnp4"):  # from_onnx后的扩展变换，~= from_onnx
                from tvm.contrib.edgex import load_model_from_file
                kwargs = {"dtype": dtype_dict}
                self._relay, self._params = load_model_from_file(
                    self._weight, self._framework, shape_dict, **kwargs)
        elif self._framework == "pytorch":
            import torch
            model = torch.jit.load(self._weight)
            input_shapes = list()
            for _input in self._inputs:
                input_shapes.append((_input["name"], _input["shape"]))
            self._relay, self._params = relay.frontend.from_pytorch(model, input_shapes)
        elif self._framework == "mxnet":
            from mxnet import symbol
            from mxnet import ndarray as nd
            model = symbol.load(self._graph)
            save_dict = nd.load(self._weight)
            if not save_dict:
                logger.warning("Params file is empty -> {}".format(self._weight))
            arg_params, aux_params = dict(), dict()
            for k, v in save_dict.items():
                tp, name = k.split(":", 1)
                if tp == "arg":
                    arg_params[name] = v
                if tp == "aux":
                    aux_params[name] = v

            self._relay, self._params = relay.frontend.from_mxnet(
                model,
                shape=shape_dict,
                dtype=dtype_dict,
                arg_params=arg_params,
                aux_params=aux_params
            )
        else:
            logger.error("Not support {} yet".format(self._framework))
            exit(-1)

        return self._relay, self._params

    def set_custom_preprocess(self):
        """检查是否存在自定义预处理
        不使能aipp/CR的情况下需要使用外部预处理(resize和cvtColor)，
        1. 多输入情况需要自定义
        2. 默认预处理不能满足的情况
        :return:
        """
        # 自定义预处理
        if self._custom_preprocess_cls:
            m = importlib.import_module(self._custom_preprocess_module)
            if hasattr(m, self._custom_preprocess_cls):
                # 实例化预处理对象
                self._custom_preprocess_cls = getattr(m, self._custom_preprocess_cls)(
                    self._inputs, self._quant_cfg["prof_img_num"], self._quant_cfg["data_dir"])
            else:
                logger.error("{}.py has no class named {}".format(
                    self._custom_preprocess_module, self._custom_preprocess_cls))
                exit(-1)

    def get_datas(self, use_norm=False, force_cr=False, to_file=True):
        """获取量化/编译阶段处理数据
        :return:
        """
        in_datas = collections.OrderedDict()  # 保证输入顺序一致
        for idx, _input in enumerate(self._inputs):
            data_path = _input["data_path"]
            n, c, h, w = self.shape(idx)
            if data_path:
                if not os.path.exists(data_path):
                    logger.error("Not found data_path -> {}".format(data_path))
                    return None
                if not _input["support"]:
                    # not support will use custom preprocess
                    in_datas[_input["name"]] = self._custom_preprocess_cls.get_single_data(data_path, idx)
                else:
                    # default preprocess，only image channel 1 or 3
                    im = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE if self._pixel_formats[idx] == PixelFormat.GRAY else cv2.IMREAD_COLOR)
                    if (not _input["enable_aipp"]) or force_cr:
                        _input["padding_size"], _ = calc_padding_size(im, (w, h), self.padding_mode(idx))
                        in_datas[_input["name"]] = default_preprocess(
                            im,
                            (w, h),
                            mean=self.mean(idx),
                            std=self.std(idx),
                            use_norm=use_norm,
                            use_rgb=True if self._pixel_formats[idx] == PixelFormat.RGB else False,
                            use_resize=True,
                            resize_type=_input["resize_type"],
                            padding_value=_input["padding_value"],
                            padding_mode=self.padding_mode(idx)
                        )
                    else:
                        if self._pixel_formats[idx] == PixelFormat.RGB:
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # BGR -> RGB, aipp need

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
                logger.warning("Not set data_path, will use random data")
                # 采用随机数据
                if use_norm:
                    in_datas[_input["name"]] = np.random.randn(n, c, h, w).astype(dtype=np.float32)
                else:
                    in_datas[_input["name"]] = np.random.randint(0, 255, (n, c, h, w)).astype(dtype=np.uint8)
            if to_file:
                # save data
                in_datas[_input["name"]].tofile(os.path.join(self._result_dir, "data_{}_CRN.bin".format(idx)))
                in_datas[_input["name"]].tofile(os.path.join(self._result_dir, "data_{}_CRN.txt".format(idx)), sep="\n")

        return in_datas

    def _nnp3xx_quantization(self, quantize_config, norm):
        from tvm.relay.quantization import quantize
        # 保存路径设置
        return quantize(
            self._relay,
            self._params,
            model_name="opt_ir",
            # 用户使用云天自带的预处理时，配置为输入量化profile(统计模型的层分布用来 calibrate 生成 scale)
            # 的图片集路径，支持图片格式为 jpg，jpeg，png，bmp。也可配置为用户自定义的预处理。类型str/generator
            dataset=self._quant_cfg["data_dir"] if not self._custom_preprocess_cls else self._custom_preprocess_cls.get_data,
            # 使用校准数据数量
            prof_img_num=self._quant_cfg["prof_img_num"],
            # 此配置仅在 dataset 配置为图片集路径（即使用云天自带的预处理），且输入为3通道时有效，对生成芯片模型无效
            rgb_en=1 if (self.num_inputs == 1 and self._pixel_formats[0] == PixelFormat.RGB and (not self.has_custom_preprocess)) else 0,
            # 均值方差，对生成芯片模型生效
            norm=norm,
            # 量化配置
            quantize_config=quantize_config,
            # 用来进行相似度以及相关量化效果确认
            debug_level=self._quant_cfg["debug_level"],
            # 进行相似度比对的图片的个数，默认为1。如果大于1，最终会输出所有图片的平均相似度。
            similarity_img_num=self._quant_cfg["similarity_img_num"],
            # 进行相似度比对的输入数据，配置为None，则默认取前述dataset接口表示的输入。
            # 具体配置参照dataset，可定义为图片集路径，或用户自定义的预处理。
            similarity_dataset=self._quant_cfg["similarity_dataset"],
            # 用来保存
            # 1. /fp32/output_tensors.params和quant/output_tensors.params 表示dump每一层浮点和定点的数据
            # 2. "model_name" /opt_ir.pdf 表示optimize之后模型的拓扑结构
            # 3. scale_record.json 存储每一层的scale信息
            # 4. display_quant_vs_float 文件夹通过png图片显示每一层浮点和定点的分布及包络图
            # 5. relay.model量化后onnx格式的IR
            save_dir=self._result_dir,
            # 默认为True，量化后模型输出tensor的dtype与原始模型保持同步。若指定为false，则不做dtype同步
            sync_outdtype=True,
        )

    def _nnp4xx_quantization(self, quantize_config, norm):
        quantized_mod, quantized_params = tvm.relay.quantization.quantize(
            self._relay,
            self._params,
            model_name="opt_ir",
            dataset=self._quant_cfg["data_dir"] if not self._custom_preprocess_cls else self._custom_preprocess_cls.get_data,
            prof_img_num=self._quant_cfg["prof_img_num"],
            rgb_en=1 if (self.num_inputs == 1 and self._pixel_formats[0] == PixelFormat.RGB and (not self.has_custom_preprocess)) else 0,
            norm=norm,
            quantize_config=quantize_config,
            debug_level=self._quant_cfg["debug_level"],
            save_dir=self._result_dir,
        )
        return quantized_mod, quantized_params

    def relay_quantization(self, in_datas):
        """量化，将浮点relay函数转为成定点relay函数
        """
        logger.info("################   quantization start  ######################")
        # from tvm.relay.quantization import quantize, get_quantize_config

        in_dtypes, norm = dict(), dict()
        # 如果配置文件为设置mean/std
        for idx, _input in enumerate(self._inputs):
            data_type = "uint8"
            if in_datas[_input["name"]].dtype == np.uint8:
                pass
            elif in_datas[_input["name"]].dtype == np.float32:
                data_type = "float32"
            else:
                logger.error("Input dtype not support -> {}".format(in_datas[_input["name"]].dtype))
                exit(-1)
            # 与最终量化后的模型输入数据类型相对应
            in_dtypes[_input["name"]] = data_type if self.has_custom_preprocess else "uint8"
            norm[_input["name"]] = {"mean": self.mean(idx), "std": self.std(idx), "axis": 1}
            logger.info("Input({}) dtype -> {}".format(_input["name"], in_dtypes[_input["name"]]))
            logger.info("Input({}) mean/std -> {}".format(_input["name"], norm[_input["name"]]))

        quantize_config = tvm.relay.quantization.get_quantize_config(self._target, in_dtypes)
        quantize_config["calib_method"] = self._quant_cfg["calib_method"]

        quantize_config["float_list"] = list()
        if self._quant_cfg["skip_layer_idxes"]:
            quantize_config["float_list"].extend(self._quant_cfg["skip_layer_idxes"])
        if self._quant_cfg["skip_layer_types"]:
            quantize_config["float_list"].extend(self._quant_cfg["skip_layer_types"])
        logger.info("quantize_config -> {}".format(quantize_config))

        quan_json_path = os.path.join(self._result_dir, "quantized.json")
        quan_model_path = os.path.join(self._result_dir, "quantized.model")
        if self._target.startswith("nnp3"):
            self._relay_quant = self._nnp3xx_quantization(quantize_config, norm)
            tvm.relay.quantize.save_deepeye_quan_model(self._relay_quant, quan_json_path)  # 将relay_func存为json
            from tvm.contrib.export import RelayExporter
            RelayExporter().save(self._relay_quant, quan_model_path)  # 生成Netron模型可视化文件
            logger.info("save quant model to {}".format(quan_model_path))
        elif self._target.startswith("nnp4"):
            self._relay_quant, self._params_quant = self._nnp4xx_quantization(quantize_config, norm)
            tvm.relay.quantization.save_ir_to_json(quan_json_path, self._relay_quant, self._params_quant)

        if self._quant_cfg["debug_level"] == 1:
            # layer_outs: dict，key为每层的name(为方便用户获知原始浮点模型每一层的数据状况，
            # 所以尽可能使用了原始浮点模型自带的op_name，但实际处理会获取不到原始模型的op_name，
            # 此时会使用opt_ir.pdf中的op_name)，相应的value为浮点和定点结果的组成的list
            layer_outs = tvm.relay.quantization.compare_layer_outputs(self._result_dir)

        logger.info("save quant json to {}".format(quan_json_path))
        logger.info("################   quantization end  ######################")

    def load_relay_quant_from_json(self):
        """加载
        :return:
        """
        quan_json_path = os.path.join(self._result_dir, "quantized.json")
        if not os.path.exists(quan_json_path):
            logger.error("Not found quant_json")
            exit(-1)
        if self._target.startswith("nnp3"):
            self._relay_quant, self._params_quant = nnp3xx_load_from_json(quan_json_path)
        elif self._target.startswith("nnp4"):
            self._relay_quant, self._params_quant = nnp4xx_load_from_json(quan_json_path)

    @staticmethod
    def _nnp3xx_eval_relay(relay_func, params, in_datas):
        """
        :param relay_func:
        :param params:
        :param in_datas:
        :return:
        """
        import deepeye
        return deepeye.eval_relay(relay_func, params, in_datas)

    def tvm_fixed_output(self, in_datas, to_file=True):
        """量化后模型在CPU端仿真推理
        :param in_datas:
        :param to_file:
        :return:
        """
        outputs = list()
        if self._target.startswith("nnp3"):
            outputs = self._nnp3xx_eval_relay(self._relay_quant, {}, in_datas)
        elif self._target.startswith("nnp4"):
            outputs = self._nnp4xx_tvm_fixed(in_datas)
        if to_file and len(outputs) > 0:
            for idx, output in enumerate(outputs):
                output.tofile(os.path.join(self._result_dir, "host_tvm_fixed_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self._result_dir, "host_tvm_fixed_out_{}.txt".format(idx)), sep="\n")
        return outputs

    def tvm_float_output(self, in_datas, to_file=True):
        """获取转译后的relay(可视为与原始模型等价)推理仿真结果
        :param in_datas:
        :param to_file:
        :return:
        """
        outputs = list()
        if self._target.startswith("nnp3"):
            outputs = self._nnp3xx_eval_relay(self._relay, self._params, in_datas)
        elif self._target.startswith("nnp4"):
            outputs = self._nnp4xx_tvm_float(in_datas)
        if to_file and len(outputs) > 0:
            for idx, output in enumerate(outputs):
                output.tofile(os.path.join(self._result_dir, "host_tvm_float_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self._result_dir, "host_tvm_float_out_{}.txt".format(idx)), sep="\n")
        return outputs

    def _nnp3xx_make_netbin(self, in_datas):
        """nnp3xx编译relay_func, 并生产netbin模型
        :param in_datas:
        :return:
        """
        logger.info("################### build start ####################")
        import deepeye
        from deepeye.relay_pass import dump_func_output
        input_info = dict()
        for idx, _input in enumerate(self._inputs):
            # 模型输入信息其key包含"layout", "resize_type", "padding_size", "padding_value"，都为可选参数，
            # 但如果配置了任意参数，则"layout"参数就必须设置为"RGB", "BGR"，"GRAY"三者之一
            # 开启dump功能，会禁止CR模块，需要将layout强设成NCHW来关闭CR功能
            if not _input["enable_aipp"]:
                logger.warning("Input({}) will disable_aipp".format(_input["name"]))
                input_info[_input["name"]] = {"layout": "NCHW"}
            else:
                input_info[_input["name"]] = {
                    "layout": _input["pixel_format"],
                    "resize_type": _input["resize_type"],
                    "padding_size": None if PaddingMode.CENTER == self.padding_mode(idx) else _input["padding_size"],
                    "padding_value": _input["padding_value"],
                }
                logger.info("Input({}): will enable_aipp".format(_input["name"]))
            logger.info("Input({}) info -> {}".format(input_info[_input["name"]], input_info[_input["name"]]))

        deepeye.make_netbin(
            self._relay_quant,  # 提供输入数据类型信息
            self._target,
            self._model_dir,
            input_info,
            params=None,
            model_name="opt_ir",
            return_buffer=False,
            debug_level=1 if self._enable_dump else 0,
            opt_cfg=None,
            extra_info=""
        )
        logger.info("################### build end ####################")

        iss_fixed_outputs = None
        if self._target.startswith("nnp3") and self._enable_dump:
            from deepeye.run_net_bin.run_net_bin import run_net_bin
            netbin_file = os.path.join(self._model_dir, "net_combine.bin")
            if not os.path.exists(netbin_file):
                logger.error("Not found netbin_file -> {}".format(netbin_file))
                exit(-1)

            # in_datas = self.get_datas(use_norm=False, force_cr=True)
            in_datas_list = [in_datas[key] for key in in_datas]
            iss_fixed_outputs = run_net_bin(netbin_file, in_datas_list)
            for idx, output in enumerate(iss_fixed_outputs):
                output.tofile(os.path.join(self._result_dir, "host_iss_fixed_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self._result_dir, "host_iss_fixed_out_{}.txt".format(idx)), sep="\n")

        if self._enable_dump:
            # iss芯片软仿，生成每个融合算子在iss上的输出数据，用于和芯片硬仿做对比。
            # 目前debug_level=1启动dump功能，无法使用CR模块，需要将layout强设成NCHW来关闭CR功能
            dump_dict, weight = dump_func_output(self._model_dir, in_datas, self._target)
            with open(os.path.join(self._result_dir, "host_iss_fused_out.pickle"), "wb") as fp:
                pickle.dump(dump_dict, fp)
            with open(os.path.join(self._result_dir, "host_iss_fused_weight.pickle"), "wb") as fp:
                pickle.dump(weight, fp)

        return iss_fixed_outputs

    def print_nnp4xx_estimate_flops(self):
        from tvm.contrib.edgex import estimate_FLOPs
        estimate_FLOPs(self._relay_quant)

    def print_nnp4xx_estimate_cycles(self):
        from tvm.contrib.edgex import estimate_cycles
        estimate_cycles(self._relay_quant)

    def nnp4xx_iss_fixed(self, lib, in_datas):
        from tvm.contrib import graph_executor
        logger.info("Executing model on edgex...")
        edgex_module = graph_executor.GraphModule(lib["default"](tvm.edgex(), tvm.cpu()))
        return nnp4xx_inference(edgex_module, in_datas)

    def _nnp4xx_tvm_fixed(self, in_datas):
        return nnp4xx_inference(nnp4xx_build_lib(self._relay_quant, self._params_quant), in_datas)

    def _nnp4xx_tvm_float(self, in_datas):
        return nnp4xx_inference(nnp4xx_build_lib(self._relay, self._params), in_datas)

    def _nnp4xx_make_netbin(self, in_datas):
        from tvm.contrib.edgex import compile_nnp_model
        # compile edgex lib
        edgex_lib = compile_nnp_model(
            self._relay_quant,
            self._params_quant,
            output_path="{}/net_combine.bin".format(self._model_dir),
            opt_level=2,
        )
        logger.info("Executing model on edgex...")
        return self.nnp4xx_iss_fixed(edgex_lib, in_datas)

    @staticmethod
    def _set_nnp4xx_env():
        dep_path = "{}/de-dcl/client/lib".format(tvm.__path__[0])
        ld_path = os.getenv("LD_LIBRARY_PATH")
        ld_path = dep_path if ld_path is None else dep_path + ":" + ld_path
        os.environ["LD_LIBRARY_PATH"] = ld_path
        os.environ["EDGEX_DEBUG_ISS"] = "on"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    def make_netbin(self, in_datas):
        if self._target.startswith("nnp3"):
            return self._nnp3xx_make_netbin(in_datas)
        elif self._target.startswith("nnp4"):
            self._set_nnp4xx_env()
            return self._nnp4xx_make_netbin(in_datas)
