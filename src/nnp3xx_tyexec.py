#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp3xx_tyexec.py 
@time: 2022/12/14
@Author  : xingwg
@software: PyCharm 
"""
import os
import json
import pickle
import numpy as np
from abc import ABC
from prettytable import PrettyTable
from collections import OrderedDict
from utils import logger
from utils.enum_type import PaddingMode
from .base_tyexec import BaseTyExec


class Nnp3xxTyExec(BaseTyExec, ABC):
    def __init__(self, cfg: dict):
        super(Nnp3xxTyExec, self).__init__(cfg)

    def quantization(self, in_datas):
        """量化，将浮点relay函数转为成定点relay函数
        """
        if self.enable_quant:
            quantize_config, norm = self.set_quantization_cfg(in_datas)

            quant_data_dir = self.quant_cfg["data_dir"]
            dataset = quant_data_dir
            if not quant_data_dir:  # 未配置量化路径使用随机数据情况
                dataset = self.gen_random_data
            else:
                if self.has_custom_preprocess:  # 配置量化数据目录情况下存在自定义预处理
                    dataset = self.custom_preprocess_cls.get_data

            from tvm.relay.quantization import quantize
            logger.info("################   quantization start  ######################")
            # 保存路径设置
            self.relay_quant = quantize(
                self.relay,
                self.params,
                model_name="opt_ir",
                # 用户使用云天自带的预处理时，配置为输入量化profile(统计模型的层分布用来 calibrate 生成 scale)
                # 的图片集路径，支持图片格式为 jpg，jpeg，png，bmp。也可配置为用户自定义的预处理。类型str/generator
                dataset=dataset,
                # 使用校准数据数量
                prof_img_num=self.quant_cfg["prof_img_num"],
                # 此配置仅在 dataset 配置为图片集路径（即使用云天自带的预处理），且输入为3通道时有效，对生成芯片模型无效
                rgb_en=1 if (self.num_inputs == 1 and self.inputs[0]["pixel_format"] == "RGB") else 0,
                # 均值方差，对生成芯片模型生效
                norm=norm,
                # 量化配置
                quantize_config=quantize_config,
                # 用来进行相似度以及相关量化效果确认
                debug_level=self.quant_cfg["debug_level"],
                # 进行相似度比对的图片的个数，默认为1。如果大于1，最终会输出所有图片的平均相似度。
                similarity_img_num=self.quant_cfg["similarity_img_num"],
                # 进行相似度比对的输入数据，配置为None，则默认取前述dataset接口表示的输入。
                # 具体配置参照dataset，可定义为图片集路径，或用户自定义的预处理。
                similarity_dataset=self.quant_cfg["similarity_dataset"],
                # 用来保存
                # 1. /fp32/output_tensors.params和quant/output_tensors.params 表示dump每一层浮点和定点的数据
                # 2. "model_name" /opt_ir.pdf 表示optimize之后模型的拓扑结构
                # 3. scale_record.json 存储每一层的scale信息
                # 4. display_quant_vs_float 文件夹通过png图片显示每一层浮点和定点的分布及包络图
                # 5. relay.model量化后onnx格式的IR
                save_dir=self.result_dir,
                # 默认为True，量化后模型输出tensor的dtype与原始模型保持同步。若指定为false，则不做dtype同步
                sync_outdtype=True,
            )
            logger.info("################   quantization end  ######################")

            self.save_relay_to_json(self.quant_json_path, self.relay_quant, self.params_quant)
            self.save_relay_to_model(self.quant_model_path, self.relay_quant, self.params_quant)
        else:
            self.relay_quant = self.load_relay_from_json(self.quant_json_path)

        self.save_compare_layer_outputs()

    def save_compare_layer_outputs(self):
        if self.quant_cfg["debug_level"] == 1:
            # layer_outs: dict，key为每层的name(为方便用户获知原始浮点模型每一层的数据状况，
            # 所以尽可能使用了原始浮点模型自带的op_name，但实际处理会获取不到原始模型的op_name，
            # 此时会使用opt_ir.pdf中的op_name)，相应的value为浮点和定点结果的组成的list
            import tvm
            from tvm import relay
            layer_outs = tvm.relay.quantization.compare_layer_outputs(self.result_dir)

    def _set_input_info_for_build(self):
        input_info = OrderedDict()
        for idx, _input in enumerate(self.inputs):
            name = _input["name"]
            # 模型输入信息其key包含"layout", "resize_type", "padding_size", "padding_value"，都为可选参数，
            # 但如果配置了任意参数，则"layout"参数就必须设置为"RGB", "BGR"，"GRAY"三者之一
            # 开启dump功能，会禁止CR模块，需要将layout强设成NCHW来关闭CR功能
            if not _input["enable_aipp"]:
                logger.warning("Input({}) will disable_aipp".format(name))
                input_info[name] = {"layout": "NCHW"}
            else:
                input_info[name] = {
                    "layout": _input["pixel_format"],
                    "resize_type": _input["resize_type"],
                    "padding_size": None if PaddingMode.CENTER == _input["padding_mode"] else _input["padding_size"],
                    "padding_value": _input["padding_value"],
                }
                logger.info("Input({}): will enable_aipp".format(name))
            logger.info("Input({}) info -> {}".format(name, input_info[name]))
        return input_info

    def tvm_float_inference(self, in_datas, to_file=False):
        engine = self.build_x86_64(self.relay, self.params)
        tvm_float_outputs = self.tvm_inference(engine, in_datas)
        # tvm_float_outputs = tvm_float_outputs.values()  # dict to list
        if to_file and len(tvm_float_outputs) > 0:
            for idx, output in enumerate(tvm_float_outputs):
                output.tofile(os.path.join(self.result_dir, "tvm_float_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self.result_dir, "tvm_float_out_{}.txt".format(idx)), sep="\n")
        return tvm_float_outputs

    def tvm_fixed_inference(self, in_datas, to_file=False):
        engine = self.build_x86_64(self.relay_quant, self.params_quant)
        tvm_float_outputs = self.tvm_inference(engine, in_datas)
        if to_file and len(tvm_float_outputs) > 0:
            for idx, output in enumerate(tvm_float_outputs):
                output.tofile(os.path.join(self.result_dir, "tvm_fixed_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self.result_dir, "tvm_fixed_out_{}.txt".format(idx)), sep="\n")
        return tvm_float_outputs

    def iss_fixed_inference(self, in_datas, to_file=False):
        iss_fixed_outputs = None
        if self.enable_dump:
            from deepeye.run_net_bin.run_net_bin import run_net_bin
            netbin_file = os.path.join(self.model_dir, "{}.ty".format(self.model_name))
            if not os.path.exists(netbin_file):
                logger.error("Not found netbin_file -> {}".format(netbin_file))
                exit(-1)

            iss_fixed_outputs = run_net_bin(netbin_file, in_datas, work_path=self.model_dir, target=self.target)  # disable aipp
            iss_fixed_outputs = [iss_fixed_outputs[name] for name in iss_fixed_outputs]  # dict to list
            if to_file and len(iss_fixed_outputs) > 0:
                for idx, iss_fixed_output in enumerate(iss_fixed_outputs):
                    iss_fixed_output.tofile(os.path.join(self.result_dir, "iss_fixed_out_{}.bin".format(idx)))
                    iss_fixed_output.tofile(os.path.join(self.result_dir, "iss_fixed_out_{}.txt".format(idx)), sep="\n")
        return iss_fixed_outputs

    def iss_dump_output(self, in_datas):
        if self.enable_dump == 1:
            # iss芯片软仿，生成每个融合算子在iss上的输出数据，用于和芯片硬仿做对比。
            from deepeye.relay_pass import dump_func_output
            dump_dict, weight = dump_func_output(self.model_dir, in_datas, self.target)
            with open(os.path.join(self.result_dir, "iss_fused_out.pickle"), "wb") as fp:
                pickle.dump(dump_dict, fp)
            with open(os.path.join(self.result_dir, "iss_fused_weight.pickle"), "wb") as fp:
                pickle.dump(weight, fp)

    @staticmethod
    def build_x86_64(relay_func, params, save_path=""):
        """ build tvm cpu model
        @param relay_func:
        @param params:
        @param save_path:
        @return:
        """
        try:
            import tvm
            from tvm import relay
            from tvm.contrib import graph_runtime
            from deepeye.relay_pass import rewrite_for_cpu
            relay_func = relay.relay_pass.bind_params(relay_func, params)
            relay_func.ret_type = None
            relay_func, cpu_params = rewrite_for_cpu(relay_func, sim_target="nnp300")
            ctx = tvm.cpu(0)
            with relay.build_config(opt_level=3):
                graph, lib, cpu_params = relay.build(relay_func, "llvm", params=cpu_params)
            engine = graph_runtime.create(graph, lib, ctx)
            engine.set_input(**cpu_params)
            return engine
        except Exception as e:
            logger.error("Failed to compile model -> {}".format(e))
            exit(-1)

    def build(self, in_datas):
        """build relay quant
        @param in_datas:  iss infer data
        @return:
        """
        if self.enable_build:
            input_info = self._set_input_info_for_build()

            # opt_cfg = dict()
            # opt_cfg["SUPPRESS_LONG_FUNC"] = 1

            logger.info("################### build start ####################")
            import deepeye
            deepeye.make_netbin(
                self.relay_quant,  # 提供输入数据类型信息
                self.target,
                self.model_dir,
                input_info,
                params=None,
                model_name="opt_ir",
                return_buffer=False,
                debug_level=1 if self.enable_dump else 0,
                opt_cfg=None,
                extra_info=""
            )
            logger.info("################### build end ####################")
            # NOTE 临时重命名输出模型
            self._rename()
        else:
            logger.warning("disable build")

        # self.model_analysis()
        self.get_profile_info()

        iss_fixed_outputs = self.iss_fixed_inference(in_datas)
        self.iss_dump_output(in_datas)
        return iss_fixed_outputs

    @staticmethod
    def save_relay_to_model(quant_model_path, relay_func, params):
        from tvm.contrib.export import RelayExporter
        RelayExporter().save(relay_func, quant_model_path, params)  # 生成Netron模型可视化文件
        logger.info("save quant model to {}".format(quant_model_path))

    def get_version(self):
        try:
            import deepeye
            version = deepeye.util.get_version()
            logger.info("TyTVM Version: {}".format(version))
        except Exception as e:
            logger.error("Failed to get tytvm version -> {}".format(e))
            exit(-1)

    @property
    def targets(self):
        return {"nnp320": 768, "nnp300": 792, "nnp200": 750}

    def get_relay_mac(self):
        from deepeye.util import count_mac
        logger.info("float relay MAC: {}".format(count_mac(self.relay)))
        logger.info("fixed relay MAC: {}".format(count_mac(self.relay_quant)))

    def get_profile_info(self):
        filepath = os.path.join(self.model_dir, "model_profile.json")
        if not os.path.exists(filepath):
            logger.warning("Not found file -> {}".format(filepath))
            return
        graph_filepath = os.path.join(self.model_dir, "graph.json")
        if not os.path.exists(graph_filepath):
            logger.warning("Not found file -> {}".format(graph_filepath))
            return
        with open(filepath, "rb") as f:
            model_profile = json.load(f)
        with open(graph_filepath, "rb") as f:
            graph = json.load(f)

        op_name_map = dict()
        nodes = graph["nodes"]
        for node in nodes:
            if "debug_name" not in node or "name" not in node:
                continue
            op_name = node["name"]
            debug_name = node["debug_name"]
            op_name_map[op_name] = debug_name

        header = ["Id", "OpName", "DDR/Read", "DDR/Write", "MAC", "Cycles", "Span/ms"]
        table = PrettyTable(header)
        func_info = model_profile["func_info"]
        for idx, op_name in enumerate(func_info):
            table.add_row([
                idx,
                op_name_map[op_name],
                func_info[op_name]["ddr_read"],
                func_info[op_name]["ddr_write"],
                func_info[op_name]["mac"],
                func_info[op_name]["cost"],
                "{:.3f}".format(func_info[op_name]["cost"] * 2.0 * 10**-3 / self.targets[self.target])
            ])
        logger.info("model profile:\n{}".format(table))

    def get_device_type(self):
        if self.enable_dump == 0:
            logger.warning("Not enable_dump, can`t get device type")
            return
        import deepeye
        ops_dict = deepeye.util.get_device_type(self.model_dir, node_name=None)
        header = ["Id", "OpName", "DeviceType"]
        table = PrettyTable(header)
        for idx, op_name in enumerate(ops_dict):
            table.add_row([idx, op_name, ops_dict[op_name]])
        logger.info("Op device type:\n{}".format(table))

    def compress_analysis(self):
        import tvm
        from tvm import relay
        from tvm.contrib import graph_runtime
        from tvm.relay import expr as _expr
        from tvm.relay import ir_pass as _ir_pass

        relay_func = self.load_relay_from_json(self.quant_json_path)

        feature_map_nodes = list()
        weights = list()

        def visit_func(expr):
            """visit_func"""
            if isinstance(expr, _expr.Call) and expr.op.name in self.target_ops:
                data, weight = expr.args
                if isinstance(weight, _expr.Constant):
                    weights.append(weight.data.asnumpy())
                feature_map_nodes.append(data)

        _ir_pass.post_order_visit(relay_func, visit_func)
        graph = relay.Function(relay_func.params, tvm.relay.Tuple(feature_map_nodes))

        weight_size = 0
        weight_count = 0
        for idx, w in enumerate(weights):
            weight_count += len(np.where(w.flatten() == 0)[0])
            weight_size += len(w.flatten())

        ctx = tvm.cpu(0)
        with tvm.relay.build_config(opt_level=3):
            graph, lib, params = tvm.relay.build(graph, "llvm", params=None)
        module = graph_runtime.create(graph, lib, ctx)
        module.set_input(**params)

        data_dir = self.quant_cfg["data_dir"]
        prof_img_num = self.quant_cfg["prof_img_num"]
        data_lists = ["" for _ in range(prof_img_num)]
        if os.path.exists(data_dir):
            data_lists = os.listdir(data_dir)
            if prof_img_num < len(data_lists):
                data_lists = data_lists[0:prof_img_num]
            else:
                prof_img_num = len(data_lists)
        else:
            logger.warning("Not set data_dir, will use random data")

        ret = []
        feature_map_size = 0
        feature_map_count = 0
        for filename in data_lists:
            filepath = ""
            if filename:
                _, ext = os.path.splitext(filename)
                if ext not in [".JPEG", ".jpg", ".bmp", ".png", ".PNG", ".npy"]:
                    continue
                filepath = os.path.join(data_dir, filename)
            in_datas = self.get_datas(filepath=filepath, use_norm=False, force_cr=True, to_file=False)
            outputs = self.tvm_inference(module, in_datas)
            for idx, output in enumerate(outputs):
                feature_map_count += len(np.where(output.flatten() == 0)[0])
                feature_map_size += len(output.flatten())
            ret.append(outputs)
        logger.info("feature_map: {}/{}={:.6f}".format(
            feature_map_count, feature_map_size, float(feature_map_count) / (feature_map_size + np.finfo(float).eps)))
        logger.info("weight: {}/{}={:.6f}".format(weight_count, weight_size, float(weight_count) / weight_size))
        return ret, weights

    def model_analysis(self):
        if self.target != "nnp300":
            logger.warning("model_analysis only support nnp300")
            return
        import deepeye
        deepeye.net_bin.net_bin_analysis(
            self.model_dir,
            file_name="{}.ty".format(self.model_name),
            nnp_dev="nnp300 -mnnp=nnp3xx"
        )

    def infer(self):
        """ infer one time """
        from .nnp3xx_infer import Nnp3xxSdkInfer
        in_datas = self.get_datas(use_norm=False, force_cr=False, to_file=False)
        infer = Nnp3xxSdkInfer(enable_dump=self.enable_dump, enable_aipp=True)
        infer.set_input_enable_aipps([_input["support"] for _input in self.inputs])
        infer.set_input_pixel_format([_input["pixel_format"] for _input in self.inputs])
        infer.load(self.model_path)
        outputs = infer.run(in_datas, to_file=True)
        return outputs, infer.backend

    def profile(self):
        from .nnp3xx_profiler import Nnp3xxSdkProfiler
        profiler = Nnp3xxSdkProfiler(
            net_cfg_file="/DEngine/tyhcp/net.cfg",
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            target=self.target,
        )
        in_datas = self.get_datas(use_norm=False, force_cr=True, to_file=False)
        in_datas = [in_datas[key] for key in in_datas]
        profiler.load(self.model_path)
        profiler.run(in_datas)
        profiler.unload()
        profiler.save_profile()
        profiler.parse()

    def _rename(self):
        import shutil
        src = os.path.join(self.model_dir, "{}.bin".format(self.model_name))
        if not os.path.exists(src):
            logger.error("Not found netbin_file -> {}".format(src))
            exit(-1)
        dst = os.path.join(self.model_dir, "{}.ty".format(self.model_name))
        shutil.move(src, dst)
        logger.info("rename {} -> {}".format(src, dst))

    def caffe2relay(self):
        from tvm import relay
        from google.protobuf import text_format
        from tvm.relay.frontend import caffe_pb2 as pb
        deploy_net, weight_net = pb.NetParameter(), pb.NetParameter()
        with open(self.graph, "r") as f:
            text_format.Merge(f.read(), deploy_net)
        with open(self.weight, "rb") as f:
            weight_net.ParseFromString(f.read())
        self.relay, self.params, _ = relay.frontend.from_caffe(weight_net, deploy_net, self.shape_dict, self.dtype_dict)

    def onnx2relay(self):
        import onnx
        from tvm import relay
        model = onnx.load(self.weight)
        self.relay, self.params = relay.frontend.from_onnx(model, self.shape_dict, self.dtype_dict)

    def pytorch2relay(self):
        import torch
        from tvm import relay
        model = torch.jit.load(self.weight)
        input_shapes = list()
        for idx, _input in enumerate(self.inputs):
            input_shapes.append((_input["name"], _input["shape"]))
        self.relay, self.params = relay.frontend.from_pytorch(model, input_shapes)

    def mxnet2relay(self):
        from mxnet import symbol
        from mxnet import ndarray as nd
        from tvm import relay
        model = symbol.load(self.graph)
        save_dict = nd.load(self.weight)
        if not save_dict:
            logger.warning("Params file is empty -> {}".format(self.weight))
        arg_params, aux_params = dict(), dict()
        for k, v in save_dict.items():
            tp, name = k.split(":", 1)
            if tp == "arg":
                arg_params[name] = v
            if tp == "aux":
                aux_params[name] = v

        self.relay, self.params = relay.frontend.from_mxnet(
            model,
            shape=self.shape_dict,
            dtype=self.dtype_dict,
            arg_params=arg_params,
            aux_params=aux_params
        )

    def tensorflow2relay(self):
        from tvm import relay
        import tvm.relay.testing.tf as tf_testing
        output_names = [item["name"] for item in self.outputs]
        graph_def = tf_testing.get_graph_def_from_pb(self.weight, output_names)
        shape_dict = dict()
        for idx, _input in enumerate(self.inputs):
            shape_dict[_input["name"]] = _input["shape"]
        sym, params = relay.frontend.from_tensorflow(
            graph=graph_def,
            layout="NCHW",  # 可选, 输出的目标布局
            shape=shape_dict,
            outputs=output_names
        )
        sym = sym["main"]
        self.relay, self.params = relay.relay_pass.convert_nhwc_to_nchw(
            sym,
            params,
            self.shape_dict,
            convert_input_as_nchw=True,
            convert_output_as_nchw=True,
        )

    def tflite2relay(self):
        import tflite
        from tvm import relay
        with open(self.weight, "rb") as f:
            tflite_model_buf = f.read()
        model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        shape_dict = dict()
        for idx, _input in enumerate(self.inputs):
            shape_dict[_input["name"]] = _input["shape"]
        sym, params = relay.frontend.from_tflite(model, shape_dict, self.dtype_dict)
        self.relay, self.params = relay.relay_pass.tflite_frontend_convert(
            sym,
            params,
            self.shape_dict,
            convert_input_as_nchw=True,
            convert_output_as_nchw=True
        )

    def tflite_qnn2relay(self):
        import tflite
        import deepeye
        with open(self.weight, "rb") as f:
            tflite_model_buf = f.read()
        model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        input_infos = dict()
        for idx, _input in enumerate(self.inputs):
            input_infos[_input["name"]] = {"shape": _input["shape"], "dtype": "float32"}
        self.relay = deepeye.from_qnn(model, input_infos, self.target, convert_input_as_nchw=True)
