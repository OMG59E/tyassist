#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_tyexec.py 
@time: 2022/12/14
@Author  : xingwg
@software: PyCharm 
"""
import importlib
import torch
import time
import json
import os
import onnx
import onnxruntime
from collections import OrderedDict
from abc import ABC
from utils import logger
from utils.utils import get_method
from .base_tyexec import BaseTyExec


class Nnp4xxTyExec(BaseTyExec, ABC):
    def __init__(self, cfg: dict):
        super(Nnp4xxTyExec, self).__init__(cfg)

        self.fused_json_path = os.path.join(self.result_dir, "model_fused.json")
        self.model_path_x86_64 = os.path.join(self.model_dir, "{}_x86_64.ty".format(self.model_name))
        self.model_path_aarch64 = os.path.join(self.model_dir, "{}_aarch64.ty".format(self.model_name))
        self.custom_op_module = self.cfg["model"].get("custom_op_module")

        ARM_C_COMPILER = os.getenv("ARM_C_COMPILER")
        if ARM_C_COMPILER is None:
            logger.error("Not found ARM_C_COMPILER ENV")
            exit(-1)
        elif not os.path.exists(ARM_C_COMPILER):
            logger.error("Not found ARM_C_COMPILER -> {}".format(ARM_C_COMPILER))
            exit(-1)

        logo_setting_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "setting.cfg")
        assert os.path.exists(logo_setting_filepath)
        with open(logo_setting_filepath, "r") as f:
            setting = json.load(f)
            self.logo_module = setting["logo"]

    @staticmethod
    def set_env():
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    def get_version(self):
        get_version = get_method("tvm.contrib.{}".format(self.logo_module), "get_version")
        logger.info("TyTVM Version: {}".format(get_version()))

    def onnx2relay(self):
        import onnx
        dtype_dict = dict()
        norm_dict = dict()
        qnn_in_dtype = dict()
        output_info = dict()
        for _, _input in enumerate(self.inputs):
            dtype_dict[_input["name"]] = "float32"
            norm_dict[_input["name"]] = {"mean": _input["mean"], "std": _input["std"]}
            qnn_in_dtype[_input["name"]] = "uint8" if _input["pixel_format"] else _input["dtype"]

        kwargs = {"dtype": dtype_dict, "qnn": self.is_qnn}
        if self.is_qnn:
            kwargs["norm"] = norm_dict
            kwargs["net_in_dtype"] = qnn_in_dtype
            # kwargs["output_info"] = {"output_name": {"dtype": "float32", "skip_dequant": False}}

        if self.cfg["model"].get("extras"):
            kwargs["extras"] = self.cfg["model"].get("extras")
        if self.custom_op_module is not None:
            logger.info(self.custom_op_module)
            custom_op_module = importlib.import_module(self.custom_op_module)
            from_onnx = get_method("tvm.contrib.{}.relay.frontend.onnx".format(self.logo_module), "from_onnx")
            self.relay, self.params = from_onnx(onnx.load(self.weight), shape=self.shape_dict, **kwargs)
            return
        load_model_from_file = get_method("tvm.contrib.{}".format(self.logo_module), "load_model_from_file")
        self.relay, self.params = load_model_from_file(self.weight, "onnx", self.shape_dict, **kwargs)

    def onnx_qnn2relay(self):
        self.is_qnn = True
        self.onnx2relay()

    def pytorch2relay(self):
        load_model_from_file = get_method("tvm.contrib.{}".format(self.logo_module), "load_model_from_file")
        self.relay, self.params = load_model_from_file(self.weight, "pytorch", self.shape_dict)

    @staticmethod
    def _tf_convert_nhwc_to_nchw(mod, params):
        import tvm
        from tvm import relay
        from tvm.relay.build_module import bind_params_by_name
        mod["main"] = bind_params_by_name(mod["main"], params)

        with tvm.transform.PassContext(opt_level=3):
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}),
                    relay.transform.FoldConstant(),
                ]
            )
            mod = seq(mod)

        class RemoveLayoutTransform(tvm.relay.ExprMutator):
            def __init__(self, mod):
                super(self.__class__, self).__init__()
                self.count = 0
                mod["main"] = self.visit(mod["main"])
                self.new_mod = relay.transform.InferType()(mod)

            def visit_var(self, var):
                if var.name_hint == "input_tensor":
                    new_var = relay.Var(var.name_hint, relay.TensorType([1, 3, 224, 224], "float32"))
                else:
                    new_var = var
                return new_var

            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                self.count = self.count + 1
                if self.count == 1:
                    assert call.op.name == "nn.pad"
                    new_call = relay.nn.pad(new_args[0], pad_width=((0, 0), (0, 0), (3, 3), (3, 3)))
                elif self.count == 3:
                    assert call.args[0].op.name == "layout_transform"
                    new_args[0] = new_args[0].args[0]
                    new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                else:
                    new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                return new_call

            def visit_function(self, fn):
                new_params = []
                for x in fn.params:
                    new_params.append(self.visit(x))
                new_body = self.visit(fn.body)
                return relay.Function(new_params, new_body, fn.ret_type, fn.type_params, fn.attrs, fn.span)
        return RemoveLayoutTransform(mod).new_mod

    def tensorflow2relay(self):
        load_model_from_file = get_method("tvm.contrib.{}".format(self.logo_module), "load_model_from_file")
        extract_constants = get_method("tvm.contrib.{}.relay.transform".format(self.logo_module), "extract_constants")
        shape_dict = dict()
        for _, _input in enumerate(self.inputs):
            shape_dict[_input["name"]] = _input["shape"]
        mod, params = load_model_from_file(self.weight, "pb", shape_dict=shape_dict, layout="NHWC")
        self.relay, self.params = extract_constants(self._tf_convert_nhwc_to_nchw(mod, params))

    def tflite2relay(self):
        load_model_from_file = get_method("tvm.contrib.{}".format(self.logo_module), "load_model_from_file")
        extract_constants = get_method("tvm.contrib.{}.relay.transform".format(self.logo_module), "extract_constants")
        shape_dict = dict()
        for _, _input in enumerate(self.inputs):
            shape_dict[_input["name"]] = _input["shape"]
        mod, params = load_model_from_file(self.weight, "tflite", shape_dict, layout="NHWC")
        self.relay, self.params = extract_constants(self._tf_convert_nhwc_to_nchw(mod, params))

    def quantization(self, in_datas):
        """量化，将浮点relay函数转为成定点relay函数
        """
        if self.is_qnn:
            self.relay_quant, self.params_quant = self.relay, self.params
            logger.warning("Qnn model don`t need quantization, will skip tvm dump by layer")
            return

        t_start = time.time()
        if self.enable_quant:
            quantize_config, norm = self.set_quantization_cfg(in_datas)
            logger.info("################   quantization start  ######################")
            import tvm
            from tvm.relay.quantization import quantize
            self.relay_quant, self.params_quant = tvm.relay.quantization.quantize(
                self.relay,
                self.params,
                model_name="opt_ir",
                dataset=self.get_dataset(),
                prof_img_num=self.prof_img_num,
                rgb_en=1 if (self.num_inputs == 1 and self.inputs[0]["pixel_format"] == "RGB") else 0,
                norm=norm,
                quantize_config=quantize_config,
                debug_level=self.quant_cfg["debug_level"],
                save_dir=self.result_dir,
            )
            logger.info("################   quantization end  ######################")

            self.save_relay_to_json(self.quant_json_path, self.relay_quant, self.params_quant)
            self.save_relay_to_model(self.quant_json_path, self.relay_quant, self.params_quant)
        else:
            self.relay_quant = self.load_relay_from_json(self.quant_json_path)
        self.quantization_span = time.time() - t_start

        t_start = time.time()
        self.save_compare_layer_outputs()
        self.tvm_layerwise_dump_span = time.time() - t_start

    @staticmethod
    def compile_extern_lib(relay_func, lib, save_path):
        """Compile extern lib for custom ops."""
        import tvm
        addons = []
        for _, func in relay_func.functions.items():
            if isinstance(func, tvm.tir.PrimFunc):
                if "nnp_c_link_files" in func.attrs:
                    link_files = func.attrs["nnp_c_link_files"]
                    if link_files:
                        addons.extend(link_files)

        # prepare cc options
        cc_options = []
        if addons:
            cc_options.extend(
                ["-std=c++17", "-O2", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]
            )
            tvm_root_cands = [
                tvm.__path__[0],
                os.path.join(tvm.__path__[0], "..", ".."),
                os.environ.get("TVM_HOME", "."),
            ]
            for cand in tvm_root_cands:
                cc_options.append(f"-I{cand}/include")
                cc_options.append(f"-I{cand}/3rdparty/dlpack/include")
                cc_options.append(f"-I{cand}/3rdparty/dmlc-core/include")
        if save_path:
            lib.export_library(save_path, addons=addons, options=cc_options)
        return lib

    @staticmethod
    def build_x86_64(relay_func, params, target, save_path=""):
        try:
            import tvm
            from tvm.contrib import graph_executor
            from tvm.relay import build

            cpu_target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                cpu_lib = build(relay_func, target=cpu_target, params=params)
            cpu_lib = Nnp4xxTyExec.compile_extern_lib(relay_func, cpu_lib, save_path)
            # if save_path:
            #     cpu_lib.export_library(save_path)
            cpu_lib = tvm.runtime.load_module(save_path)
            module = graph_executor.GraphModule(cpu_lib["default"](tvm.cpu()))
            return module
        except Exception as e:
            logger.error("Failed to load model -> {}".format(e))
            exit(-1)

    def build(self, in_datas):
        t_start = time.time()
        if self.enable_build:
            import tvm
            optimize_and_compile = get_method("tvm.contrib.{}".format(self.logo_module), "optimize_and_compile")
            estimate_compiled_mod_Cycles = get_method("tvm.contrib.{}".format(self.logo_module), "estimate_compiled_mod_Cycles")
            estimate_compiled_mod_MACs = get_method("tvm.contrib.{}".format(self.logo_module), "estimate_compiled_mod_MACs")
            # TODO support c920
            if self.enable_dump == 0:
                export_lib_path = []
                target_host = []
                target_host_cc = []
            else:
                export_lib_path = [self.model_path_x86_64]
                target_host = ["llvm -mtriple=x86_64"]
                target_host_cc = [None]
            ARM_C_COMPILER = os.getenv("ARM_C_COMPILER")
            assert os.path.exists(ARM_C_COMPILER), "Not found ARM_C_COMPILER env"
            export_lib_path.append(self.model_path_aarch64)
            target_host.append("llvm -mtriple=aarch64")
            target_host_cc.append(ARM_C_COMPILER)

            config = {
                "tir.{}.EstimateCost.enable".format(self.logo_module): True,  #
                "tir.{}.CalculateMac.enable".format(self.logo_module): True,  #
            }
            if self.cfg["build"].get("multi_thread"):
                config["edgex.relay_to_tir.compile_thread"] = self.cfg["build"].get("multi_thread")
            num_cube = self.cfg["build"].get("num_cube")
            if num_cube:
                assert num_cube in [1, 2, 3]
                config["hardware.pe_num"] = num_cube
            target_device = tvm.target.Target(
                self.logo_module, host="{}_virtual_host".format(self.logo_module) if self.enable_dump != 0 else None)

            mods = optimize_and_compile(
                self.relay_quant,
                self.params_quant,
                working_dir=self.model_dir,
                export_lib_path=export_lib_path,
                opt_level=self.build_opt_level,
                target_host=target_host,
                target_host_cc=target_host_cc,
                target_device=target_device,
                config=config
            )
            logger.info("Executing model on {}...".format(self.logo_module))

            if isinstance(mods, list):
                mod = mods[0]
                cycles = estimate_compiled_mod_Cycles(mod)  #
                macs = estimate_compiled_mod_MACs(mod)  #
                with open(os.path.join(self.result_dir, "macs.json"), "w") as f:
                    f.write(json.dumps(macs, indent=2))
                with open(os.path.join(self.result_dir, "cycles.json"), "w") as f:
                    f.write(json.dumps(cycles, indent=2))
        else:
            logger.warning("nnp4xx disable build")
        self.build_span = time.time() - t_start

    def infer(self):
        from .nnp4xx_infer import Nnp4xxSdkInfer
        in_datas = self.get_datas(force_cr=True, to_file=False)
        infer = Nnp4xxSdkInfer(enable_dump=self.enable_dump, enable_aipp=True)
        infer.backend = self.backend
        model_path = self.model_path_x86_64 if infer.backend == "sdk_iss" else self.model_path_aarch64
        infer.load(model_path)
        outputs = infer.run(in_datas, to_file=True)
        infer.unload()
        ave_latency_ms = infer.ave_latency_ms
        # 清理profile输出
        if os.path.exists(infer.profile_dir):
            import shutil
            shutil.rmtree(infer.profile_dir, ignore_errors=True)
        logger.info("[{}] average cost: {:.3f}ms".format(self.target, ave_latency_ms))
        return outputs

    def iss_dump_output(self, in_datas):
        t_start = time.time()
        if self.enable_dump == 1:
            import pickle
            iss_layerwise_input_output = get_method("tvm.contrib.{}".format(self.logo_module), "iss_layerwise_input_output")
            layerwise_inputs, layerwise_outputs = iss_layerwise_input_output(in_datas, self.model_path_x86_64)
            with open(os.path.join(self.result_dir, "iss_fused_out.pickle"), "wb") as fp:
                pickle.dump(layerwise_outputs, fp)
            with open(os.path.join(self.result_dir, "iss_fused_in.pickle"), "wb") as fp:
                pickle.dump(layerwise_inputs, fp)
        self.iss_layerwise_dump_span = time.time() - t_start

    def tvm_float_inference(self, in_datas, to_file=False):
        logger.info("start tvm-float simu")
        tvm_float_outputs = self.tvm_inference(
            self.build_x86_64(self.relay, self.params, self.target, self.cpu_model_float_path), in_datas)
        if to_file and len(tvm_float_outputs) > 0:
            for idx, output in enumerate(tvm_float_outputs):
                txt_path = os.path.join(self.result_dir, "tvm_float_out_{}.bin".format(idx))
                bin_path = os.path.join(self.result_dir, "tvm_float_out_{}.txt".format(idx))
                output.tofile(txt_path)
                output.tofile(bin_path, sep="\n")
                logger.info("save tvm-float output[{}]: {}".format(idx, txt_path))
                logger.info("save tvm-float output[{}]: {}".format(idx, bin_path))
        logger.info("tvm-float simu successfully")
        return tvm_float_outputs

    def tvm_fixed_inference(self, in_datas, to_file=False):
        logger.info("start tvm-fixed simu")
        tvm_fixed_outputs = self.tvm_inference(
            self.build_x86_64(self.relay_quant, self.params_quant, self.target, self.cpu_model_fixed_path), in_datas)
        if to_file and len(tvm_fixed_outputs) > 0:
            for idx, output in enumerate(tvm_fixed_outputs):
                txt_path = os.path.join(self.result_dir, "tvm_fixed_out_{}.bin".format(idx))
                bin_path = os.path.join(self.result_dir, "tvm_fixed_out_{}.txt".format(idx))
                output.tofile(txt_path)
                output.tofile(bin_path, sep="\n")
                logger.info("save tvm-fixed output[{}]: {}".format(idx, txt_path))
                logger.info("save tvm-fixed output[{}]: {}".format(idx, bin_path))
        logger.info("tvm-fixed simu successfully")
        return tvm_fixed_outputs

    def iss_fixed_inference(self, in_datas, to_file=False):
        """x86_64 iss"""
        iss_fixed_outputs = 0
        if self.enable_dump:
            logger.info("start iss-fixed simu")
            if not os.path.exists(self.model_path_x86_64):
                logger.error("Not found model path -> {}".format(self.model_path_x86_64))
                exit(-1)
            t_start = time.time()
            import tvm
            from tvm.contrib import graph_executor
            logger.info("Executing model on {}...".format(self.logo_module))
            lib = tvm.runtime.load_module(self.model_path_x86_64)
            func = get_method("tvm", "{}".format(self.logo_module))
            module = graph_executor.GraphModule(lib["default"](func(), tvm.cpu()))
            iss_fixed_outputs = self.tvm_inference(module, in_datas)
            self.iss_simu_span = time.time() - t_start
            if to_file and len(iss_fixed_outputs) > 0:
                for idx, output in enumerate(iss_fixed_outputs):
                    txt_path = os.path.join(self.result_dir, "iss_fixed_out_{}.bin".format(idx))
                    bin_path = os.path.join(self.result_dir, "iss_fixed_out_{}.txt".format(idx))
                    output.tofile(txt_path)
                    output.tofile(bin_path, sep="\n")
                    logger.info("save iss-fixed output[{}]: {}".format(idx, txt_path))
                    logger.info("save iss-fixed output[{}]: {}".format(idx, bin_path))
            logger.info("iss-fixed simu successfully")
        return iss_fixed_outputs

    def save_compare_layer_outputs(self):
        if self.quant_cfg["debug_level"] == 1:
            # layer_outs: dict，key为每层的name(为方便用户获知原始浮点模型每一层的数据状况，
            # 所以尽可能使用了原始浮点模型自带的op_name，但实际处理会获取不到原始模型的op_name，
            # 此时会使用opt_ir.pdf中的op_name)，相应的value为浮点和定点结果的组成的list
            import tvm
            from tvm import relay
            layer_outs = tvm.relay.quantization.compare_layer_outputs(self.result_dir)

    def profile(self):
        from .nnp4xx_profiler import Nnp4xxSdkProfiler
        profiler = Nnp4xxSdkProfiler()
        in_datas = self.get_datas(force_cr=True, to_file=False)
        profiler.load(self.model_path_aarch64)
        profiler.run(in_datas)
        profiler.unload()
        profiler.parse()
        # 清理profile输出
        if os.path.exists(profiler.profile_dir):
            import shutil
            shutil.rmtree(profiler.profile_dir, ignore_errors=True)

    def get_relay_mac(self):
        logger.warning("Nnp4xx not support get relay mac")

    def get_device_type(self):
        logger.warning("Nnp4xx not support get device type")

    def compress_analysis(self):
        logger.warning("Nnp4xx not support compress analysis")

    def get_profile_info(self):
        pass

    @staticmethod
    def save_relay_to_model(quant_model_path, relay_func, params):
        logger.warning("Not support save relay to model, can be visualized by netron")

    def compare_layerwise(self):
        data_float = self.get_datas(force_float=True, force_cr=True, to_file=True)  # 浮点模型输入
        data_fixed = self.get_datas(force_float=False, force_cr=True, to_file=True)  # 量化后模型输入

        import tvm
        from tvm.relay.transform import SimplifyInference
        layerwise_error = get_method("tvm.contrib.{}.relay.analysis".format(self.logo_module), "LayerwiseError")
        compile_cpuref_model = get_method("tvm.contrib.{}".format(self.logo_module), "compile_cpuref_model")
        get_available_graph_spans = get_method("tvm.contrib.{}".format(self.logo_module), "get_available_graph_spans")
        get_cos_similarity_per_channel_average = get_method(
            "tvm.contrib.{}.testing".format(self.logo_module), "get_cos_similarity_per_channel_average")

        # 加载tvm浮点模型, 编译为包含span信息的cpu可执行模型
        relay_float = self.load_relay_from_json(self.original_json_path)
        callback = SimplifyInference()
        relay_float = callback(relay_float)
        tvm_float_lib = compile_cpuref_model(relay_float, params=None)
        span_infos = get_available_graph_spans(tvm_float_lib)
        for term in span_infos:
            logger.info(term)

        # 这些span名为示例要比对的中间层名字
        span_keys = list()
        for span_info in span_infos:
            if span_info["name"] not in span_keys:
                span_keys.append(span_info["name"])

        # onnx
        model = onnx.shape_inference.infer_shapes(onnx.load(self.weight))
        for node in model.graph.node:
            logger.info("op_names: {}, output_names: {}".format(node.name, node.output))
            for output in node.output:
                model.graph.output.insert(-1, onnx.ValueInfoProto(name=output))
        ort_session = onnxruntime.InferenceSession(model.SerializeToString())

        outputs = [x.name for x in ort_session.get_outputs()]
        ort_outs = ort_session.run(outputs, data_float)
        ort_outs = OrderedDict(zip(outputs, ort_outs))

        # 前端tvm float模型逐层结果
        _, _, _, tvm_float_outputs = layerwise_error.run(tvm_float_lib, inputs=data_float)

        # tvm量化模型逐层结果
        relay_quant = self.load_relay_from_json(self.quant_json_path)
        quant_lib = compile_cpuref_model(relay_float, params=None)
        _, _, _, quant_outputs = layerwise_error.run(quant_lib, inputs=data_fixed)

        # 图层融合和优化后模型CPU逐层结果
        build_config_nnp = get_method("tvm.contrib.{}".format(self.logo_module), "build_config_nnp")
        optimize_nnp_model = get_method("tvm.contrib.{}".format(self.logo_module), "optimize_nnp_model")
        with build_config_nnp(opt_level=self.build_opt_level):
            fuse_mod, fuse_params = optimize_nnp_model(relay_quant, None, tvm.target.edgex())
        fuse_lib = compile_cpuref_model(fuse_mod, params=fuse_params)
        _, _, _, fuse_outputs = layerwise_error.run(fuse_lib, inputs=data_fixed)

        # 获取模拟器上的逐层结果
        simu_outputs = dict()
        # lib_x86 = tvm.runtime.load_module(self.model_path_x86_64)
        # _, _, _, simu_outputs = layerwise_error.run(lib_x86, inputs=data_fixed)

        # 绘制比对表格
        summary = []
        for key in ort_outs:
            onnx_data = ort_outs[key]

            def compare(key, expect, outputs):
                if key not in outputs:
                    return "Missing"
                actual = outputs[key][0]
                cosine = get_cos_similarity_per_channel_average(expect, actual, layout="NCHW")
                if cosine is None:
                    return f"Mismatch shape: {actual.shape}"
                return "%.3f" % cosine

            summary.append([
                key,
                str(onnx_data.shape),
                compare(key, onnx_data, tvm_float_outputs),
                compare(key, onnx_data, quant_outputs),
                compare(key, onnx_data, fuse_outputs),
                compare(key, onnx_data, simu_outputs),
            ])

        import prettytable
        table = prettytable.PrettyTable()
        table.field_names = [
            "layer", "onnx_shape", "tvm-float vs onnx", "tvm-fixed vs onnx", "tvm-fused vs onnx", "tvm-iss vs onnx",]
        table.add_rows(summary)
        logger.info("\n{}".format(table))

