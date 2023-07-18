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
from abc import ABC
from utils import logger
from .base_tyexec import BaseTyExec


class Nnp4xxTyExec(BaseTyExec, ABC):
    def __init__(self, cfg: dict):
        super(Nnp4xxTyExec, self).__init__(cfg)

        self.model_path_x86_64 = os.path.join(self.model_dir, "{}_x86_64.ty".format(self.model_name))
        self.model_path_aarch64 = os.path.join(self.model_dir, "{}_aarch64.ty".format(self.model_name))

        ARM_C_COMPILER = os.getenv("ARM_C_COMPILER")
        if ARM_C_COMPILER is None:
            logger.error("Not found ARM_C_COMPILER ENV")
            exit(-1)
        elif not os.path.exists(ARM_C_COMPILER):
            logger.error("Not found ARM_C_COMPILER -> {}".format(ARM_C_COMPILER))
            exit(-1)

    @staticmethod
    def set_env():
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    def get_version(self):
        # m = importlib.import_module("tvm.contrib.{}".format("fuxiao"))
        from tvm.contrib.fuxiao import get_version
        logger.info("TyTVM Version: {}".format(get_version()))

    def onnx2relay(self, is_qnn=False):
        from tvm.contrib.fuxiao import load_model_from_file
        dtype_dict = dict()
        for _, _input in enumerate(self.inputs):
            dtype_dict[_input["name"]] = "float32"
        kwargs = {"dtype": dtype_dict}
        self.relay, self.params = load_model_from_file(self.weight, "onnx", self.shape_dict, is_qnn=is_qnn, **kwargs)

    def onnx_qnn2relay(self):
        self.onnx2relay(is_qnn=True)
        self.is_qnn = True

    def pytorch2relay(self):
        from tvm.contrib.fuxiao import load_model_from_file
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
        from tvm.contrib.fuxiao import load_model_from_file
        from tvm.contrib.fuxiao.relay.transform import extract_constants
        shape_dict = dict()
        for _, _input in enumerate(self.inputs):
            shape_dict[_input["name"]] = _input["shape"]
        mod, params = load_model_from_file(self.weight, "pb", shape_dict=shape_dict, layout="NHWC")
        self.relay, self.params = extract_constants(self._tf_convert_nhwc_to_nchw(mod, params))

    def tflite2relay(self):
        from tvm.contrib.fuxiao import load_model_from_file
        from tvm.contrib.fuxiao.relay.transform import extract_constants
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
                prof_img_num=self.quant_cfg["prof_img_num"],
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
    def build_x86_64(relay_func, params, target, save_path=""):
        try:
            import tvm
            from tvm.contrib import graph_executor
            from tvm.relay import build

            # 将标量转换成张量
            # def rewrite_scalar(mod):
            #     class ScalarRewriter(tvm.relay.ExprMutator):
            #         def visit_constant(self, const):
            #             if len(const.data.shape) == 0:
            #                 return tvm.relay.const([const.data.asnumpy()], const.data.dtype)
            #             return super().visit_constant(const)
            #
            #     mod = tvm.IRModule.from_expr(ScalarRewriter().visit(mod["main"]))
            #     return tvm.relay.transform.InferType()(mod)

            cpu_target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                # cpu_lib = build(rewrite_scalar(relay_func), target=cpu_target, params=params)
                cpu_lib = build(relay_func, target=cpu_target, params=params)
                if save_path:
                    cpu_lib.export_library(save_path)
            module = graph_executor.GraphModule(cpu_lib["default"](tvm.cpu()))
            return module
        except Exception as e:
            logger.error("Failed to load model -> {}".format(e))
            exit(-1)

    def build(self, in_datas):
        t_start = time.time()
        if self.enable_build:
            import tvm
            from tvm.contrib.fuxiao import compile_nnp_model, optimize_nnp_model, \
                optimize_and_compile, estimate_compiled_mod_Cycles, estimate_compiled_mod_MACs
            # TODO support c920
            export_lib_path = [self.model_path_x86_64]
            target_host = ["llvm -mtriple=x86_64"]
            target_host_cc = [None]
            ARM_C_COMPILER = os.getenv("ARM_C_COMPILER")
            assert os.path.exists(ARM_C_COMPILER), "Not found ARM_C_COMPILER env"
            export_lib_path.append(self.model_path_aarch64)
            target_host.append("llvm -mtriple=aarch64")
            target_host_cc.append(ARM_C_COMPILER)

            config = {
                "tir.fuxiao.EstimateCost.enable": True,  #
                "tir.fuxiao.CalculateMac.enable": True,  #
            }
            target_device = tvm.target.Target("fuxiao", host="fuxiao_virtual_host")

            # compile fuxiao lib
            fuxiao_x86_lib, fuxiao_a55_lib = optimize_and_compile(
                self.relay_quant,
                self.params_quant,
                working_dir=self.model_dir,
                export_lib_path=export_lib_path,
                opt_level=self.build_opt_level,
                target_host=target_host,
                target_host_cc=target_host_cc,
                target_device=target_device,
                extra_config=config
            )
            logger.info("Executing model on fuxiao...")

            # cycles = estimate_compiled_mod_Cycles(fuxiao_x86_lib)  #
            # macs = estimate_compiled_mod_MACs(fuxiao_x86_lib)  #
            # with open(os.path.join(self.result_dir, "macs.json"), "w") as f:
            #     f.write(json.dumps(macs, indent=2))
            # with open(os.path.join(self.result_dir, "cycles.json"), "w") as f:
            #     f.write(json.dumps(cycles, indent=2))
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
        logger.info("[{}] average cost: {:.3f}ms".format(self.target, ave_latency_ms))
        return outputs

    def iss_dump_output(self, in_datas):
        t_start = time.time()
        if self.enable_dump == 1:
            import pickle
            from tvm.contrib.fuxiao import iss_layerwise_input_output
            layerwise_inputs, layerwise_outputs = iss_layerwise_input_output(in_datas, self.model_path_x86_64)
            with open(os.path.join(self.result_dir, "iss_fused_out.pickle"), "wb") as fp:
                pickle.dump(layerwise_outputs, fp)
            with open(os.path.join(self.result_dir, "iss_fused_in.pickle"), "wb") as fp:
                pickle.dump(layerwise_inputs, fp)
        self.iss_layerwise_dump_span = time.time() - t_start

    def tvm_float_inference(self, in_datas, to_file=False):
        tvm_float_outputs = self.tvm_inference(
            self.build_x86_64(self.relay, self.params, self.target, self.cpu_model_float_path), in_datas)
        if to_file and len(tvm_float_outputs) > 0:
            for idx, output in enumerate(tvm_float_outputs):
                output.tofile(os.path.join(self.result_dir, "tvm_float_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self.result_dir, "tvm_float_out_{}.txt".format(idx)), sep="\n")
        return tvm_float_outputs

    def tvm_fixed_inference(self, in_datas, to_file=False):
        tvm_fixed_outputs = self.tvm_inference(
            self.build_x86_64(self.relay_quant, self.params_quant, self.target, self.cpu_model_fixed_path), in_datas)
        if to_file and len(tvm_fixed_outputs) > 0:
            for idx, output in enumerate(tvm_fixed_outputs):
                output.tofile(os.path.join(self.result_dir, "tvm_fixed_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self.result_dir, "tvm_fixed_out_{}.txt".format(idx)), sep="\n")
        return tvm_fixed_outputs

    def iss_fixed_inference(self, in_datas, to_file=False):
        """x86_64 iss"""
        iss_fixed_outputs = 0
        if self.enable_dump:
            if not os.path.exists(self.model_path_x86_64):
                logger.error("Not found model path -> {}".format(self.model_path_x86_64))
                exit(-1)
            t_start = time.time()
            import tvm
            from tvm.contrib import graph_executor
            logger.info("Executing model on fuxiao...")
            lib = tvm.runtime.load_module(self.model_path_x86_64)
            module = graph_executor.GraphModule(lib["default"](tvm.fuxiao(), tvm.cpu()))
            iss_fixed_outputs = self.tvm_inference(module, in_datas)
            self.iss_simu_span = time.time() - t_start
            if to_file and len(iss_fixed_outputs) > 0:
                for idx, output in enumerate(iss_fixed_outputs):
                    output.tofile(os.path.join(self.result_dir, "iss_fixed_out_{}.bin".format(idx)))
                    output.tofile(os.path.join(self.result_dir, "iss_fixed_out_{}.txt".format(idx)), sep="\n")
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

    def get_relay_mac(self):
        logger.warning("Nnp4xx not support get relay mac")

    def get_device_type(self):
        logger.warning("Nnp4xx not support get device type")

    def compress_analysis(self):
        logger.warning("Nnp4xx not support compress analysis")

    def get_profile_info(self):
        # import tvm
        # from tvm.contrib.fuxiao import estimate_FLOPs
        # from tvm.contrib.fuxiao import estimate_cycles
        # from tvm.contrib.fuxiao import build_config_nnp, optimize_nnp_model
        # flops = estimate_FLOPs(self.relay)
        # target_device = tvm.target.Target("fuxiao", host="fuxiao_virtual_host")
        # with build_config_nnp():
        #     optimized_mod, optimized_params = optimize_nnp_model(
        #         self.relay_quant,
        #         self.params_quant,
        #         target_device=target_device,
        #         keep_params=True,
        #     )
        # cycles = estimate_cycles(optimized_mod)
        # with open(os.path.join(self.result_dir, "flops.json"), "w") as f:
        #     f.write(json.dumps(flops, indent=2))
        # with open(os.path.join(self.result_dir, "cycles.json"), "w") as f:
        #     f.write(json.dumps(cycles, indent=2))
        pass

    @staticmethod
    def save_relay_to_model(quant_model_path, relay_func, params):
        logger.warning("Not support save relay to model, can be visualized by netron")
