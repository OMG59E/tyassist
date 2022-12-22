#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_tyexec.py 
@time: 2022/12/14
@contact: xing.weiguo@intellif.com
@author: xingwg 
@site: www.intellif.com
@software: PyCharm 
"""
import os
from abc import ABC
from utils import logger
from .base_tyexec import BaseTyExec


class Nnp4xxTyExec(BaseTyExec, ABC):
    def __init__(self, cfg: dict):
        super(Nnp4xxTyExec, self).__init__(cfg)

    @staticmethod
    def set_nnp4xx_env():
        import tvm
        dep_path = "{}/de-dcl/client/lib".format(tvm.__path__[0])
        ld_path = os.getenv("LD_LIBRARY_PATH")
        ld_path = dep_path if ld_path is None else dep_path + ":" + ld_path
        os.environ["LD_LIBRARY_PATH"] = ld_path
        os.environ["EDGEX_DEBUG_ISS"] = "on"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    def get_version(self):
        from tvm.contrib.edgex import get_version
        logger.info("TyTVM Version: {}".format(get_version()))

    def onnx2relay(self):
        from tvm.contrib.edgex import load_model_from_file
        kwargs = {"dtype": self.dtype_dict}
        self.relay, self.params = load_model_from_file(self.weight, self.framework, self.shape_dict, **kwargs)

    def quantization(self, in_datas):
        """量化，将浮点relay函数转为成定点relay函数
        """
        if self.enable_quant:
            quantize_config, norm = self.set_quantization_cfg(in_datas)
            logger.info("################   quantization start  ######################")
            import tvm
            from tvm.relay.quantization import quantize
            self.relay_quant, self.params_quant = tvm.relay.quantization.quantize(
                self.relay,
                self.params,
                model_name="opt_ir",
                dataset=self.quant_cfg["data_dir"] if not self.has_custom_preprocess else self.custom_preprocess_cls.get_data,
                prof_img_num=self.quant_cfg["prof_img_num"],
                rgb_en=1 if (self.num_inputs == 1 and self.inputs[0]["pixel_format"] == "RGB"
                             and (not self.has_custom_preprocess)) else 0,
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

        self.save_compare_layer_outputs()

    @staticmethod
    def build_x86_64(relay_func, params, save_path=""):
        try:
            import tvm
            from tvm.contrib import graph_executor
            from tvm.relay import build
            cpu_target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                cpu_lib = build(relay_func, target=cpu_target, params=params)
                if save_path:
                    cpu_lib.export_library(save_path)
            module = graph_executor.GraphModule(cpu_lib["default"](tvm.cpu()))
            return module
        except Exception as e:
            logger.error("Failed to load model -> {}".format(e))
            exit(-1)

    def build(self, in_datas):
        if self.enable_build:
            from tvm.contrib.edgex import compile_nnp_model
            # compile edgex lib
            _ = compile_nnp_model(
                self.relay_quant,
                self.params_quant,
                working_dir=self.model_dir,
                export_lib_path="{}/{}.ty".format(self.model_dir, self.model_name),
                opt_level=2,
            )
            logger.info("Executing model on edgex...")
        else:
            logger.warning("nnp4xx disable build")

        iss_fixed_outputs = self.iss_fixed_inference(in_datas, to_file=True)
        return iss_fixed_outputs

    def tvm_float_inference(self, in_datas, to_file=False):
        tvm_float_outputs = self.tvm_inference(
            self.build_x86_64(self.relay, self.params, self.cpu_model_float_path), in_datas)
        if to_file and len(tvm_float_outputs) > 0:
            for idx, output in enumerate(tvm_float_outputs):
                output.tofile(os.path.join(self.result_dir, "tvm_float_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self.result_dir, "tvm_float_out_{}.txt".format(idx)), sep="\n")
        return tvm_float_outputs

    def tvm_fixed_inference(self, in_datas, to_file=False):
        tvm_fixed_outputs = self.tvm_inference(
            self.build_x86_64(self.relay_quant, self.params_quant, self.cpu_model_fixed_path), in_datas)
        if to_file and len(tvm_fixed_outputs) > 0:
            for idx, output in enumerate(tvm_fixed_outputs):
                output.tofile(os.path.join(self.result_dir, "tvm_fixed_out_{}.bin".format(idx)))
                output.tofile(os.path.join(self.result_dir, "tvm_fixed_out_{}.txt".format(idx)), sep="\n")
        return tvm_fixed_outputs

    def iss_fixed_inference(self, in_datas, to_file=False):
        """x86_64 iss"""
        model_path = "{}/{}.ty".format(self.model_dir, self.model_name)
        if not os.path.exists(model_path):
            logger.error("Not found model path -> {}".format(model_path))
            exit(-1)
        import tvm
        from tvm.contrib import graph_executor
        logger.info("Executing model on edgex...")
        lib = tvm.runtime.load_module(model_path)
        module = graph_executor.GraphModule(lib["default"](tvm.edgex(), tvm.cpu()))
        iss_fixed_outputs = self.tvm_inference(module, in_datas)
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

    def get_relay_mac(self):
        logger.warning("Nnp4xx not support get relay mac")

    def get_device_type(self):
        logger.warning("Nnp4xx not support get device type")

    def compress_analysis(self):
        logger.warning("Nnp4xx not support compress analysis")

    @staticmethod
    def save_relay_to_model(quant_model_path, relay_func, params):
        logger.warning("Not support save relay to model, can be visualized by netron")

    # def load(self, net_cfg_file, sdk_cfg_file, model_path, enable_aipp, enable_dump):
    #     self.sdk_enable_dump = enable_dump
    #     self.sdk_enable_aipp = enable_aipp
    #     if self.sdk_enable_aipp:
    #         logger.warning("Nnp4xx not support aipp")
    #     if self.sdk_enable_dump:
    #         logger.warning("Nnp4xx not support dump")
    #     try:
    #         import python._sdk as _sdk
    #         logger.info("sdk config path: {}".format(sdk_cfg_file))
    #         _sdk.init(sdk_cfg_file)
    #         logger.info("tyhcp init succeed.")
    #
    #         self.engine = _sdk.CNetOperator()
    #
    #         logger.info("load model " + model_path)
    #         if not self.engine.load(model_path):
    #             logger.error("Failed to load model")
    #             exit(-1)
    #         logger.info("load model success")
    #         self.sdk = True
    #     except Exception as e:
    #         logger.error("load failed -> {}".format(e))
    #         exit(-1)
    #
    # def infer(self, in_datas, to_file=False):
    #     outputs = self.engine.inference(in_datas)
    #     if to_file:
    #         logger.info("[{}] predict result: outputs size -> {}".format(self.prefix, len(outputs)))
    #         for idx, output in enumerate(outputs):
    #             logger.info("outputs[{}], shape={}, dtype={}".format(idx, output.shape, output.dtype))
    #             filepath_txt = os.path.join(self.result_dir, "{}_fixed_out_{}.txt".format(self.prefix, idx))
    #             filepath_bin = os.path.join(self.result_dir, "{}_fixed_out_{}.bin".format(self.prefix, idx))
    #             output.tofile(filepath_txt, sep="\n")
    #             output.tofile(filepath_bin)
    #             logger.info("save {}_fixed_output[{}] to {}".format(self.prefix, idx, filepath_txt))
    #             logger.info("save {}_fixed_output[{}] to {}".format(self.prefix, idx, filepath_bin))
    #     return outputs
    #
    # def unload(self):
    #     if self.engine:
    #         self.engine.unload()
    #         logger.info("unload model")
    #         self.engine = None
    #     if self.sdk:
    #         import python._sdk as _sdk
    #         _sdk.finalize()
    #         self.sdk = False
    #
    # def __del__(self):
    #     self.unload()
    #
    # def compare_layer_out(self):
    #     logger.warning("Nnp4xx not support compare")


