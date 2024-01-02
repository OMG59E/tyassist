#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_tyexec.py 
@time: 2022/12/14
@Author  : xingwg
@software: PyCharm 
"""
import os
import time
import json
import psutil
import importlib
import numpy as np
from collections import OrderedDict
from abc import ABC
from utils import logger
from utils.utils import get_method
from utils.dist_metrics import get_cos_similarity_per_channel_average
from .base_tyexec import BaseTyExec


class Nnp4xxTyExec(BaseTyExec, ABC):
    def __init__(self, cfg: dict):
        super(Nnp4xxTyExec, self).__init__(cfg)
        
        logo_setting_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "setting.cfg")
        assert os.path.exists(logo_setting_filepath)
        with open(logo_setting_filepath, "r") as f:
            setting = json.load(f)
            self.logo_module = setting["logo"]
        
        self.fused_json_path = os.path.join(self.result_dir, "model_fused.json")

        self.model_path_x86_64 = os.path.join(self.model_dir, "{}_O{}_x86_64.ty".format(self.model_name, self.build_opt_level))
        self.model_path_aarch64 = os.path.join(self.model_dir, "{}_O{}_aarch64.ty".format(self.model_name, self.build_opt_level))
              
        self.custom_op_module = self.cfg["model"].get("custom_op_module")

        ARM_C_COMPILER = os.getenv("ARM_C_COMPILER")
        if ARM_C_COMPILER is None:
            logger.error("Not found ARM_C_COMPILER ENV")
            exit(-1)
        elif not os.path.exists(ARM_C_COMPILER):
            logger.error("Not found ARM_C_COMPILER -> {}".format(ARM_C_COMPILER))
            exit(-1)

    @staticmethod
    def set_env():
        pass
        # os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    def get_version(self):
        get_version = get_method("tvm.contrib.{}".format(self.logo_module), "get_version")
        version_info = get_version()
        # version = version_info["TYTVM_VERSION"][6:]
        logger.info("TyTVM Version: {}".format(version_info))

    def onnx2relay(self):
        import onnx
        dtype_dict = dict()
        norm_dict = dict()
        qnn_in_dtype = dict()
        output_info = dict()
        for _, _input in enumerate(self.inputs):
            input_name = _input["name"]
            dtype_dict[input_name] = "float32"
            if _input["mean"]:
                norm_dict[input_name] = {"mean": _input["mean"], "std": _input["std"]}
            qnn_in_dtype[input_name] = "uint8" if _input["pixel_format"] in \
                                ["RGB", "BGR", "GRAY"] else _input["dtype"]

        kwargs = {"dtype": dtype_dict}
        if self.is_qnn:
            kwargs["qnn"] = self.is_qnn
            if len(norm_dict) > 0:
                kwargs["norm"] = norm_dict
            kwargs["net_in_dtype"] = qnn_in_dtype
            # kwargs["output_info"] = {"output_name": {"dtype": "float32", "skip_dequant": False}}

        if self.model_cfg.get("extras"):
            kwargs["extras"] = self.model_cfg.get("extras")
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

    def mxnet2relay(self):
        dtype_dict = dict()
        for _, _input in enumerate(self.inputs):
            dtype_dict[_input["name"]] = "float32"
        kwargs = {"dtype": dtype_dict}
        load_model_from_file = get_method("tvm.contrib.{}".format(self.logo_module), "load_model_from_file")
        self.relay, self.params = load_model_from_file(self.weight, "mxnet", self.shape_dict, **kwargs)

    def quantization(self, in_datas):
        """量化, 将浮点relay函数转为成定点relay函数
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
                norm=norm if len(norm) > 0 else None,
                quantize_config=quantize_config,
                debug_level=self.quant_debug_level,
                similarity_img_num=self.similarity_img_num,
                similarity_dataset=self.similarity_dataset,
                save_dir=self.result_dir,
            )
            logger.info("################   quantization end  ######################")
            self.save_relay_to_json(self.quant_json_path, self.relay_quant, self.params_quant)
            # self.save_relay_to_model(self.quant_json_path, self.relay_quant, self.params_quant)
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
            with tvm.transform.PassContext(opt_level=0):
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
            # if self.enable_dump == 0:
            #     export_lib_path = []
            #     target_host = []
            #     target_host_cc = []
            # else:
            #     export_lib_path = [self.model_path_x86_64]
            #     target_host = ["llvm -mtriple=x86_64"]
            #     target_host_cc = [None]
            export_lib_path = [self.model_path_x86_64]
            target_host = ["llvm -mtriple=x86_64"]
            target_host_cc = [None]
            ARM_C_COMPILER = os.getenv("ARM_C_COMPILER")
            assert os.path.exists(ARM_C_COMPILER), "Not found ARM_C_COMPILER env"
            export_lib_path.append(self.model_path_aarch64)
            target_host.append("llvm -mtriple=aarch64")
            target_host_cc.append(ARM_C_COMPILER)

            config = {
                "tir.{}.EstimateCost.enable".format(self.logo_module): True,
                "tir.{}.CalculateMac.enable".format(self.logo_module): True,
                "{}.relay_to_tir.collect_lower_errors".format(self.logo_module): False,
                "tir.{}.InjectCheckpoint.enable".format(self.logo_module): False,
                # "relay.{}.byoa".format(self.logo_module): False,
            }
            multi_thread = self.cfg["build"].get("multi_thread")
            if multi_thread is not None:
                config["{}.relay_to_tir.compile_thread".format(self.logo_module)] = multi_thread
            else:
                multi_thread = psutil.cpu_count(False)
            logger.info("Build thread: {}".format(multi_thread))
            # fuse_subgraph_ms_threshold
            fuse_subgraph_ms_threshold = self.cfg["build"].get("fuse_subgraph_ms_threshold")
            if fuse_subgraph_ms_threshold is not None:
                if fuse_subgraph_ms_threshold <= 0:
                    logger.error("fuse_subgraph_ms_threshold must be > 0")
                    exit(-1)
                config["{}.FuseSubgraph.ms_threshold".format(self.logo_module)] = float(fuse_subgraph_ms_threshold)
                logger.info("{}.FuseSubgraph.ms_threshold: {}ms".format(self.logo_module, fuse_subgraph_ms_threshold))
            num_cube = self.cfg["build"].get("num_cube")
            if num_cube:
                assert num_cube in [1, 2, 3]
                config["hardware.pe_num"] = num_cube
                logger.info("hardware.pe_num: {}".format(num_cube))
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

    def infer(self, device_id, node_id):
        from .nnp4xx_infer import Nnp4xxSdkInfer
        in_datas = self.get_datas(force_cr=True, to_file=False)
        infer = Nnp4xxSdkInfer(enable_dump=self.enable_dump, device_id=device_id, node_id=node_id)
        infer.backend = self.backend
        model_path = self.model_path_x86_64 if infer.backend == "sdk_iss" else self.model_path_aarch64
        infer.load(model_path)
        outputs = infer.run(in_datas, to_file=True)
        infer.unload()
        ave_latency_ms = infer.ave_latency_ms
        if infer.enable_dump:
            if infer.backend == "chip":
                infer.compare_layer_out()
            elif infer.backend == "sdk_iss":
                logger.warning("Inference time cannot be output when enable_dump == 1")
            else:
                logger.error("Not support backend -> {}".format(self.backend))
        # # 清理profile输出
        # if os.path.exists(infer.profile_dir):
        #     import shutil
        #     shutil.rmtree(infer.profile_dir, ignore_errors=True)
        logger.info("[{}] python infer average cost: {:.3f}ms".format(self.target, ave_latency_ms))
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
        t_start = time.time()
        tvm_float_outputs = self.tvm_inference(
            self.build_x86_64(self.relay, self.params, self.target, self.cpu_model_float_path), in_datas)
        self.tvm_float_simu_span = time.time() - t_start
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
        t_start = time.time()
        tvm_fixed_outputs = self.tvm_inference(
            self.build_x86_64(self.relay_quant, self.params_quant, self.target, self.cpu_model_fixed_path), in_datas)
        self.tvm_fixed_simu_span = time.time() - t_start
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
        if self.quant_debug_level == 1:
            # layer_outs: dict，key为每层的name(为方便用户获知原始浮点模型每一层的数据状况，
            # 所以尽可能使用了原始浮点模型自带的op_name，但实际处理会获取不到原始模型的op_name，
            # 此时会使用opt_ir.pdf中的op_name)，相应的value为浮点和定点结果的组成的list
            import tvm
            from tvm import relay
            layer_outs = tvm.relay.quantization.compare_layer_outputs(self.result_dir)

    def profile(self, device_id, node_id):
        from .nnp4xx_profiler import Nnp4xxSdkProfiler
        profiler = Nnp4xxSdkProfiler(device_id=device_id, node_id=node_id)
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
        mac_path = os.path.join(self.result_dir, "macs.json")
        if not os.path.exists(mac_path):
            logger.warning("Not found -> {}".format(mac_path))
            return
        with open(mac_path, "rb") as f:
            compiled_model_MACs_info = json.load(f)
        compiled_model_MACs = 0
        for key in compiled_model_MACs_info:
            compiled_model_MACs += compiled_model_MACs_info[key]  # 工具链内乘加算1次
        import tvm
        estimate_origin_mod_FLOPs = get_method("tvm.contrib.{}".format(self.logo_module), "estimate_origin_mod_FLOPs")
        original_model_MACs_info = estimate_origin_mod_FLOPs(self.relay)
        original_model_MACs_total_info = original_model_MACs_info["total"]
        original_model_MACs = 0
        for key in original_model_MACs_total_info:
            original_model_MACs += original_model_MACs_total_info[key]
        logger.info("Original model MACs: {}".format(original_model_MACs))
        logger.info("Compiled model MACs: {}".format(compiled_model_MACs))

    def get_device_type(self):
        logger.warning("Nnp4xx not support get device type")

    def compress_analysis(self):
        logger.warning("Nnp4xx not support compress analysis")

    def get_profile_info(self):
        pass

    @staticmethod
    def save_relay_to_model(quant_model_path, relay_func, params):
        logger.warning("Not support save relay to model, can be visualized by netron")

    @staticmethod
    def get_spans(graph_json: dict):
        import base64
        results = list()
        for idx, node in enumerate(graph_json["nodes"]):
            if node["op"] == "tvm_op":
                func_name = node["attrs"]["func_name"]
                if "LoweredFunctionNameHint" in node["attrs"]:
                    func_name = node["attrs"]["LoweredFunctionNameHint"]
                if "span_info" in node["attrs"]:
                    span_info = node["attrs"]["span_info"]
                    try:
                        span_info = json.loads(base64.b64decode(span_info))
                        for span_name, out_idx in span_info.items():
                            span_dict = dict()
                            span_dict["name"] = span_name
                            span_dict["func_name"] = func_name
                            span_dict["output_idx"] = out_idx
                            span_dict["node_idx"] = idx
                            results.append(span_dict)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.warning(f"Illegal span info: {span_info}:\n{e}")
        return results

    def compare_layerwise(self, filepath=None, backend="iss"):
        """
        onnx-float vs tvm-float vs tvm-fixed vs iss-fixed vs chip-fixed
        """
        data_float = self.get_datas(filepath=filepath, use_norm=True, force_cr=True, to_file=False)  # 浮点模型输入
        data_fixed = self.get_datas(filepath=filepath, use_norm=False, force_cr=True, to_file=False)  # 量化后模型输入
        
        # 获取芯片逐层输出
        chip_spans = dict()
        if backend == "chip":
            try:
                logger.info("Step1 - inference chip start")
                t = time.time()
                from .nnp4xx_infer import Nnp4xxSdkInfer
                infer = Nnp4xxSdkInfer(enable_dump=1, device_id=0, node_id=0)
                infer.backend = self.backend
                infer.load(self.model_path_aarch64)
                json_graph = infer.engine.get_json_graph()
                chip_spans = self.get_spans(json.loads(json_graph))
                outputs = infer.run(data_fixed, to_file=False)
                infer.unload()
                logger.info("Step1 - inference chip finished, and time: {:.3f}s".format(time.time() - t))
                #
                dump_root_path = infer.dump_root_path
                if not os.path.exists(dump_root_path):
                    logger.error("Not found dump_root_path: {}".format(dump_root_path))
                    exit(-1)
            except Exception as e:
                pass
        
        import tvm
        from tvm.relay.transform import SimplifyInference
        layerwise_error = get_method("tvm.contrib.{}.relay.analysis".format(self.logo_module), "LayerwiseError")
        compile_cpuref_model = get_method("tvm.contrib.{}".format(self.logo_module), "compile_cpuref_model")
        get_available_graph_spans = get_method("tvm.contrib.{}".format(self.logo_module), "get_available_graph_spans")

        self.get_version()
        
        # onnx
        logger.info("Step2 - inference onnx start")
        t = time.time()
        import onnx
        import onnxruntime
        model = onnx.shape_inference.infer_shapes(onnx.load(self.weight))
        ops_kv = dict()
        for node in model.graph.node:
            logger.info("op_name: {}, output_names: {}".format(node.name, node.output))
            for output in node.output:
                # ops_kv[node.name] = node.output
                for idx, out_name in enumerate(node.output):
                    ops_kv[out_name] = (node.name, idx)
                model.graph.output.insert(-1, onnx.ValueInfoProto(name=output))
        ort_session = onnxruntime.InferenceSession(model.SerializeToString())
                
        outputs = list()
        for output in ort_session.get_outputs():
            if output.name not in outputs:
                outputs.append(output.name)
        ort_outs = ort_session.run(outputs, data_float)
        ort_outs = OrderedDict(zip(outputs, ort_outs))
        logger.info("Step2 - inference onnx finished, and time: {:.3f}s".format(time.time() - t))

        tvm_float_outputs = dict()
        span_infos = list()
        if not self.is_qnn:
            logger.info("Step3 - inference tvm-float start")
            t = time.time()
            # 非qnn模型，加载tvm浮点模型, 编译为包含span信息的cpu可执行模型
            relay_float = self.load_relay_from_json(self.original_json_path)
            callback = SimplifyInference()
            relay_float = callback(relay_float)
            tvm_float_lib = compile_cpuref_model(relay_float, params=None)
            # 前端tvm float模型逐层结果
            _, _, _, tvm_float_outputs = layerwise_error.run(tvm_float_lib, inputs=data_float)
            logger.info("Step3 - inference tvm-float finished, and time: {:.3f}s".format(time.time() - t))

        # tvm量化模型逐层结果
        logger.info("Step4 - inference tvm-fixed start")
        t = time.time()
        relay_quant = self.load_relay_from_json(self.quant_json_path if not self.is_qnn else self.original_json_path)
        quant_lib = compile_cpuref_model(relay_quant, params=None)
        _, _, _, quant_outputs = layerwise_error.run(quant_lib, inputs=data_fixed)
        logger.info("Step4 - inference tvm-fixed finished, and time: {:.3f}s".format(time.time() - t))

        # 图层融合和优化后模型CPU逐层结果
        logger.info("Step5 - inference tvm-fused start")
        t = time.time()
        build_config_nnp = get_method("tvm.contrib.{}".format(self.logo_module), "build_config_nnp")
        optimize_nnp_model = get_method("tvm.contrib.{}".format(self.logo_module), "optimize_nnp_model")
        target_device = tvm.target.Target(self.logo_module)
        with build_config_nnp(opt_level=self.build_opt_level):
            fuse_mod, fuse_params = optimize_nnp_model(relay_quant, None, target_device)
        fuse_lib = compile_cpuref_model(fuse_mod, params=fuse_params)
        span_infos = get_available_graph_spans(fuse_lib)
        spans_dict = dict()
        for item in span_infos:
            print(item)
            spans_dict[item["name"]] = item["func_name"]
        _, _, _, fuse_outputs = layerwise_error.run(fuse_lib, inputs=data_fixed)
        logger.info("Step5 - inference tvm-fused finished, and time: {:.3f}s".format(time.time() - t))

        # 获取模拟器上的逐层结果
        logger.info("Step6 - inference tvm-iss start")
        t = time.time()
        simu_outputs = dict()
        if backend in ["iss", "chip"] and self.enable_dump in [1, 2]:
            lib_x86 = tvm.runtime.load_module(self.model_path_x86_64)
            _, _, _, simu_outputs = layerwise_error.run(lib_x86, inputs=data_fixed)
        logger.info("Step6 - inference tvm-iss finished, and time: {:.3f}s".format(time.time() - t))

        chip_outs = dict()
        for term in chip_spans:
            name = term["name"]
            func_name = term["func_name"]
            output_idxes = term["output_idx"]
            res = list()
            for idx in output_idxes:
                json_path = os.path.join(dump_root_path, "{}_out{}.json".format(func_name, idx))
                bin_path = os.path.join(dump_root_path, "{}_out{}.bin".format(func_name, idx))
                if not os.path.exists(json_path):
                    logger.warning("Not found chip output:", json_path)
                    continue
                if not os.path.exists(bin_path):
                    logger.warning("Not found chip output:", bin_path)
                    continue
                with open(json_path, "rb") as f:
                    out_infos = json.load(f)
                dtype = out_infos["dtype"]
                shape = out_infos["shape"]
                data = np.fromfile(bin_path, dtype=dtype).reshape(shape)
                res.append(data)
            chip_outs[name] = res

        # 绘制比对表格
        summary = []
        for output_name in ort_outs:
            onnx_data = ort_outs[output_name]  # output_name

            def compare(expect, actual):
                cosine = get_cos_similarity_per_channel_average(expect, actual, layout="NCHW")
                if cosine is None:
                    return f"Mismatch shape: {actual.shape}"
                return "%.6f" % cosine

            # 找到output_name的对应opname
            opname, idx = ops_kv[output_name]
            summary.append([
                "{}_out{}".format(opname, idx),
                "Missing" if opname not in spans_dict else spans_dict[opname],
                str(onnx_data.shape),
                "Missing" if opname not in tvm_float_outputs else compare(onnx_data, tvm_float_outputs[opname][idx]),
                "Missing" if opname not in quant_outputs else compare(onnx_data, quant_outputs[opname][idx]),
                "Missing" if opname not in fuse_outputs else compare(onnx_data, fuse_outputs[opname][idx]),
                "Missing" if opname not in simu_outputs else compare(onnx_data, simu_outputs[opname][idx]),
                "Missing" if opname not in chip_outs else compare(onnx_data, chip_outs[opname][idx]),
            ])

        import prettytable
        table = prettytable.PrettyTable()
        table.field_names = [
            "opanme", "func_name", "onnx_shape", "tvm-float vs onnx", "tvm-fixed vs onnx", "tvm-fused vs onnx", "iss vs onnx", "chip vs onnx"]
        table.add_rows(summary)
        logger.info("\n{}".format(table))

