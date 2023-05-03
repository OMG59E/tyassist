#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_infer.py 
@time: 2022/11/30
@Author  : xingwg
@software: PyCharm 
"""
import os
import pickle
import uuid
import json
import traceback
import csv
import numpy as np
from prettytable import PrettyTable
from abc import ABC
from utils import logger
from .base_infer import BaseInfer
from .nnp4xx_tyexec import Nnp4xxTyExec
from .nnp3xx_infer import Nnp3xxTvmInfer
from .nnp4xx_profiler import Nnp4xxProfileTypeEnum
from utils.dist_metrics import cosine_distance
from utils.compare import compare_dump_out2


class Nnp4xxSdkInfer(BaseInfer, ABC):
    def __init__(
            self,
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg",
            enable_dump=0,
            enable_aipp=False
    ):
        super(Nnp4xxSdkInfer, self).__init__()

        self.sdk_cfg_file = sdk_cfg_file

        with open(self.sdk_cfg_file) as f:
            cfg = json.load(f)
        # self.profile_dir = cfg["profiler"]["host_output"]
        self.ip = cfg["rpc"]["ip_addr"]
        self.backend = "chip"
        if self.ip == "127.0.0.1":   # TODO 非127.0.0.1的地址也可能是ISS服务
            self.backend = "sdk_iss"

        self.enable_dump = True if enable_dump == 1 else False
        self.enable_aipp = enable_aipp

        self.dump_root_path = ""
        self.result_dir = ""
        self.uuid = str(uuid.uuid1())
        self.profile_dir = os.path.join("/tmp", self.uuid)
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)

    def load(self, model_path):
        self.result_dir = os.path.join(os.path.dirname(model_path), "result")
        self.dump_root_path = os.path.join(self.result_dir, "chip_dump_out")
        if not os.path.exists(self.dump_root_path):
            os.makedirs(self.dump_root_path)
        if not os.path.exists(self.result_dir):
            logger.error("Not found result_dir -> {}".format(self.result_dir))
            exit(-1)

        if not os.path.isfile(model_path):
            logger.error("netbin_file not file -> {}".format(model_path))
            exit(-1)

        try:
            import python._sdk as _sdk
            logger.info("sdk config path: {}".format(self.sdk_cfg_file))
            _sdk.init(self.sdk_cfg_file)
            logger.info("tyhcp init succeed.")

            self.engine = _sdk.CNetOperator()

            if self.backend != "sdk_iss":
                if not self.engine.profile(Nnp4xxProfileTypeEnum.DCL_PROF_DCL_API, self.profile_dir):  # profile
                    logger.error("Failed to set profile")
                    exit(-1)

            logger.info("load model " + model_path)
            if not self.engine.load(model_path, self.enable_dump, self.dump_root_path):
                logger.error("Failed to load model")
                exit(-1)
            logger.info("load model success")
        except Exception as e:
            logger.error("load failed -> {}".format(e))
            exit(-1)

    def run(self, in_datas: dict, to_file=False):
        if isinstance(in_datas, dict):
            in_datas = [in_datas[key] for key in in_datas]  # to list
        outputs = self.engine.inference(in_datas)
        self.total += 1

        if self.enable_dump == 1:
            if self.backend == "chip":
                self.compare_layer_out()
            elif self.backend == "sdk_iss":
                logger.warning("Inference time cannot be output when enable_dump == 1")
            else:
                logger.error("Not support backend -> {}".format(self.backend))

        if to_file:
            logger.info("[{}] predict result: outputs size -> {}".format(self.backend, len(outputs)))
            for idx, output in enumerate(outputs):
                logger.info("outputs[{}], shape={}, dtype={}".format(idx, output.shape, output.dtype))
                filepath_txt = os.path.join(self.result_dir, "{}_fixed_out_{}.txt".format(self.backend, idx))
                filepath_bin = os.path.join(self.result_dir, "{}_fixed_out_{}.bin".format(self.backend, idx))
                output.tofile(filepath_txt, sep="\n")
                output.tofile(filepath_bin)
                logger.info("save {}_fixed_output[{}] to {}".format(self.backend, idx, filepath_txt))
                logger.info("save {}_fixed_output[{}] to {}".format(self.backend, idx, filepath_bin))
        return outputs

    def unload(self):
        if self.engine:
            self.engine.unload()
            logger.info("unload model")
            self.engine = None
            import python._sdk as _sdk
            _sdk.finalize()

    def __del__(self):
        self.unload()

    def compare_layer_out(self):
        if not os.path.exists(self.dump_root_path):
            logger.error("Not found model dump path -> {}".format(self.dump_root_path))
            return

        iss_fixed_dump_out = os.path.join(self.result_dir, "iss_fused_out.pickle")
        if not os.path.exists(iss_fixed_dump_out):
            logger.error("Not found iss_fixed_dump_out -> {}".format(iss_fixed_dump_out))
            exit(-1)

        logger.info("Layer compare ==> iss[fixed]  vs chip[fixed]")
        with open(iss_fixed_dump_out, "rb") as f:
            iss_fixed_dump_out = pickle.load(f)

        header = ["Id", "OpName", "Dtype", "Shape", "ISS(fixed) vs Chip(fixed)"]
        table = PrettyTable(header)
        csv_filepath = os.path.join(self.dump_root_path, "similarity.csv")
        f = open(csv_filepath, "w")
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        for op_idx, opname in enumerate(iss_fixed_dump_out):
            iss_fixed_outs = iss_fixed_dump_out[opname]
            for out_idx in range(len(iss_fixed_outs)):
                iss_fixed_out = iss_fixed_outs[out_idx].flatten()
                chip_out_path = os.path.join(self.dump_root_path, "{}_out{}.bin".format(opname, out_idx))
                assert os.path.isfile(chip_out_path), "chip_out_path -> {}".format(chip_out_path)
                shape = iss_fixed_out.shape
                dtype = iss_fixed_out.dtype
                chip_fixed_out = np.fromfile(chip_out_path, dtype=iss_fixed_out.dtype)
                chip_fixed_vs_iss_fixed_dist = "{:.6f}".format(cosine_distance(iss_fixed_out, chip_fixed_out))
                result = [op_idx, opname, dtype, shape, chip_fixed_vs_iss_fixed_dist]
                table.add_row(result)
                f_csv.writerow(result)
        f.close()
        logger.info("\n{}".format(table))

        tvm_fixed_dump_out = os.path.join(self.result_dir, "quant", "output_tensors.params")
        if not os.path.exists(tvm_fixed_dump_out):
            logger.warning("Not found tvm_fixed_dump_out -> {}".format(tvm_fixed_dump_out))
            tvm_fixed_dump_out = None
        tvm_fp32_dump_out = os.path.join(self.result_dir, "fp32", "output_tensors.params")
        if not os.path.exists(tvm_fp32_dump_out):
            logger.warning("Not found tvm_fp32_dump_out -> {}".format(tvm_fp32_dump_out))
            tvm_fp32_dump_out = None
        if tvm_fp32_dump_out and tvm_fixed_dump_out:
            logger.info("Layer compare ==> tvm[fixed] vs tvm[float]")
            compare_dump_out2(tvm_fp32_dump_out, tvm_fixed_dump_out)

    @property
    def ave_latency_ms(self):
        if self.backend == "sdk_iss" or self.enable_dump:
            return 0

        profile_file = os.path.join(self.profile_dir, "model_prof.bin")
        if not os.path.exists(profile_file):
            logger.error("Not found profile file -> {}".format(profile_file))
            exit(-1)

        try:
            import python._sdk as _sdk
            profile_json = _sdk.parse_dcl_api(profile_file)
            profile = json.loads(profile_json)
            total_time = profile["dclmdlExecute"] / 10**6 / self.total
            return total_time

        except Exception as e:
            logger.error("Failed to parse profile -> {}\n{}".format(e, traceback.format_exc()))
            exit(-1)


class Nnp4xxTvmInfer(Nnp3xxTvmInfer, ABC):
    def __init__(self):
        super().__init__()

    def load_json(self, model_path):
        import tvm
        import tvm.relay.quantization
        from tvm import relay
        relay_func = tvm.relay.quantization.get_ir_from_json(model_path)
        self.engine = Nnp4xxTyExec.build_x86_64(relay_func, {}, self.target)

    def load(self, model_path):
        import tvm
        from tvm.contrib import graph_executor
        lib = tvm.runtime.load_module(model_path)
        self.engine = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu()))
