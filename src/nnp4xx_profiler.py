#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@file: nnp4xx_profiler.py 
@time: 2022/12/22
@Author  : xingwg
@software: PyCharm 
"""
import os
import abc
import json
import time
import shutil
import uuid
import traceback
from .base_profiler import BaseSdkProfiler
from utils import logger


class Nnp4xxProfileTypeEnum(object):
    DCL_PROF_DCL_API = 0x0001
    DCL_PROF_TASK_TIME = 0x0002
    DCL_PROF_AICORE_METRICS = 0x0004
    DCL_PROF_AICPU = 0x0008
    DCL_PROF_MODEL_METRICS = 0x0010
    DCL_PROF_RUNTIME_API = 0x0020
    DCL_PROF_UNVALID = 0x8000


class Nnp4xxSdkProfiler(BaseSdkProfiler, abc.ABC):
    def __init__(
            self,
            sdk_cfg_file="/DEngine/tyhcp/config/sdk.cfg"
    ):
        super(Nnp4xxSdkProfiler, self).__init__(sdk_cfg_file, "nnp400")

        with open(self.sdk_cfg_file) as f:
            cfg = json.load(f)
        # self.profile_dir = cfg["profiler"]["host_output"]
        self.result_dir = ""

        self.ip = cfg["rpc"]["ip_addr"]
        if self.ip == "127.0.0.1":   # TODO 非127.0.0.1的地址也可能是ISS服务
            logger.error("ISS mode not support profile")
            exit(-1)

        self.uuid = str(uuid.uuid1())
        self.profile_dir = os.path.join("/tmp", self.uuid)
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)

    def load(self, model_path):
        self.result_dir = os.path.join(os.path.dirname(model_path), "result")
        if not os.path.exists(self.result_dir):
            logger.error("Not found result_dir -> {}".format(self.result_dir))
            exit(-1)

        if not os.path.isfile(model_path):
            logger.error("netbin_file not file -> {}".format(model_path))
            exit(-1)

        try:
            import python.pydcl as dcl
            logger.info("sdk config path: {}".format(self.sdk_cfg_file))
            dcl.init(self.sdk_cfg_file)
            logger.info("tyhcp init succeed.")

            self.engine = dcl.CNetOperator()

            if not self.engine.profile(Nnp4xxProfileTypeEnum.DCL_PROF_AICORE_METRICS | Nnp4xxProfileTypeEnum.DCL_PROF_DCL_API, self.profile_dir):  # profile
                logger.error("Failed to set profile")
                exit(-1)

            logger.info("load model " + model_path)
            if not self.engine.load(model_path):
                logger.error("Failed to load model")
                exit(-1)
            logger.info("load model success")

            json_graph = self.engine.get_json_graph()
            with open(os.path.join(self.result_dir, "graph.json"), "w") as f:
                f.write(json_graph)
        except Exception as e:
            logger.error("load failed -> {}".format(e))
            exit(-1)

    def run(self, in_datas: dict, to_file=False):
        if isinstance(in_datas, dict):
            in_datas = [in_datas[key] for key in in_datas]  # to list
        t_start = time.time()
        for _ in range(10):
            _ = self.engine.inference(in_datas)
        logger.info("Python interface: {:.3f}ms".format((time.time() - t_start) * 1000 / 10))

    def unload(self):
        if self.engine:
            self.engine.unload()
            logger.info("unload model")
            self.engine = None
            import python.pydcl as dcl
            dcl.finalize()

    def __del__(self):
        self.unload()

    def parse_dcl_api(self):
        model_profile = os.path.join(self.profile_dir, "model_prof.bin")
        if not os.path.exists(model_profile):
            logger.error("Not found profile file -> {}".format(model_profile))
            exit(-1)

        try:
            import python.pydcl as dcl
            profile_json = dcl.parse_dcl_api(model_profile)
            profile = json.loads(profile_json)
            total_time = profile["dclmdlExecute"] / 10**6 / 10
            return total_time

        except Exception as e:
            logger.error("Failed to parse profile -> {}\n{}".format(e, traceback.format_exc()))
            exit(-1)

    def parse(self):
        # ave_latency_ms = self.parse_dcl_api()

        graph_json = os.path.join(self.result_dir, "graph.json")
        if not os.path.exists(graph_json):
            logger.error("Not found {}".format(graph_json))
            exit(-1)

        model_profile = os.path.join(self.profile_dir, "model_prof.bin")
        if not os.path.exists(model_profile):
            logger.error("Not found profile file -> {}".format(model_profile))
            exit(-1)
        try:
            import python.pydcl as dcl
            profile_json = dcl.parse_ai_core(model_profile)
            profile = json.loads(profile_json)
            # dump
            profile_json = json.dumps(profile, indent=2)
            with open(os.path.join(self.result_dir, "profile.json"), "w") as f:
                f.write(profile_json)

            op_names = dict()
            with open(graph_json, "r") as f:
                graph = json.load(f)
            nodes = graph["nodes"]
            for node in nodes:
                if "attrs" not in node:
                    continue
                if "func_name" not in node["attrs"]:
                    continue
                func_name = node["attrs"]["func_name"]
                key = func_name.split("_")[0]
                idx = int(key.replace("f", ""))
                op_names[idx] = func_name

            from prettytable import PrettyTable
            header = ["Id", "OpName", "MAC.", "DDR/R(GB/s)", "DDR/W(GB/s)", "Exec Cycles", "Gap Cycles",
                      "Exec Span/ms", "Gap Span/ms"]
            table = PrettyTable(header)
            num_iter = len(profile)
            total_op_exec_cycles = dict()
            total_op_gap_cycles = dict()
            total_op_ddr_read_bytes = dict()
            total_op_ddr_write_bytes = dict()
            total_op_ddr_read_cycles = dict()
            total_op_ddr_write_cycles = dict()
            total_exec_cycles = 0
            total_gap_cycles = 0
            total_time = 0  # ns
            for p in profile:
                total_exec_cycles += p["total_exec_time"]
                total_gap_cycles += p["total_gap_time"]
                total_time += p["total_time"]
                ops = p["ops"]
                for idx in op_names:
                    op_name = op_names[idx]
                    op = ops[idx]
                    exec_cycles = op["exec_cycles"]
                    gap_cycles = op["gap_cycles"]
                    ddr_read_bytes = op["ddr_read_bytes"]
                    ddr_write_bytes = op["ddr_write_bytes"]
                    ddr_read_cycles = op["ddr_read_cycles"]
                    ddr_write_cycles = op["ddr_write_cycles"]
                    if op_name in total_op_exec_cycles:
                        total_op_exec_cycles[op_name] += exec_cycles
                        total_op_gap_cycles[op_name] += gap_cycles
                        total_op_ddr_read_bytes[op_name] += ddr_read_bytes
                        total_op_ddr_write_bytes[op_name] += ddr_write_bytes
                        total_op_ddr_read_cycles[op_name] += ddr_read_cycles
                        total_op_ddr_write_cycles[op_name] += ddr_write_cycles
                    else:
                        total_op_exec_cycles[op_name] = exec_cycles
                        total_op_gap_cycles[op_name] = gap_cycles
                        total_op_ddr_read_bytes[op_name] = ddr_read_bytes
                        total_op_ddr_write_bytes[op_name] = ddr_write_bytes
                        total_op_ddr_read_cycles[op_name] = ddr_read_cycles
                        total_op_ddr_write_cycles[op_name] = ddr_write_cycles

            for idx in op_names:
                op_name = op_names[idx]
                mean_op_exec_cycles = int(total_op_exec_cycles[op_name] / num_iter)
                mean_op_gap_cycles = int(total_op_gap_cycles[op_name] / num_iter)
                mean_op_ddr_read_cycles = int(total_op_ddr_read_cycles[op_name] / num_iter)
                mean_op_ddr_write_cycles = int(total_op_ddr_write_cycles[op_name] / num_iter)
                mean_op_ddr_read_bytes = int(total_op_ddr_read_bytes[op_name] / num_iter)
                mean_op_ddr_write_bytes = int(total_op_ddr_write_bytes[op_name] / num_iter)
                mac_num = 0
                ddr_read_span = mean_op_ddr_read_cycles * 10**-3 / self.targets[self.target]
                ddr_write_span = mean_op_ddr_write_cycles * 10**-3 / self.targets[self.target]
                ddr_read_bw = mean_op_ddr_read_bytes * 1000 / ddr_read_span / 1024**3  # GB/s
                ddr_write_bw = mean_op_ddr_write_bytes * 1000 / ddr_write_span / 1024**3  # GB/s
                mean_op_exec_span = mean_op_exec_cycles * 10**-3 / self.targets[self.target]  # ms
                mean_op_gap_span = mean_op_gap_cycles * 10**-3 / self.targets[self.target]  # ms
                table.add_row([
                    idx,
                    op_name,
                    mac_num,
                    "{:.3f}".format(ddr_read_bw),
                    "{:.3f}".format(ddr_write_bw),
                    mean_op_exec_cycles,
                    mean_op_gap_cycles,
                    "{:.3f}".format(mean_op_exec_span),
                    "{:.3f}".format(mean_op_gap_span),
                ])

            logger.info("\n{}".format(table))

            mean_total_exec_cycles = int(total_exec_cycles / num_iter)
            mean_total_gap_cycles = int(total_gap_cycles / num_iter)
            mean_total_time = int(total_time / num_iter)  # ns
            logger.info("NumIter: {}, Exec Span: {:.3f}ms, Gap Span: {:.3f}ms".format(
                num_iter,
                mean_total_exec_cycles * 10**-3 / self.targets[self.target],
                mean_total_gap_cycles * 10**-3 / self.targets[self.target],
            ))
            # logger.info("[{}] average cost: {:.3f}ms".format(self.target, ave_latency_ms))

        except Exception as e:
            logger.error("Failed to parse profile -> {}\n{}".format(e, traceback.format_exc()))
            exit(-1)




