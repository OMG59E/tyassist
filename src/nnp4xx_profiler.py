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
import glob
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
            sdk_cfg_file="/DEngine/tyhcp/client/config/sdk.cfg",
            device_id=0,  # 指定目标设备，目前只支持0
            node_id=0,  # 指定某设备的目标Die
    ):
        super(Nnp4xxSdkProfiler, self).__init__(sdk_cfg_file, "nnp400")

        with open(self.sdk_cfg_file) as f:
            cfg = json.load(f)
            
        assert device_id == 0
        connect_type = cfg["node_cfg"]["connect"]
        devices = cfg["node_cfg"]["devices"]   # 设备数量
        if connect_type in ["socket", "usb"]:
            if node_id != 0:
                logger.error("connect_type is socket or usb, only support die0")
                exit(-1)
        elif connect_type == "pcie":
            nodes = devices[0]["nodes"]  # 设备0 Die数量
            node_ids = [node["node_id"] for node in nodes]
            if node_id not in node_ids:
                logger.error("node_id must be in {}".format(node_ids))
                exit(-1)
        else:
            logger.error("Not support connect type: {}".format(connect_type))
            exit(-1)
        self.device_id = device_id
        self.node_id = node_id  
        
        # self.profile_dir = cfg["profiler"]["host_output"]
        self.result_dir = ""

        self.ip = devices[0]["nodes"][0]["clients"]["dcl"]["targetId"]
        if self.ip == "127.0.0.1":   # 非127.0.0.1的地址也可能是ISS服务
            logger.error("ISS mode not support profile")
            exit(-1)

        # profile结果保存目录
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
            # 指定目标Die
            dcl.set_device(self.node_id)
            
            self.engine = dcl.CNetOperator()

            # profile
            if not self.engine.profile(
                Nnp4xxProfileTypeEnum.DCL_PROF_AICORE_METRICS | Nnp4xxProfileTypeEnum.DCL_PROF_AICPU, self.profile_dir): 
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

    def find_model_prof_bin(self):
        filenames = glob.glob(os.path.join(self.profile_dir, "model_prof_*.bin"))
        assert len(filenames) > 0;
        model_profile = filenames[0]
        return model_profile

    def parse_dcl_api(self):
        model_profile = self.find_model_prof_bin()
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

        model_profile = self.find_model_prof_bin()
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

            with open(graph_json, "r") as f:
                graph = json.load(f)
            nodes = graph["nodes"]

            op_names = list()
            op_macs = list()
            for node in nodes:
                if node["op"] != "tvm_op":
                    continue
                func_name = node["attrs"]["func_name"]
                nnp_macs = node["attrs"].get("nnp_macs", "0")
                op_names.append(func_name)
                op_macs.append(nnp_macs)

            from prettytable import PrettyTable
            header = ["Id", "OpName", "Device", "MAC.", "DDR/R(GB/s)", "DDR/W(GB/s)", "Exec Cycles", "Gap Cycles", "Exec Span/ms", "Gap Span/ms"]
            table = PrettyTable(header)
            num_iter = len(profile)
            total_op_exec_cycles = list()
            total_op_gap_cycles = list()
            total_op_ddr_read_bytes = list()
            total_op_ddr_write_bytes = list()
            total_op_ddr_read_cycles = list()
            total_op_ddr_write_cycles = list()
            total_op_gap_time = list()
            op_types = list()
            total_exec_cycles = 0
            total_gap_cycles = 0
            total_time = 0  # ns
            cpu_ops = dict()
            for p in profile:
                total_exec_cycles += p["total_exec_cycles"]
                total_gap_cycles += p["total_gap_cycles"]
                total_time += p["total_time"]
                ops = p["ops"]
                assert len(ops) == len(op_names)
                for idx, op_name in enumerate(op_names):
                    op = ops[idx]
                    op_type = op["type"]
                    assert op_type in [0, 1]
                    exec_cycles = op["exec_cycles"]
                    gap_cycles = op["gap_cycles"]
                    gap_time = op["gap_time"]
                    ddr_read_bytes = op["ddr_read_bytes"]
                    ddr_write_bytes = op["ddr_write_bytes"]
                    ddr_read_cycles = op["ddr_read_cycles"]
                    ddr_write_cycles = op["ddr_write_cycles"]

                    if len(total_op_exec_cycles) > idx:
                        total_op_exec_cycles[idx] += exec_cycles
                        total_op_gap_cycles[idx] += gap_cycles
                        total_op_ddr_read_bytes[idx] += ddr_read_bytes
                        total_op_ddr_write_bytes[idx] += ddr_write_bytes
                        total_op_ddr_read_cycles[idx] += ddr_read_cycles
                        total_op_ddr_write_cycles[idx] += ddr_write_cycles
                        total_op_gap_time[idx] += gap_time
                    else:
                        op_types.append(op_type)
                        total_op_exec_cycles.append(exec_cycles)
                        total_op_gap_cycles.append(gap_cycles)
                        total_op_ddr_read_bytes.append(ddr_read_bytes)
                        total_op_ddr_write_bytes.append(ddr_write_bytes)
                        total_op_ddr_read_cycles.append(ddr_read_cycles)
                        total_op_ddr_write_cycles.append(ddr_write_cycles)
                        total_op_gap_time.append(gap_time)

            mean_total_time = 0
            for idx, op_name in enumerate(op_names):
                mean_op_exec_cycles = int(total_op_exec_cycles[idx] / num_iter)
                mean_op_gap_cycles = int(total_op_gap_cycles[idx] / num_iter)
                mean_op_ddr_read_cycles = int(total_op_ddr_read_cycles[idx] / num_iter)
                mean_op_ddr_write_cycles = int(total_op_ddr_write_cycles[idx] / num_iter)
                mean_op_ddr_read_bytes = int(total_op_ddr_read_bytes[idx] / num_iter)
                mean_op_ddr_write_bytes = int(total_op_ddr_write_bytes[idx] / num_iter)
                mac_num = op_macs[idx]
                ddr_read_span = mean_op_ddr_read_cycles * 10**-3 / self.targets[self.target]
                ddr_write_span = mean_op_ddr_write_cycles * 10**-3 / self.targets[self.target]
                ddr_read_bw = 0 if ddr_read_span == 0 else (mean_op_ddr_read_bytes * 1000 / ddr_read_span / 1024**3)  # GB/s
                ddr_write_bw = 0 if ddr_read_span == 0 else (mean_op_ddr_write_bytes * 1000 / ddr_write_span / 1024**3)  # GB/s
                mean_op_exec_span = mean_op_exec_cycles * 10**-3 / self.targets[self.target]  # ms
                mean_op_gap_span = (total_op_gap_time[idx] / num_iter) if op_types[idx] == 1 else (mean_op_gap_cycles * 10**-3 / self.targets[self.target])  # ms
                mean_total_time += mean_op_gap_span
                table.add_row([
                    idx,
                    op_name,
                    "NPU" if op_types[idx] == 0 else "CPU",
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
            # mean_total_time = int(total_time / num_iter)  # ns
            logger.info("NumIter: {}, SW Exec Span: {:.3f}ms, SW Gap Span: {:.3f}ms, Total Span: {:.3f}ms".format(
                num_iter,
                mean_total_exec_cycles * 10**-3 / self.targets[self.target],
                mean_total_gap_cycles * 10**-3 / self.targets[self.target],
                mean_total_time,
            ))
            # logger.info("[{}] average cost: {:.3f}ms".format(self.target, ave_latency_ms))

        except Exception as e:
            logger.error("Failed to parse profile -> {}\n{}".format(e, traceback.format_exc()))
            exit(-1)




