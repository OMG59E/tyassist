import os
import yaml
import onnx
import onnx_graphsurgeon as gs
from utils import logger


def is_qnn(graph: gs.Graph):
    for node in graph.nodes:
        op_type = str(node.op)
        if op_type.startswith("Quantize") or op_type.startswith("QLinear") or op_type.startswith("Dequantize"):
            return True
    return False


def is_dynamic_input(input_infos):
    for input_info in input_infos:
        shape = input_info["shape"]
        for dim in shape:
            if not isinstance(dim, int):
                return True
    return False


def gen_default_config(onnx_file):
    graph = gs.import_onnx(onnx.load(onnx_file))
    input_tensors = graph.inputs
    input_infos = list()
    for input_tensor in input_tensors:
        input_info = dict()
        input_info["name"] = input_tensor.name
        input_info["shape"] = input_tensor.shape
        input_info["dtype"] = "float32"
        input_info["layout"] = "None"
        input_info["pixel_format"] = "None"
        input_info["mean"] = None
        input_info["std"] = None
        input_info["resize_type"] = 0
        input_info["padding_value"] = 128
        input_info["padding_mode"] = 1
        input_info["data_path"] = ""
        input_infos.append(input_info)
    
    # check dynamic input
    if is_dynamic_input(input_infos):
        logger.error("Not support dynamic input model")
        exit(-1)

    basename, _ = os.path.splitext(os.path.basename(onnx_file))
    default_cfg = {
        "model": {
            "framework": "onnx" if not is_qnn(graph) else "onnx-qnn",
            "weight": onnx_file,
            "graph": "",
            "save_dir": "outputs/{}".format(basename),
            "inputs": input_infos
        },
        "build": {
            "target": "nnp400",
            "enable_quant": True,
            "enable_build": True,
            "opt_level": 0,
            "multi_thread": None,
            "quant": {
                "data_dir": "",
                "prof_img_num": 1,
                "similarity_img_num": 1,
                "similarity_dataset": None,
                "debug_level": -1,
                "opt_level": 0,
                "calib_method": "kld",
                "custom_preprocess_module": None,
                "custom_preprocess_cls": None,
                "disable_pass": [],
                "num_cube": 3,
                "skip_layer_idxes": [],
                "skip_layer_types": [],
                "skip_layer_names": [],
            },
            "enable_dump": 0,
        }
    }
    config_yml = "{}.yml".format(basename)
    with open(config_yml, "w") as f:
        yaml.dump(default_cfg, f, default_flow_style=False, sort_keys=True)
    logger.info("Save default config to: {}".format(config_yml))
    return default_cfg
