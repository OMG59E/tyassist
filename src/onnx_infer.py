import time
from abc import ABC
from .base_infer import BaseInfer


class OnnxInfer(BaseInfer, ABC):
    def __init__(self):
        super().__init__()
        self.input_names = None
        self.backend = "onnx"

    def load_json(self, model_path):
        self.load(model_path)

    def load(self, model_path):
        import onnxruntime
        self.engine = onnxruntime.InferenceSession(model_path)

    def set_input_names(self, input_names: list):
        self.input_names = input_names
        
    def run(self, in_datas: dict, to_file=False):
        if not isinstance(in_datas, dict):
            _in_datas = dict()
            for idx, input_name in enumerate(self.input_names):
                _in_datas[input_name] = in_datas[idx]
            in_datas = _in_datas
        self.total += 1
        t_start = time.time()
        outputs = self.engine.run(None, in_datas)
        self.time_span += (time.time() - t_start) * 1000
        return outputs

    def unload(self):
        pass
