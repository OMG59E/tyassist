
import numpy as np
import onnx
import onnx_graphsurgeon as gs

sourceOnnx = "../../models/onnx/onnx_yolov5/best.onnx"
destinationOnnx = "yolov5s_surgeon.onnx"

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

detections = gs.Variable("detections", np.float32, [1, 200, 6])
detection_out = gs.Node(
    op="DetectionOut",
    name="detection_out",
    attrs={"keep_top_k": 200, "top_k": 400, "num_classes": 80, "iou_threshold": 0.45, "conf_threshold": 0.25},
    inputs=graph.nodes[-1].outputs,
    outputs=[detections]
)

graph.nodes.append(detection_out)
graph.outputs[0] = detections

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), destinationOnnx)
