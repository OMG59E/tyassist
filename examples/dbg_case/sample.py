import numpy as np
import python.pydcl as dcl
import python.dclOpDbg as dclOpDbg
import python.dclMdl as dclMdl
import python.dclNnpRtt as dclNnpRtt

cfg = "/DEngine/tyhcp/config/sdk.cfg"
model_file = "yolov5s_a55.so"

breakpoint_lists = ["f8_sigmoid", "f341_concatenate"]

dcl.init(cfg)  # 初始化sdk
dclOpDbg.Enable()  # 使能Debug模式
dclNnpRtt.Enable(1024 * 1024)  # rtt信息只读一次，需要尽量设置大点，1024对齐

model_id = dclMdl.Load(model_file)  # 加载模型
print("ModelId", model_id)
dclOpDbg.SetModelId(model_id)  # 绑定模型id

for opname in breakpoint_lists:
    dclOpDbg.SetBpByName(opname)  # 通过opname设置断点, 由用户查询相关描述文件获得opname

# 设置输入并启动
input_data = (np.random.randn(1, 3, 640, 640) * 255).astype(np.uint8)
dclMdl.SetInput(model_id, 0, input_data)
dclMdl.Execute(model_id)

state = dclOpDbg.GetState()  # 检查状态
if not state or state != "INIT":
    print("Failed to get state")
    exit(-1)

for opname in breakpoint_lists:
    breakpoint_idx = dclOpDbg.Continue()
    rtt_info = dclNnpRtt.Get()  # 获取rtt信息
    rtt_info.tofile("{}_rtt_info.bin".format(opname))
    state = dclOpDbg.GetState()
    if not state:
        print("Failed to get state")
    breakpoint_name = dclOpDbg.GetCurOpName()
    breakpoint_info = dclOpDbg.InfoBP()
    outputs = dclOpDbg.GetCurOpIOData(data_copy=True)
    for idx, output in enumerate(outputs):
        output.tofile("{}_{}.bin".format(opname, idx))
        output.tofile("{}_{}.txt".format(opname, idx), sep="\n")
        print(output.shape, output.dtype)
        print(output.flatten())

dclNnpRtt.Disable()
dclMdl.Unload(model_id)  # 卸载模型
dcl.finalize()  # 清理sdk
