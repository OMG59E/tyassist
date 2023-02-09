# ONNX_YOLOV5

该示例主要用于自定义算子的教学，在原模型中增加后处理算子

## 依赖

- onnx-graphsurgeon
- onnx

## 文件结构

```
onnx_yolov5s
├── build_custom_op.sh  自定义算子编译脚本
├── gen_custom_op_code.sh  自定义算子框架代码生成脚本
├── custom_config.json 自定义算子定义文件
├── yolov5_surgeon.py  编辑onnx模型增加自定义节点
├── config.yaml  模型编译配置文件
├── custom_config.json 编译生成文件
├── DetectionOut_arm.cc 编译生成文件
├── DetectionOut_inferer.py 编译生成文件
└── README.md
```

## 使用说明

### step1 - 定义自定义算子

编辑custom_config.json文件，追加算子定义，具体方法，见"*云天励飞TyTVM工具链用户使用说明(nnp3xx).pdf*"

```json
[
  {
    "op_from": "onnx",
    "op_type": "DetectionOut",
    "op_target": "arm",
    "input_num": 1,
    "output_num": 1,
    "version": 0.2,
    "custom_param": {
      "conf_threshold": "float",
      "iou_threshold": "float",
      "top_k": "int",
      "keep_top_k": "int",
      "num_classes": "int"
    }
  }
]
```

### step2 - 生成框架代码

依据定义生成框架代码，生成.cc和.py文件，.py声明输出shape, .cc用于具体算子实现

```shell
sh gen_custom_op_code.sh
```

### step3 - 实现自定义算子

自行实现

### step4 - 编译算子

```shell
sh build_custom_op.sh
```

### step5 - ONNX模型增加自定义算子

```shell
python3 yolov5_surgeon.py
```

### step6 - 编译模型

依赖tytvm容器环境

```shell
python3 /DEngine/tyassist/tyassist.py build -c config.yml --target nnp300
```

### step7 - 校验模型相似度(可选)

依赖tyhcp容器环境

```shell
python3 /DEngine/tyassist/tyassist.py compare -c config.yml --target nnp300 --backend chip
```

### step8 - demo

```shell
python3 /DEngine/tyassist/tyassist.py demo -c config.yml --target nnp300 --backend chip
```



