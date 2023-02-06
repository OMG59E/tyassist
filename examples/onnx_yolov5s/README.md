# YoloV5

## Model Info
https://github.com/ultralytics/yolov5

- input_size: 1,3,640,640
- pixel_format: RGB
- mean: [0, 0, 0]
- std: [255.0, 255.0, 255.0]
- dataset_val: coco_val2017

## Modify
1. 由于sigmoid在NPU上性能较差，将原激活函数SiLU替换为线性激活函数Hardswish，修改models/common.py中代码

```python
class Conv(nn.Module):
   # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
   # act = nn.SiLU()  # default activation
   act = nn.Hardswish()
```