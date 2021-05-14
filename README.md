## EasyRT:TensorRT推理加速工具
### 环境配置
具体参考NVIDIA TensorRT配置指南
### 使用方法
#### 1.1 onnx模型转换为trt engine模型
```python
python onnx2trt.py onnx_path trt_path
```
指定对应输入与输出的文件路径

#### 1.2 TensorRT推理
```python
from infer import Infer
trt_file_path = '' #trt模型路径
output_shapes = [(1,1000)] #，模型输出shape
model = Infer(trt_file_path, output_shapes) #初始模型 
### 自行准备数据读取，
for img,label in dataloader:
    ##img 为对应创建onnx模型时的输入格式，类型为numpy.array
    output = model(img)
    ...
```
**[推理参考demo.ipnb](demo.ipynb)**

