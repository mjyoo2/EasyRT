from infer import Infer
from utils.data_prepare import pipeline
import time
begin1 = time.time()
data = pipeline() #初始数据读取
trt_file_path = 'resnet152.trt'
output_shape = [(1,1000)]
model = Infer(trt_file_path, output_shape)#初始化模型
print("加载模型用时：{:.3f}".format(time.time() - begin1))
begin2 = time.time()

for i in range(100):
    img = data.preprocess('test.jpg')
    _ = model(img)
print("模型推理用时：{:.3f}".format(time.time() - begin2))