import torchvision
import torch
import time
from utils.data_prepare import pipeline
begin1 = time.time()
data = pipeline() #初始数据读取
model = torchvision.models.resnet152(pretrained=False).cuda()
model = model.cuda()
print("加载模型用时：{:.3f}".format(time.time() - begin1))
begin2 = time.time()
for i in range(100):
    img = data.preprocess('test.jpg')
    img = torch.tensor(img, dtype=torch.float32).view(1,3,224,224).cuda()
    _ = model(img)
print("模型推理用时：{:.3f}".format(time.time() - begin2))


