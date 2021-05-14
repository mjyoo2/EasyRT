import torch
import torchvision

model = torchvision.models.resnet152(pretrained=True)

batch_size = 1  #批处理大小
input_shape = (3, 244, 224)   #输入数据,改成自己的输入shape

# #set the model to inference mode
model.eval()

x = torch.randn(batch_size, *input_shape)	# 生成张量
export_onnx_file = "resnet152.onnx"			# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    input_names=["input"],
                    output_names=["output"])