from __future__ import print_function
import utils.common as common
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
parser = argparse.ArgumentParser(description='onnx2trt')
parser.add_argument('onnx_path')
parser.add_argument('trt_path')
args = parser.parse_args()

###create trt engine from parsering onnx model
TRT_LOGGER = trt.Logger() #init logger
onnx_file_path = args.onnx_path
engine_file_path = args.trt_path

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
    config.max_workspace_size = 1 << 28 # 256MiB
    builder.max_batch_size = 1
     # Parse model file
    print('Loading ONNX file from path {}...'.format(onnx_file_path))
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    network.get_input(0).shape = [1, 3, 224, 224]
    print('Completed parsing of ONNX file')
    print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
    engine = builder.build_engine(network, config)
    print("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())                                                                                                                                                                                                                                               