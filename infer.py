from __future__ import print_function
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import utils.common as common
import os
       
class Infer:
    def __init__(self, trt_file_path, output_shapes,*inputs):
        self.TRT_LOGGER = trt.Logger()
        self.engine_file_path = trt_file_path 
        self.output_shapes = output_shapes
        self.engine = self.get_engine
        self.context = self.engine.create_execution_context() 
    
    @property
    def get_engine(self):
        assert os.path.exists(self.engine_file_path)
        print("Reading engine from file {}".format(self.engine_file_path))
        with open(self.engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    def __call__(self, tensor):
        trt_outputs = []
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        inputs[0].host = tensor
        trt_outputs = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        return trt_outputs
