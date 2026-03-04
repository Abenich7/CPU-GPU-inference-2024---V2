from torch.utils.data import DataLoader
import itertools
import sys

from inflect import engine
from matplotlib.style import context

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tensorrt_repo.trt_src.samples.python.dds_faster_rcnn.build_engine import EngineBuilder

import torch
from torch.cuda import Stream
from torch.utils.benchmark import Timer
import os
from PIL import Image
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import kagglehub
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import tensorrt as trt
from cuda import cudart
from pathlib import Path
import time
from typing import Optional, Union, Tuple
import sys
import glob
import cv2
import torch
from torchvision.transforms import v2
from .min_max_calibrator import EngineCalibrator
from .image_batcher import ImageBatcher

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))




# Use original transform pipeline
inference_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])


device='cuda'

# Download latest version
path ='/home/workspace/benilla/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images'
dataset=datasets.ImageFolder(root=path, transform=inference_transform)

# 2. Split the INDICES first
# Stratify ensures all 120 classes are represented in both sets
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.2, 
    stratify=dataset.targets, 
    random_state=42
)

test_dataset = Subset(dataset, test_idx)


def check_cuda_error(error):
    if isinstance(error, tuple):
        error = error[0]
    if error != cudart.cudaError_t.cudaSuccess:
        error_name = cudart.cudaGetErrorName(error)[1]
        error_string = cudart.cudaGetErrorString(error)[1]
        raise RuntimeError(f"CUDA Error: {error_name} ({error_string})")



class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
       # self.config = self.builder.create_builder_config()
        #self.config.set_memory_pool_limit(
         #   trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        #)

        self.batch_size = None
        self.network = None
        self.parser = None


    def create_network(self, onnx_path):
            """
            Parse the ONNX graph and create the corresponding TensorRT network definition.
            :param onnx_path: The path to the ONNX graph to load.
            """
            self.network = self.builder.create_network(0)
            self.parser = trt.OnnxParser(self.network, self.trt_logger)

            onnx_path = os.path.realpath(onnx_path)
            with open(onnx_path, "rb") as f:
                if not self.parser.parse(f.read()):
                    log.error("Failed to load ONNX file: {}".format(onnx_path))
                    for error in range(self.parser.num_errors):
                        log.error(self.parser.get_error(error))
                    sys.exit(1)

            inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
            outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

            log.info("Network Description")
            for input in inputs:
                self.batch_size = input.shape[0]
                log.info(
                    "Input '{}' with shape {} and dtype {}".format(
                        input.name, input.shape, input.dtype
                    )
                )
            for output in outputs:
                log.info(
                    "Output '{}' with shape {} and dtype {}".format(
                        output.name, output.shape, output.dtype
                    )
                )
            assert self.batch_size == 8

    def build_trt_engine(self,
                        engine_path,
                        precision: str,
                        config_file,
                        calib_input= None,
                        calib_cache=None,
                        calib_num_images=5000,
                        calib_batch_size=8):

        
    #    logger = trt.Logger(trt.Logger.VERBOSE)
    #    builder = trt.Builder(logger)
     #   network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

      #  parser = trt.OnnxParser(network, logger)
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

  

        with self.builder.create_builder_config() as config:
            
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB


            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)


            if precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)


                try:
                    config.int8_calibrator = EngineCalibrator(None)
                except Exception as e:
                        print(f"an error occurred: {e}")

                if calib_cache is None or not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    try:
                        config.int8_calibrator.set_image_batcher(
                                ImageBatcher(
                                
                                    calib_input,
                                    calib_shape,
                                    calib_dtype,
                                    max_num_images=calib_num_images,
                                    exact_batches=False,
                                    config_file=config_file,
                                )
                            )
                    except Exception as e:
                        print(f"an error occurred: {e}")

                
            print("Building TensorRT engine. This may take a few minutes...")   

        try:
            engine = self.builder.build_serialized_network(self.network, config)

        except Exception as e:
                        print(f"an error occurred: {e}")
        if engine is None:
            print("Failed to build the engine.")
            exit()

        
        with open(engine_path, "wb") as f:
                f.write(engine)
        print(f"TensorRT engine saved to {engine_path}")

        return engine

def quantize_engine(trt_engine,calibrator):


    pass

def show_trt_engine_info(trt_engine):
    pass

def run_inference(engine_path):

    #get static batch size from models.csv (because for trt there is a single batch per engine)
    #read models.csv
    import csv
    with open('models.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        models_data = [row for row in csv_reader]
    #find the row with the engine_path
  
    for row in models_data:
        if row['model_path'] == engine_path:
            batch_size = int(row['batch_size'])
            break
    #look for the row# of the path in the csv.
    #get the batch_size for that row
    


   # load (deserialize) the engine
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    
    # 1. Manually identify input and output names (or use engine.get_tensor_name(0))
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    num_classes = 120

    # 2. Pre-allocate MAX buffers to reuse for every batch
    # This prevents memory spikes and allocation overhead
    max_input_nbytes = batch_size * 3 * 224 * 224 * np.dtype(np.float32).itemsize
    max_output_nbytes = batch_size * num_classes * np.dtype(np.float32).itemsize
    
    _, d_input = cudart.cudaMalloc(max_input_nbytes)
    _, d_output = cudart.cudaMalloc(max_output_nbytes)
    _, stream = cudart.cudaStreamCreate()

    #image_iterator = glob.iglob(f"{input_folder}/**/*.jpg", recursive=True)
    
    #image_samples=test_dataset.samples

    

    all_results = []
    labels=[]
    # 3. Main Loop: Move through images in chunks of batch size
    with tqdm(itertools.batched(test_dataset, batch_size),desc="TensorRT Inference") as t:
        for batch in t:
            
            batch_images = [img.numpy() for img, label in batch]
            labels.extend([label for img, label in batch])
           
            
            if not batch_images: continue
            
            # Determine actual size for the current chunk (important for the last batch)
            curr_batch_len = len(batch_images)
            input_data = np.ascontiguousarray(np.stack(batch_images))

            # A. Set input shape for THIS specific batch (Fixes the ValueError)
        # context.set_input_shape(input_name, (curr_batch_len, 3, 224, 224))
            
            # B. Set the memory addresses
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))

            # C. Copy data to GPU
            cudart.cudaMemcpyAsync(d_input, input_data.ctypes.data, input_data.nbytes, 
                                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

            # D. Execute Inference
            context.execute_async_v3(stream_handle=stream)

            # E. Copy results back to CPU
            h_output = np.empty((curr_batch_len, num_classes), dtype=np.float32)
            cudart.cudaMemcpyAsync(h_output.ctypes.data, d_output, h_output.nbytes, 
                                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            
            # F. Synchronize so we can use h_output immediately
            all_results.append(h_output)

    
    cudart.cudaStreamSynchronize(stream)
    
    elapsed_time=t.format_dict['elapsed']
    iteration_rate=t.format_dict['rate']

    # 4. Cleanup
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)
    
    return np.vstack(all_results), labels,elapsed_time,iteration_rate

def output_accuracy(output, labels, topk=(1, 5)):
    maxk = max(topk)
    batch_size = output.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def analyze_inference_results(all_preds, all_labels):
    pass



if __name__ == "__main__":

 #   full_dataset = datasets.ImageFolder(root=path, transform=inference_transform)

    
  #  targets = np.array(full_dataset.targets)

   # calib_idx, _ = train_test_split(
   # np.arange(len(targets)),
   #train_size=500,            # 300–1000 is usually enough
   # stratify=targets,
    #random_state=42 
    #)

   # calib_dataset = Subset(full_dataset, calib_idx)

    #calib_dataloader=DataLoader(calib_dataset)

    input_dir="/home/workspace/benilla/project/stanford_dogs_calib_subset_flat"



    config_file="/home/workspace/benilla/project/stanford_dogs.cfg"


    builder=EngineBuilder()


    batch_size=8
    #convert model to onnx
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    onnx_path='resnet18_dogs_80%_acc.onnx'
    with open(os.path.join(PROJECT_ROOT,'resnet18_dogs_80%_acc.pth'),'rb') as f:
            
          model=torch.hub.load('pytorch/vision', 'resnet18', weights='IMAGENET1K_V1')

          in_ftr=model.fc.in_features
          out_ftr=120
          model.fc=torch.nn.Linear(in_ftr,out_ftr,bias=True)
            
          model.to(device)
          model.load_state_dict(torch.load(f))

            
          torch.onnx.export(model, dummy_input,onnx_path, export_params=True, opset_version=11)
        

    builder.create_network(onnx_path)



    builder.build_trt_engine(engine_path='trt_engine_dogs_80%_acc_int8',
                             precision='int8',
                             
                             config_file=config_file,

                             calib_input=input_dir
                             )