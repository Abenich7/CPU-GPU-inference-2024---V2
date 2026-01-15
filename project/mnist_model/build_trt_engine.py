
import itertools
import sys

from inflect import engine
from matplotlib.style import context


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

path='/home/benilla/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images'

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

def build_trt_engine(model):

    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)

    #convert model to onnx
    dummy_input = torch.randn(1, 3, 224, 224).to('cuda')
    torch.onnx.export(model, dummy_input, "resnet18_finetuned.onnx", export_params=True, opset_version=11)
    with open("resnet18_finetuned.onnx", "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
            exit()

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    print("Building TensorRT engine. This may take a few minutes...")   


    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("Failed to build the engine.")
        exit()

    engine_path = "trt_engine"
    with open(engine_path, "wb") as f:
            f.write(engine)
    print(f"TensorRT engine saved to {engine_path}")

    return engine

def quantize_engine(trt_engine,calibrator):
    pass

def show_trt_engine_info(trt_engine):
    pass

def run_inference(engine_path, input_folder, batch_size=1):
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
    for batch in tqdm(itertools.batched(test_dataset, batch_size),desc="TensorRT Inference"):
        
        batch_images = [img.numpy() for img, label in batch]
        labels.extend([label for img, label in batch])
        #for f_path, label_id in batch:
            #print(f_path,label_id)


         #   img = cv2.imread(f_path)
    
          #  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            # Apply the exact v2 transform pipeline
            # v2.Compose handles the conversion from NumPy to Tensor via v2.ToImage()
           # transformed_img = inference_transform(img)
            
            # Convert back to NumPy for TensorRT input buffer
            #batch_images.append(transformed_img.numpy())
           
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
        cudart.cudaStreamSynchronize(stream)
        all_results.append(h_output)

    # 4. Cleanup
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)
    
    return np.vstack(all_results), labels

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