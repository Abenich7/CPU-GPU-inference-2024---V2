# goal: Implement accuracy calculation for model evaluation

import torch
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from pathlib import Path
from kagglehub import kagglehub
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.transforms import v2
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import tensorrt as trt
from .build_trt_engine import build_trt_engine, run_inference, output_accuracy
import matplotlib
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import csv 
import time
import contextlib

from torch.cuda import Stream


#define device
device='cuda:1'

#define project root for all model files
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))





model=resnet18(
     weights="IMAGENET1K_FBGEMM_V1",  # pre-quantized weights
    quantize=True
)



in_ftr=model.fc.in_features
out_ftr=120
model.fc=torch.nn.Linear(in_ftr,out_ftr,bias=True)


model.fc.load_state_dict(torch.load("fc_head_resnet18_stanford_dogs.pth"))




def create_onnx(model,quantization: str,batch_size : int):
    dummy_input = torch.randn(batch_size, 3, 224, 224)
   

    onnx_path = os.path.join(PROJECT_ROOT,f'resnet18_{quantization}_{batch_size}.onnx')


    torch.onnx.export(model, dummy_input, onnx_path, export_params=True)

create_onnx(model,'int8',1)
create_onnx(model,'int8',16)
create_onnx(model,'int8',64)
