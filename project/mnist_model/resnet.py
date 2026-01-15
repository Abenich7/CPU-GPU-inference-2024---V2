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
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import tensorrt as trt
from cuda import cudart
from pathlib import Path
import time
from typing import Optional, Union, Tuple
import sys

# ======== CONFIG =========




# ================================
#DOWNLOAD dataset from kaggle if not already present

# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

#print("Path to dataset files:", path)

# ======== Prepare the dataset =========
# STANFORD DOGS DATASET
path='/home/benilla/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images'


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





## print example from dataset
#image, label = test_dataset[1]
#print(f"Example image shape: {image.shape}, label: {label}")

#num_classes=len(test_dataset.classes)
#print(f"Number of classes: {num_classes}")

#input_tensor = preprocess(input_image)
#pin_memory=True
#input_tensor = input_tensor.pin_memory()
#input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
#if torch.cuda.is_available():
  #  input_batch = input_batch.to('cuda',non_blocking=True)
  #  model.to('cuda')

# ======= PROFILER =======

#profile 
s=Stream()

def benchmark_with_profiler(engine,
                                input_tensor: torch.cuda.FloatTensor,
                                rank: Optional[int] = 0
) ->torch.cuda.FloatTensor :
    torch._C._profiler._set_cuda_sync_enabled_val(True)
    wait, warmup, active = 1, 1, 2
    num_steps = wait + warmup + active
    rank = 0

    print(f"Starting profiler with schedule: wait={wait}, warmup={warmup}, active={active}")
    print(f"Total steps: {num_steps}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
        ),
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            print(f"Profiler step {step_idx}/{num_steps} (State: )")


            run_inference_trt(engine, input_tensor.cpu().numpy())

            print(f"Step {step_idx} completed, predicted class: ")
            
            if rank is None or rank == 0:
                prof.step()
        
        return output
    
    # Export the trace
    trace_filename = f"trace_run_0.json"
    prof.export_chrome_trace(trace_filename)
    print(f"Trace exported to: {trace_filename}")
    
    # Print summary
    print("\n=== Top CUDA operations ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))



def train_test_split(dataset, test_ratio=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(test_ratio * dataset_size)

    np.random.shuffle(indices)

    test_indices = indices[:split]
    train_indices = indices[split:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    return train_subset, test_subset




# ======= EVALUATE MODEL =======
def evaluate_model(model,test_loader,mode='Test'):
    
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        
        for images, labels in tqdm(test_loader, desc=f'Evaluating {mode}'):
            if torch.cuda.is_available():
                images = images.to('cuda', non_blocking=True)
                labels = labels.to('cuda', non_blocking=True)

            outputs = model(images)

            print(outputs)
            
            _, preds = torch.max(outputs, dim=1)

            output_size=outputs.size()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            test_acc=accuracy_score(all_labels, all_preds)
            

    print(f'\n[{mode}] Overall Accuracy: {test_acc*100:.2f}%')


def train_model(model,train_loader,learning_rate,num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
   
    running_loss = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
     
        for images, labels in tqdm(train_loader, desc='Training'):
            if torch.cuda.is_available():
                images = images.to('cuda', non_blocking=True)
                labels = labels.to('cuda', non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

                
        epoch_loss = running_loss / len(train_loader)
        
        print(f'Loss: {epoch_loss:.4f}')
    

# =========== LOAD MODEL =============

model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')

in_ftr=model.fc.in_features
out_ftr=120
model.fc=torch.nn.Linear(in_ftr,out_ftr,bias=True)

model.load_state_dict(torch.load('./models/resnet18_finetuned.pth'))

model.to('cuda' if torch.cuda.is_available() else 'cpu')

# =========== RUN =============
# Prepare input batch
test_dataset=datasets.ImageFolder(root=path,transform=preprocess)

train_dataset, test_dataset = train_test_split(test_dataset, test_ratio=0.2)

train_loader=DataLoader(train_dataset,shuffle=False,num_workers=2,pin_memory=True,batch_size=16)
test_loader=DataLoader(test_dataset,shuffle=False,num_workers=2,pin_memory=True,batch_size=16)

def finetune_model():
    # Option 1: Freeze all except the final layer (for fast adaptation)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True


    train_model(model,train_loader,num_epochs=8,learning_rate=1e-3)

    evaluate_model(model,test_loader)



    #option 2: Fine-tune the entire model (for better performance)
    for param in model.parameters():
        param.requires_grad = True

    train_model(model,train_loader,num_epochs=10,learning_rate=1e-4)


    evaluate_model(model,test_loader)

    torch.save(model.state_dict(), 'resnet18_finetuned.pth')


    


# =========== EVALUATE INFERENCE =============
#evaluate_model(model,test_loader)

# test single image


input_tensor = preprocess(Image.open(sys.argv[2]))

model.eval()

dataset=datasets.ImageFolder(root=sys.argv[1])

labels=dataset.classes

#measrue latency and throughput

timer_cmd = "model(input_tensor)"
timer = Timer(
    stmt=timer_cmd,
    globals={"model": model, "input_tensor": input_tensor.unsqueeze(0).to('cuda')},
    label="ResNet18 Inference",
    sub_label="Single Image",
    description="Using CUDA Stream",
    num_threads=1,
)   


with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda',non_blocking=False)
    
        print("Measuring inference latency...")
        latency = timer.timeit(100)
    
        print(f"Average inference latency: {latency.mean} s")
        #index=torch.argmax(output, dim=1)
        #print(f"Predicted class: {labels[index]}")


# ======= QUANTIZATION =======
#quantize with tensorRT

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

parser = trt.OnnxParser(network, logger)

#convert model to onnx
dummy_input = torch.randn(1, 3, 224, 224).to('cuda')
torch.onnx.export(model, dummy_input, "resnet18.onnx", export_params=True, opset_version=11)
with open("resnet18.onnx", "rb") as model_file:
    if not parser.parse(model_file.read()):
        print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

print("Building TensorRT engine. This may take a few minutes...")

def build_engine():
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("Failed to build the engine.")
        exit()

    engine_path = "resnet18_trt.engine"
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine saved to {engine_path}")

def check_cuda_error(error):
    if isinstance(error, tuple):
        error = error[0]
    if error != cudart.cudaError_t.cudaSuccess:
        error_name = cudart.cudaGetErrorName(error)[1]
        error_string = cudart.cudaGetErrorString(error)[1]
        raise RuntimeError(f"CUDA Error: {error_name} ({error_string})")



def run_inference_trt(engine: trt.ICudaEngine, input_data: np.ndarray):
    # Create execution context - this stores the device memory allocations
    # and bindings needed for inference
    context = engine.create_execution_context()

    # Initialize lists to store input/output information and GPU memory allocations
    inputs = []
    outputs = []
    allocations = []
    
    # Iterate through all input/output tensors to set up memory and bindings
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        # Check if this tensor is an input or output
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        # Get tensor datatype and shape information
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        
        # Calculate required memory size for this tensor
        size = np.dtype(trt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
            
        # Allocate GPU memory for this tensor
        err, allocation = cudart.cudaMalloc(size)
        check_cuda_error(err)
        
        # Store tensor information in a dictionary for easy access
        binding = {
            "index": i,
            "name": name,
            "dtype": np.dtype(trt.nptype(dtype)),
            "shape": list(shape),
            "allocation": allocation,
            "size": size,
        }
        
        # Keep track of all allocations and sort tensors into inputs/outputs
        allocations.append(allocation)
        if is_input:
            inputs.append(binding)
        else:
            outputs.append(binding)

    # Ensure input data is contiguous in memory for efficient GPU transfer
    input_data = np.ascontiguousarray(input_data)

    # Copy input data from host (CPU) to device (GPU)
    err = cudart.cudaMemcpy(
        inputs[0]["allocation"],
        input_data.ctypes.data,
        inputs[0]["size"],
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    check_cuda_error(err)

    # Set tensor addresses for all tensors
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), allocations[i])

    # Create a CUDA stream for asynchronous execution
    err, stream = cudart.cudaStreamCreate()
    check_cuda_error(err)

    # Run inference using the TensorRT engine
 
    context.execute_async_v3(stream_handle=stream)
    
    #print(f"TensorRT Inference Time: {(end_time - start_time)*1000  :.2f} ms")
    err = cudart.cudaStreamSynchronize(stream)
    check_cuda_error(err)

    # Prepare numpy array for output and copy results from GPU to CPU
    output_shape = outputs[0]["shape"]
    output = np.empty(output_shape, dtype=outputs[0]["dtype"])

    err = cudart.cudaMemcpy(
        output.ctypes.data,
        outputs[0]["allocation"],
        outputs[0]["size"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    check_cuda_error(err)

    # Free all GPU memory allocations
    for allocation in allocations:
        err = cudart.cudaFree(allocation)
        check_cuda_error(err)

    # Destroy the CUDA stream
    err = cudart.cudaStreamDestroy(stream)
    check_cuda_error(err)

    return output

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


# build_engine()

engine_path="resnet18_trt.engine"
                       
# Benchmark TensorRT
with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

trt_times = []
#for _ in range(100):

output = run_inference_trt(engine, input_tensor.cpu().numpy())
print("Output:", output)

# Calculate accuracy
accuracy = output_accuracy(output, labels, topk=(1, 5))
print(f"Top-1 Accuracy: {accuracy[0]:.2f}%")
print(f"Top-5 Accuracy: {accuracy[1]:.2f}%")

#print(f"TensorRT Inference Time: {np.mean(trt_times)*1000:.2f} ms")


# Get top 5 predictions
top5_idx = np.argsort(output[0])[-5:][::-1]
print("Top 5 predictions:")
for idx in top5_idx:
    print(f"{labels[idx]}: {output[0][idx]:.2f}%")


import onnxruntime as ort
def run_inference_onnx(session, input_data: np.ndarray):
    output = session.run(None, {'x': input_data})[0]
    return output

sample_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
# Benchmark ONNX Runtime
session = ort.InferenceSession('./resnet18.onnx')
onnx_times = []
for _ in range(100):
    start_time = time.time()
  #  _ = run_inference_onnx(session, sample_input)
    onnx_times.append(time.time() - start_time)




# benchmark_with_profiler()

