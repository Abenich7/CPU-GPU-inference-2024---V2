import time
import gc 
from pathlib import Path
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.benchmark import Timer
from torchvision import datasets, transforms

import contextlib
from torch.cuda import Stream

from mnist_model import NeuralNetwork
from mnist_train import train_epoch, evaluate

import matplotlib.pyplot as plt
import numpy as np




#import wandb
#run = wandb.init(project="my-model-training-project")
#run.config = {"epochs": 1337, "learning_rate": 3e-4}
#run.log({"metric": 42})
#my_model_artifact = run.log_artifact("./my_model.pt", type="model")
# ------------------------------- paths -------------------------------
ROOT_DIR = Path(__file__).resolve().parent          # project root
DATA_DIR = ROOT_DIR / "mnist"                      # contains raw MNIST files
OUT_DIR  = ROOT_DIR / "results"                    # where artefacts go
OUT_DIR.mkdir(exist_ok=True)





# ------------------------------- data -----------------------------------


#test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=False)

#first_image= test_ds[0][0]  # Get the first image and add batch dimension
#first_image = np.array(first_image, dtype='float')
#pixels = first_image.reshape((28, 28))
#save the image to view it (cant view it in matplotlib popup window in container)
#plt.imsave(OUT_DIR / 'first_image.png', pixels, cmap='gray')

#print first_label
#first_label = test_ds[0][1]  # Get the label of the first image
#print(f"First image label: {first_label}")





#inference_times = []
#load_times = []
#inference_time_file = OUT_DIR / "inference_times.txt"




def inference(
    Model,
    Dataset,
    num_batches: int = 5
) -> int:
    """Run inference on the test dataset and return the predicted class for the first image."""

    s= Stream()

    with torch.cuda.stream(s):
        #analyze where data goes in data loader function 
        test_loader  = DataLoader(test_ds,pin_memory=False)

        pin_mem_event=torch.cuda.current_stream().record_event()

    with torch.no_grad():
        counter=0
        predicted_class=0
    
        for X, y in test_loader:
            counter+=1
            #load_start= time.time()
            

            # command to send tensors to gpu
            X,y= X.to(device, non_blocking=True), y.to(device, non_blocking=True)
          #  if counter==1:
           #     X_h2d_event=torch.cuda.current_stream().record_event()

            #    pin_mem_event.synchronize()
                
             #   X_h2d_event.synchronize()
              #  load_time= pin_mem_event.elapsed_time(X_h2d_event)/1000.0 #convert to seconds
          
               

            #load_end= time.time()
            # convolutions of the model are done on the GPU
            output = model(X)
                    
            #softmax=mathematical function that converts logits to probabilities
            probabilities = torch.softmax(output, dim=1)
          
            #argmax returns the index of the maximum value in a tensor along a specified dimension
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
          #  if counter >= num_batches:
           #     break
            #end_time = time.time()
            
            #inference_time = end_time - start_time 
            #print(f"Predicted class: {predicted_class}, Load time: {load_time:.6f}s, Inference time: {inference_time:.6f}s")

            #load_times.append(load_time)
            #inference_times.append(inference_time)
            #f.write(f"{load_time:.6f},{inference_time:.6f}\n")

             #convert to onnx format
        return predicted_class
       # torch.onnx.export(model, X, OUT_DIR / "mnist_model.onnx", export_params=True, opset_version=11);  
            

def timer(cmd):
    median = (
        Timer(cmd, globals=globals())
        .adaptive_autorange(min_run_time=1.0, max_run_time=20.0)
        .median
        * 10
    )
    print(f"{cmd}: {median: 4.4f} ms")
    return median



device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(OUT_DIR/'mnist_model.pth')) # Load the saved model weights
model.eval() # Set the model to evaluation mode

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_ds=datasets.MNIST(root=DATA_DIR, train=False, download=False,transform=TRANSFORM)






def benchmark_with_profiler(
    model,
    test_ds,
    num_batches_per_step: int = 1
) -> None:
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
            
            # Run inference for this step
            predicted_class = inference(
                Model=model,
                Dataset=test_ds,
                num_batches=num_batches_per_step
            )
            print(f"Step {step_idx} completed, predicted class: {predicted_class}")
            
            if rank is None or rank == 0:
                prof.step()
    
    # Export the trace
    trace_filename = f"trace_run_0.json"
    prof.export_chrome_trace(trace_filename)
    print(f"Trace exported to: {trace_filename}")
    
    # Print summary
    print("\n=== Top CUDA operations ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))




benchmark_with_profiler(model,test_ds)

#if inference_times:
 #   avg_load_time = sum(load_times) / len(load_times)
  #  avg_inference_time = sum(inference_times) / len(inference_times)
   # print(f"Average load time: {avg_load_time:.6f} seconds")
    #print(f"Average inference time: {avg_inference_time:.6f} seconds")
    #with open(inference_time_file, "a") as f:
     #   f.write(f"\nAverage load time: {avg_load_time:.6f} seconds\n")
      #  f.write(f"Average inference time: {avg_inference_time:.6f} seconds\n")
#else:
 #   print("No inference times recorded.")