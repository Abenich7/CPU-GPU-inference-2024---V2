import time
import gc 
from pathlib import Path
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.benchmark import Timer
from torchvision import datasets, transforms
from datetime import datetime

import contextlib
from torch.cuda import Stream

from .mnist_model import NeuralNetwork
from .mnist_train import train_epoch, evaluate

import matplotlib.pyplot as plt
import numpy as np

import tempfile

import os


#import wandb
#run = wandb.init(project="my-model-training-project")
#run.config = {"epochs": 1337, "learning_rate": 3e-4}
#run.log({"metric": 42})
#my_model_artifact = run.log_artifact("./my_model.pt", type="model")
# ------------------------------- paths -------------------------------
ROOT_DIR = Path(__file__).resolve().parent          # project root= mnist_model
DATA_DIR = ROOT_DIR / "mnist"                      # contains mnist datasets
OUT_DIR  = ROOT_DIR / "results"                    # where model files are saved
OUT_DIR.mkdir(exist_ok=True)


# results_path
date_now=datetime.now()

results_summary_path=OUT_DIR/ f"inference_log_{date_now:%H%M%S}.txt"

def fprint(*args, **kwargs):
    with open(results_summary_path, "a", encoding="utf-8") as f:
        print(*args, file=f, **kwargs)

#trace path






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

def inference_fn_0(model, test_ds):
    model.eval()
    stream = torch.cuda.Stream()

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        pin_memory=True
    )

    with torch.no_grad():
        for X, y in test_loader:
            with torch.cuda.stream(stream):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                output = model(X)

        # wait for stream to finish before exiting
        torch.cuda.synchronize()

    return




def inference_fn_1(
    Model,
    Dataset,

) -> int:
    
    """Run inference on the test dataset and return the predicted class for the first image."""
  
    s= Stream()

    
    test_loader  = DataLoader(Dataset,pin_memory=True,batch_size=16)

  
    with torch.no_grad():
     
      
        for X, y in test_loader:
            
          
            with torch.cuda.stream(s):
           
                X,y= X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    

            # convolutions of the model are done on the GPU
            
            #wait for previous batch to finish being sent
            output = Model(X)

            #send next batch


                    
            #softmax=mathematical function that converts logits to probabilities
      #      probabilities = torch.softmax(output, dim=1)
          
            #argmax returns the index of the maximum value in a tensor along a specified dimension
       
       #     predicted_class = torch.argmax(probabilities, dim=1).item()
            
       
def inference_fn_2(model, test_ds):
    model.eval()
    stream = torch.cuda.Stream()

    loader = DataLoader(test_ds, batch_size=16, pin_memory=True)
    iterator = iter(loader)

    X, y = next(iterator)
    X = X.to(device, non_blocking=True)

    with torch.no_grad():
        for next_batch in iterator:
            with torch.cuda.stream(stream):
                next_X, next_y = next_batch
                next_X = next_X.to(device, non_blocking=True)

            # compute on current batch (default stream)
            output = model(X)

            # wait until next batch is ready
            torch.cuda.current_stream().wait_stream(stream)
            X = next_X

        # last batch
        output = model(X)
        torch.cuda.synchronize()   
      

def timer(cmd):
    median = (
        Timer(cmd, globals=globals())
        .adaptive_autorange(min_run_time=1.0, max_run_time=20.0)
        .median
        * 10
    )
    fprint(f"{cmd}: {median: 4.4f} ms")
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
    *args,
    func,
    **kwargs
    
) -> None:
    torch._C._profiler._set_cuda_sync_enabled_val(True)
    wait, warmup, active = 1, 1, 2
    num_steps = wait + warmup + active
    rank = 0
    fprint(kwargs)

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
            
            # Run inference function
            func(*args)

            print(f"Step {step_idx} completed")
            
            if rank is None or rank == 0:
                prof.step()
    
    # Export the trace
    
    trace_path = OUT_DIR / f"trace_{date_now:%H%M%S}.json"

    prof.export_chrome_trace(str(trace_path))
    print("Trace saved to:", trace_path)

    # fprint summary
    fprint(f"\n=== Top Cuda operations ===")
    fprint(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("Results saved to:",results_summary_path)


run={'model':'NN',
     'dataset':'mnist',
     'batch_size':'16',
     'quantization':'none',
     'sync/async':'async',
     'streams':'2',
     'buffers':'1'
     }

#benchmark_with_profiler(model,test_ds,func=inference_fn_2,**run)




#if inference_times:
 #   avg_load_time = sum(load_times) / len(load_times)
  #  avg_inference_time = sum(inference_times) / len(inference_times)
   # fprint(f"Average load time: {avg_load_time:.6f} seconds")
    #fprint(f"Average inference time: {avg_inference_time:.6f} seconds")
    #with open(inference_time_file, "a") as f:
     #   f.write(f"\nAverage load time: {avg_load_time:.6f} seconds\n")
      #  f.write(f"Average inference time: {avg_inference_time:.6f} seconds\n")
#else:
 #   fprint("No inference times recorded.")