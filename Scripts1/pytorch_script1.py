import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import time
import torchvision.models as models
import sys

MODELS_PATH=r'C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Models'

models_dir=os.path.join(MODELS_PATH)
sys.path.insert(0,models_dir)


import pytorch_model

TRAINING_PATH=r'C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Training'

training_dir=os.path.join(TRAINING_PATH)
sys.path.insert(0,training_dir)

import pytorch_train

INFERENCE_PATH=r'C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Inference'

inference_dir=os.path.join(INFERENCE_PATH)
sys.path.insert(0,inference_dir)

import pytorch_test



import matplotlib.pyplot as plt


# INPUT PATHS
train_dataset_path=r'C:\Users\Abeni07\data\concentric_circles_tensor_train_dataset.pt'
test_dataset_path=r'C:\Users\Abeni07\data\concentric_circles_tensor_test_dataset.pt'


# בדיקת האם הנתונים כבר קיימים
if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
    # טעינת הנתונים מהקבצים
    train_dataset = torch.load(train_dataset_path)
    test_dataset = torch.load(test_dataset_path)
    print("Datasets loaded from files.")
else:
    print("no data available")


# Get cpu, gpu or mps device for training.
device = "cpu"
print(f"Using {device} device")


batch_size = 64

# Load datasets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)




for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


model = pytorch_model.NeuralNetwork().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


epochs = 21


total_time=0


checkpoint_path = 'model_weights.pth'


# בדיקת קיום Checkpoint וטענתו אם קיים
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    start_epoch = 0
    print("No checkpoint found, starting from scratch")

#train_losses = []
#test_losses = []
#acc=[]
#model.load_state_dict(torch.load("model_weights.pth", weights_only=True))

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    
    #Load model weights
    #if t > 1:
        #model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
        #model.load_state_dict(torch.load("model_weights.pth"))


    start_time=time.time()

    #loss, acc = pytorch_test.test(test_dataloader, model, loss_fn)
    #test_losses.append(loss)


    pytorch_train.train(train_dataloader, model, loss_fn, optimizer)
    

    
   # model_filename="model_weights.pth"
    #torch.save(model.state_dict(), model_filename)

    end_time=time.time()

    time_diff=end_time-start_time
    print(f"Time for Epoch {t+1}: {time_diff} sec \n")

    pred,X_0,X_1=pytorch_test.test(test_dataloader, model, loss_fn)
    
    total_time = total_time+time_diff
    
print("Done!")

avg_time= total_time/epochs
print(f"Average training time: {avg_time} seconds \n")

#Save model weights
torch.save({
    'epoch': t,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_fn,
    }, checkpoint_path)

#torch.save(model.state_dict(), model_filename)
print(f"Saved PyTorch Model State to {checkpoint_path}\n")

fig,ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_0[:,0],X_0[:,1])
ax.scatter(X_1[:,0],X_1[:,1])

plt.title("MODEL")
plt.grid(True)
plt.show()



