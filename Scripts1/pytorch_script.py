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


batch_size = 512

# Load datasets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
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


# Paths and Initialization
model_dir = r"C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Models\trained_models"
timestamp = int(time.time())
model_name = os.path.join(model_dir, f"model_weights_{timestamp}.pth")
os.makedirs(model_dir, exist_ok=True)

total_time = 0  # Initialize total time counter

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    if t > 1:
      #  model_name = os.path.join(model_dir, model_filename)
        model.load_state_dict(torch.load(model_name,weights_only=True))  # Load model checkpoint

    start_time = time.time()

    # Training
    pytorch_train.train(train_dataloader, model, loss_fn, optimizer)

    # Save model weights
    
    torch.save(model.state_dict(), model_name)

    end_time = time.time()

    # Compute and accumulate epoch time
    time_diff = end_time - start_time
    total_time += time_diff
    print(f"Time for Epoch {t+1}: {time_diff:.2f} seconds\n")

    # Testing
    pred, X_0, X_1 = pytorch_test.test(test_dataloader, model, loss_fn)

print("Done!")

# Average Training Time
avg_time = total_time / epochs
print(f"Average training time: {avg_time:.2f} seconds\n")

# Visualization
X_0 = X_0.detach().cpu().numpy()
X_1 = X_1.detach().cpu().numpy()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_0[:, 0], X_0[:, 1], label='Class 0', alpha=0.7)
ax.scatter(X_1[:, 0], X_1[:, 1], label='Class 1', alpha=0.7)
plt.title("Model Output")
plt.grid(True)
plt.legend()
plt.show()