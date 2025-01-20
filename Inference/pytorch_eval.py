import sys
import os

MODELS_PATH=r'C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Models'

models_dir=os.path.join(MODELS_PATH)
sys.path.insert(0,models_dir)


TRAINING_PATH=r'C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Training'

training_dir=os.path.join(TRAINING_PATH)
sys.path.insert(0,training_dir)


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import pytorch_model
import pytorch_train
import numpy as np
import pytorch_test
import os
import torchvision.models as models
import pytorch_test
import matplotlib.pyplot as plt


TEST_DATASET_PATH = r"C:\Users\Abeni07\data\concentric_circles_tensor_test_dataset.pt"

device="cpu"

model = pytorch_model.NeuralNetwork().to(device)
#model = models.vgg16()
model.load_state_dict(torch.load(r"C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Models\trained_models\model_weights.pth", weights_only=True))


loss_fn = nn.CrossEntropyLoss()

num_samples=2000 

#train_dataset_path='CPU-GPU-inference-2024\data\gaussian_train_dataset.pt'


# Check if the dataset file exists
if os.path.exists(TEST_DATASET_PATH):
    # Load the dataset from the file
    test_dataset = torch.load(TEST_DATASET_PATH)
    print("Test dataset loaded from file.")
else:
    print("No data available. Ensure the file exists at:", TEST_DATASET_PATH)


# Load datasets
batch_size=64
test_dataloader = DataLoader(test_dataset,batch_size=batch_size)

###
## X_train_tensor = 

#test indexes randomly to see models performance

#pick a random index 
rng=np.random.default_rng()
sample_index=rng.integers(0,num_samples-1)    #enter index to test 


pred,X_0,X_1=pytorch_test.test(test_dataloader,model,loss_fn)

fig,ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_0[:,0],X_0[:,1])
ax.scatter(X_1[:,0],X_1[:,1])

plt.title("MODEL")
plt.grid(True)
plt.show()







#with torch.no_grad():
 #   for X, y in test_dataloader:
  #      print(X)
   #     print(y)
    #    X,y = X.to(device),y.to(device)
     #   predict = model(X)
      #  print(predict.argmax(1))
        
     #   predicted, actual = classes[predict[0].argmax(1)], classes[y]
        
        

    #    print(f'Predicted: "{predicted}", Actual: "{actual}"')