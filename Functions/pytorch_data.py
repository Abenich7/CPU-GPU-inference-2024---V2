import sys
import os
# Add your Python310/lib path at the start of sys.path
#sys.path.insert(0, 'C:/Python310/Lib') 
FN_PATH=r'C:\Users\Abeni07\source\repos\CPU-GPU-inference-2024\Functions'

functions_dir=os.path.join(FN_PATH)
sys.path.insert(0,functions_dir)

import create_data
import means_covs
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc

import torch
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_circles

# Example usage
DATA_PATH=r'C:\Users\Abeni07\data'

# Define the data directory
data_dir = os.path.join(DATA_PATH)
os.makedirs(data_dir, exist_ok=True)




def raw_data_to_tensors(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

# סטנדרטיזציה של הנתונים
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


# המרה לטנסורים
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# יצירת DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset,test_dataset

    

def save_dataset_with_new_name(dataset, data_dir, new_name, dataset_type='train'):
  
    # Construct the new path
    filename = f"{new_name}_{dataset_type}_dataset.pt"
    new_dataset_path = os.path.join(data_dir, filename)

    # Check if the file exists
    #if os.path.exists(new_dataset_path):
     #   print(f"Error: File with name {filename} already exists in {data_dir}.")
      #  return

    # Save the dataset
    torch.save(dataset, new_dataset_path)
    print(f"{dataset_type.capitalize()} dataset saved to {new_dataset_path}")



##### DATASET 1: GAUSSIAN CLOUDS for NEURAL NETWORK

num_samples=10000
num_classes=2
train_ratio = 0.8
spread = 10
overlap_percentage = 0.001 # אחוז חפיפה של 30%
sigma = 1.0


means,covs=means_covs.generate_means_covs(
    num_classes, overlap_percentage, sigma=1.0, random_state=42
)


X,y = create_data.create_data_fn(num_samples,num_classes,means,covs)

train_dataset,test_dataset=raw_data_to_tensors(X,y)


# Save train and test datasets with a new name
save_dataset_with_new_name(train_dataset, data_dir, new_name='gaussian_cloud_tensor', dataset_type='train')
save_dataset_with_new_name(test_dataset, data_dir, new_name='gaussian_cloud_tensor', dataset_type='test')

# Plot each class in a different color
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    X_class = X[y == i]  # Extract points belonging to class `i`
    plt.scatter(X_class[:, 0], X_class[:, 1], label=f"Class {i}")  # Scatter plot

  
plt.title("DATA GAUSSIAN CLOUDS")
plt.grid(True)
plt.show()


###### Dataset 2: CONCENTRIC CIRCLES for NEURAL NET

# פרמטרים ליצירת הדאטה
num_samples = 10000
noise = 0.05  # רעש להוספה לנתונים
factor = 0.5  # היחס בין רדיוס העיגולים
train_ratio=0.8

# יצירת הדאטה
X, y = make_circles(n_samples=num_samples, noise=noise, factor=factor, random_state=42)

train_dataset,test_dataset=raw_data_to_tensors(X,y)

# Save train and test datasets with a new name
save_dataset_with_new_name(train_dataset, data_dir, new_name='concentric_circles_tensor', dataset_type='train')
save_dataset_with_new_name(test_dataset, data_dir, new_name='concentric_circles_tensor', dataset_type='test')


# ויזואליזציה של הדאטה
plt.figure(figsize=(6,6))
plt.scatter(X[y==0][:,0], X[y==0][:,1], label='Class 0', alpha=0.5)
plt.scatter(X[y==1][:,0], X[y==1][:,1], label='Class 1', alpha=0.5)
plt.legend()
plt.title("Two Concentric Circles")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


#### Dataset 3: Gaussian Data 

num_samples=10000
num_classes=2
train_ratio = 0.8
spread = 10
overlap_percentage = 0.001 # אחוז חפיפה של 30%
sigma = 1.0


means,covs=means_covs.generate_means_covs(
    num_classes, overlap_percentage, sigma, random_state=42
)


X,y = create_data.create_data_fn(num_samples,num_classes,means,covs)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

# Save train and test datasets with a new name
save_dataset_with_new_name(train_dataset, data_dir, new_name='Gaussian_cloud_array', dataset_type='train')
save_dataset_with_new_name(test_dataset, data_dir, new_name='Gaussian_cloud_array', dataset_type='test')





##### Data 4: Make circles

# פרמטרים ליצירת הדאטה
num_samples = 10000
noise = 0.05  # רעש להוספה לנתונים
factor = 0.5  # היחס בין רדיוס העיגולים


# יצירת הדאטה
X, y = make_circles(n_samples=num_samples, noise=noise, factor=factor, random_state=42)


# חלוקת הנתונים
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

# סטנדרטיזציה של הנתונים
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save train and test datasets with a new name
save_dataset_with_new_name(train_dataset, data_dir, new_name='concentric_circles_array', dataset_type='train')
save_dataset_with_new_name(test_dataset, data_dir, new_name='concentric_circles_array', dataset_type='test')


