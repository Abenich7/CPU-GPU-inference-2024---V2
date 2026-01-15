# goal: Implement accuracy calculation for model evaluation

import torch
from torchvision import datasets, transforms
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

#define project root for all model files
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define transformations for the train/test dataset
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])




# ======== Prepare the dataset =========
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

path='/home/benilla/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images'

full_data_train = datasets.ImageFolder(root=path, transform=train_transform)
full_data_test = datasets.ImageFolder(root=path, transform=test_transform)
                                      

# 2. Split the INDICES first
# Stratify ensures all 120 classes are represented in both sets
indices = list(range(len(full_data_train)))
train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.2, 
    stratify=full_data_train.targets, 
    random_state=42
)

# 3. Apply those indices to the respective dataset views
train_dataset = Subset(full_data_train, train_idx)
test_dataset = Subset(full_data_test, test_idx)

# Create DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=2, pin_memory=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=2, pin_memory=True, batch_size=32)

history = {
    'epoch': [], 'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
}

#evaluate the model between training epochs
def test_model(model,test_loader):
    model.eval()
    test_loss, test_labels, test_preds = 0.0, [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
         
            images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True), labels.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    print(f"🧪 Test Loss:  {avg_test_loss:.4f} | Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

    return avg_test_loss, test_acc, test_precision, test_recall, test_f1




# Train the model
def train_model(model,train_loader,learning_rate,num_epochs):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training...")

    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate)

   
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

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        running_loss = 0.0

        # Evaluate on test set
        val_loss, val_acc, val_precision, val_recall, val_f1 = test_model(model,test_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        torch.save(model.state_dict(), pth_file)
        print("Model saved with updated weights.")

      

    return model




def finetune_model(model,train_loader,test_loader,phase1_epochs=5,phase1_lr=1e-3,phase2_epochs=20,phase2_lr=1e-4):
    #PHASE 1: Freeze all except final layer
    
    # Option 1: Freeze all except the final layer (for fast adaptation)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    model.eval()          # freeze BN stats
    model.fc.train()      # train classifier only


    # Train the model
    model = train_model(model,train_loader,phase1_lr,phase1_epochs)
    

    #PHASE 2: Fine-tune entire model
    
    for param in model.parameters():
        param.requires_grad = True

    model.train()

    model = train_model(model,train_loader,phase2_lr,phase2_epochs)

    test_model(model,test_loader)

    torch.save(model.state_dict(), pth_file)

    #append to history file 
    np.save(history_file, history)

    


# =========== LOAD MODEL =============

# Load model 
model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')

in_ftr=model.fc.in_features
out_ftr=120
model.fc=torch.nn.Linear(in_ftr,out_ftr,bias=True)

#Load weights
pth_file=os.path.join(PROJECT_ROOT,'resnet18_finetuned.pth')
model.load_state_dict(torch.load(pth_file),strict=True)


model.to('cuda' if torch.cuda.is_available() else 'cpu')


# path to history file
history_file=os.path.join(PROJECT_ROOT,'training_history.txt')




#model=train_model(model,train_loader,learning_rate=1e-4,num_epochs=30)
#print("Trainig complete.")

test_model(model,test_loader)

#torch.save(model.state_dict(), "resnet18_finetuned.pth")
dummy_input = torch.randn(1, 3, 224, 224)
dummy_input = dummy_input.to('cuda' if torch.cuda.is_available() else 'cpu')
onnx_path = os.path.join(PROJECT_ROOT,'resnet18_finetuned.onnx')

torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)
    


#engine=build_trt_engine(model)
#engine_path="trt_engine"




# Load dataset to get class labels
#dataset=datasets.ImageFolder(root=path)

#classes=dataset.classes


# This uses the index in the tuple to look up the name in dataset.classes
#true_class_indexes = [idx for path, idx in dataset.imgs]
#true_class_names = [dataset.classes[idx] for idx in true_class_indexes]




#load sample image
#define images path
images_path=os.path.join(PROJECT_ROOT,'examples')

with open(os.path.join(images_path, "tibettan_mastiff.jpeg"), "rb") as f:
    image = matplotlib.image.imread(f)

# run inference on model
outputs = model(test_transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
probabilities = torch.nn.functional.softmax(outputs, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()
#print(f"Predicted class (PyTorch model): {classes[predicted_class]}")


# Get top 5 predictions
#top5_idx = torch.argsort(outputs[0])[-5:][::-1]
#print("Top 5 predictions:")
#for idx in top5_idx:
 #   print(f"{labels[idx]}: {outputs[0][idx]:.2f}%")

# create batch of pictures  


image = np.array(image).astype(np.float32)
image = np.transpose(image, (2, 0, 1))  # Change to CxHxW
image = np.expand_dims(image, axis=0)  # Add batch


#run on tensorrt engine
#print index and class for first 5 images in the set
#for i in range(5):
#    print(f"Index {true_class_indexes[i]}: Class {true_class_names[i]}")



batch_size=1
output,labels=run_inference("trt_engine",path,batch_size)

accuracy=0
total=0
preds=[]


# Now you can iterate through individual image results
for i in range(len(output)):
    single_img_logits = torch.tensor(output[i]).unsqueeze(0)
    probabilities = torch.nn.functional.softmax(single_img_logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    preds.append(predicted_class)

test_acc = accuracy_score(labels, preds)

print(f"Overall Accuracy from TensorRT engine: {test_acc*100:.2f}%")



# Get top 5 predictions
#top5_idx = np.argsort(output[0])[-5:][::-1]
#print("Top 5 predictions:")
#for idx in top5_idx:
 #   print(f"{labels[idx]}: {output[0][idx]:.2f}%")




#finetune_model(model,train_loader,test_loader)
