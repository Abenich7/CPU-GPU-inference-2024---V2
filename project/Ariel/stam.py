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
from .build_trt_engine import run_inference
import matplotlib
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import csv 
import time
import contextlib

from torch.cuda import Stream
from .profiler_for_inference import benchmark_with_profiler

#define device
device='cuda:1'

#define project root for all model files
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


#history_file=os.path.join(PROJECT_ROOT,'training_history.npy')



# Define transformations for the train/test dataset
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
#   v2.RandomResizedCrop(224, antialias=True),
#   v2.RandomHorizontalFlip(p=0.5),
#   v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
    v2.ToDtype(torch.float16, scale=True),
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
#downloads if it is not in cache yet
# Download latest version
path = '/home/workspace/benilla/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images'
full_data_test = datasets.ImageFolder(root=path, transform=test_transform)
full_data_train = datasets.ImageFolder(root=path, transform=train_transform)



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
#train_dataset = Subset(full_data_train, train_idx)
test_dataset = Subset(full_data_test, test_idx)
train_dataset = Subset(full_data_train, test_idx)


# Create DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, pin_memory=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=True, num_workers=1, pin_memory=True, batch_size=32)

#history = {
 #   'epoch': [], 'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
  #  'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
#}





#evaluate the model between training epochs
def test_model(model_path,test_loader):
    model=load_model(model_path)

    model.eval()
    test_loss, test_labels, test_preds = 0.0, [], []
    criterion = nn.CrossEntropyLoss()

    s=Stream()
    
    with torch.no_grad():
        with tqdm(test_loader,desc="running inference") as t:
                with torch.cuda.stream(s):
                    for images, labels in t:
                        images, labels = images.to(device if torch.cuda.is_available() else 'cpu', non_blocking=True), labels.to(device if torch.cuda.is_available() else 'cpu', non_blocking=True)
                    cuda_h2d_event=s.record_event()
                
                outputs = model(images)
                cuda_output_event=torch.cuda.current_stream().record_event()
                
                cuda_h2d_event.synchronize()
                cuda_output_event.synchronize()

                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())
        
        
    elapsed_time=t.format_dict['elapsed']
    iteration_rate=t.format_dict['rate']

    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    print(f" Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

    return elapsed_time,iteration_rate,test_acc, test_precision, test_recall, test_f1




# Train the model
def train_model(dataset: datasets.ImageFolder,learning_rate:float,training_limit:int,target_accuracy:float,model_path: str):
    model=load_model(model_path)
    # 2. Split the INDICES first
    # Stratify ensures all 120 classes are represented in both sets


    indices = list(range(len(dataset)))

    
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=dataset.targets, 
        random_state=42
    )

    # 3. Apply those indices to the respective dataset views
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    #tiny = Subset(full_data_train, list(range(20)))
    


    train_loader = DataLoader(train_dataset,batch_size=16, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset,batch_size=32, shuffle=False, num_workers=2,pin_memory=True)




    criterion = torch.nn.CrossEntropyLoss()

    casting=torch.get_autocast_dtype('cuda')
    print(f"Starting training with quantization {casting}")

    #optimizer = torch.optim.Adam(
    #filter(lambda p: p.requires_grad, model.parameters()),
    #lr=learning_rate)

    optimizer=torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    
    start_time=time.time()
    Epoch=0
    accuracy_reached=False
    try: 
        while True:
            model.eval()
            model.fc.train()
            Epoch+=1
            print(f'Epoch {Epoch}')

         #   model.train()

            for images, labels in tqdm(train_loader, desc='Training'):
                
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

              
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()

            
        
            # Evaluate on test set as a sort of scheduler        
            model.eval()
            test_loss, test_labels, test_preds = 0.0, [], []
            

         #   s=Stream()
        
            with torch.no_grad():
                with tqdm(test_loader,desc="running inference") as t:
                        for images, labels in t:
                  #          with torch.cuda.stream(s):
                            images= images.to(device)
                            labels=labels.to(device)
                          #  cuda_h2d_event=s.record_event()
                            with torch.autocast(device_type="cuda",dtype=torch.float16):
                                outputs = model(images)
                         #   cuda_output_event=torch.cuda.current_stream().record_event()
                        
                       #     cuda_h2d_event.synchronize()
                        #    cuda_output_event.synchronize()

                            loss = criterion(outputs, labels)
                            test_loss += loss.item()
                            _, preds = torch.max(outputs, 1)
                            test_labels.extend(labels.cpu().numpy())
                            test_preds.extend(preds.cpu().numpy())
                
            
        
                test_acc = accuracy_score(test_labels, test_preds)
                print(f"Accuracy:{test_acc}")
                

                if test_acc > target_accuracy:
                    accuracy_reached=True
                    break
                
                time_elapsed=time.time()-start_time
                

                if time_elapsed > training_limit:
                    break
    
    except KeyboardInterrupt:
        torch.save(model.state_dict(),model_path)
            
        print(f"Model saved with updated weights at path: {model_path}.")

        torch.save(model.fc.state_dict(), "fc_head_resnet18_stanford_dogs.pth")


        return
    
    except TimeoutError:
        torch.save(model.state_dict(),model_path)
            
        print(f"Model saved with updated weights at path: {model_path}.")

        return
    

              

    time_elapsed=time.time()-start_time

    if accuracy_reached:
        print(f"model reached {target_accuracy} accuracy in {time_elapsed} seconds")
    
    torch.save(model.state_dict(),model_path)
            
    print(f"Model saved with updated weights at path: {model_path}.")

              

        
       # history['epoch'].append(epoch + 1)
       # history['val_loss'].append(val_loss)
       # history['val_acc'].append(val_acc)
       # history['val_precision'].append(val_precision)
       # history['val_recall'].append(val_recall)
       # history['val_f1'].append(val_f1)

       # np.save(history_file, history)
       # print("Training history updated at train_history.txt")

    torch.save(model.state_dict(),model_path)
        
    print(f"Model saved with updated weights at path: {model_path}.")

    




def finetune_model(train_loader,test_loader,phase1_epochs=5,phase1_lr=1e-3,phase2_epochs=20,phase2_lr=1e-4):
    
    #model=load_model(model_path)
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

    # path to history file


    #append to history file 
    np.save(history_file, history)


def test_model_fp16_no_calibration(model_path,test_loader):

    model=load_model(model_path)

    model.half()

    

    model.eval()
    test_loss, test_labels, test_preds = 0.0, [], []
    criterion = nn.CrossEntropyLoss()

    s=Stream()
    
    with torch.no_grad():
            with tqdm(test_loader,desc="running inference") as t:
                for images, labels in t:
                    with torch.cuda.stream(s):
                        images, labels = images.to(device if torch.cuda.is_available() else 'cpu', non_blocking=True).half(), labels.to(device if torch.cuda.is_available() else 'cpu', non_blocking=True)

                    cuda_h2d_event=s.record_event()
                
                outputs = model(images)
                cuda_output_event=torch.cuda.current_stream().record_event()
                
                cuda_h2d_event.synchronize()
                cuda_output_event.synchronize()

                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(preds.cpu().numpy())
            
        
    elapsed_time=t.format_dict['elapsed']
    iteration_rate=t.format_dict['rate']

    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    print(f" Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

    return elapsed_time,iteration_rate,test_acc, test_precision, test_recall, test_f1








# =========== SCRIPTS =============

# Load model 

def load_model(model_path):
    
    model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')

    in_ftr=model.fc.in_features
    out_ftr=120
    model.fc=torch.nn.Linear(in_ftr,out_ftr,bias=True)

    #Load weights
    
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT,model_path)))


    model.to(device if torch.cuda.is_available() else 'cpu')

    return model


#model=load_model()




#test_model(model,test_loader)

#torch.save(model.state_dict(), "resnet18_finetuned.pth")
def create_onnx(model):
    dummy_input = torch.randn(128, 3, 224, 224)
    dummy_input = dummy_input.to(device if torch.cuda.is_available() else 'cpu')
    onnx_path = os.path.join(PROJECT_ROOT,'resnet18_finetuned.onnx')


    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)

#model=load_model('resnet18_finetuned.pth')
#create_onnx(model)

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
def infer_single_image(model):
    images_path=os.path.join(PROJECT_ROOT,'examples')

    with open(os.path.join(images_path, "tibettan_mastiff.jpeg"), "rb") as f:
        image = matplotlib.image.imread(f)

    # run inference on model
    outputs = model(test_transform(image).unsqueeze(0).to(device if torch.cuda.is_available() else 'cpu'))
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    #print(f"Predicted class (PyTorch model): {classes[predicted_class]}")


#infer_single_image()


# Get top 5 predictions
#top5_idx = torch.argsort(outputs[0])[-5:][::-1]
#print("Top 5 predictions:")
#for idx in top5_idx:
 #   print(f"{labels[idx]}: {outputs[0][idx]:.2f}%")

# create batch of pictures  




#run on tensorrt engine
#print index and class for first 5 images in the set
#for i in range(5):
#    print(f"Index {true_class_indexes[i]}: Class {true_class_names[i]}")


def trt_inference_script(model_path):
    output,labels,elapsed_time,iteration_rate=run_inference(model_path)

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
    test_precision = precision_score(labels, preds, average='weighted', zero_division=0)
    test_recall = recall_score(labels, preds, average='weighted', zero_division=0)
    test_f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    print(f" Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

    return elapsed_time,iteration_rate,test_acc,test_precision,test_recall,test_f1
#build_trt_engine(model)

#trt_inference_script()

# Get top 5 predictions
#top5_idx = np.argsort(output[0])[-5:][::-1]
#print("Top 5 predictions:")
#for idx in top5_idx:
 #   print(f"{labels[idx]}: {output[0][idx]:.2f}%")






def prepare_model_dataset(
    model_name,
    model_type,
    model_quantization,
    batch_size,
    num_workers
):
    # ---------- MODEL ----------
    #based on model key return model path
    model_key = (model_name, model_type,model_quantization,batch_size,num_workers)
    
    
    model_path=models[model_key]
    
   
    #define dataloader
    
    dataloader=DataLoader(
            test_dataset,
           
            batch_size=batch_size,

            num_workers=num_workers,
           
            pin_memory=True,
            
           
        )


    return model_path, dataloader


# call inference functions 


def experiments_data(configurations):
    results = []
    
    file_path = "experiment_results.csv"

    fieldnames = [
        "model_name", "model_type", "quantization", "batch_size",
        "elapsed_time",
        "accuracy", "precision", "recall", "f1"
    ]

    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)


        for config in configurations:
            model_path, dataloader = prepare_model_dataset(
                config.model_name,
                config.model_type,
                config.model_quantization,
                config.batch_size,
                config.num_workers
            )

            
            
            if config.model_type == "pytorch" and config.model_quantization == "none":
                metrics = test_model(model_path, dataloader)
                benchmark_with_profiler(model_path,dataloader,func=test_model,config=config)
            
            elif config.model_type == "pytorch" and config.model_quantization == "fp16":
                metrics = test_model_fp16_no_calibration(model_path,dataloader)
                benchmark_with_profiler(model_path,dataloader,func=test_model_fp16_no_calibration,config=config)

            elif config.model_type == "trt":
                metrics = trt_inference_script(model_path)
                benchmark_with_profiler(model_path,func=trt_inference_script,config=config)
            else:
                raise ValueError("Invalid model type")

            
            result_row = {
                "model_name": config.model_name,
                "model_type": config.model_type,
                "quantization": config.model_quantization,
                "batch_size": config.batch_size,
                "elapsed_time": metrics[0],
              # "iteration_rate": metrics[1],
                "accuracy": metrics[2],
                "precision": metrics[3],
                "recall": metrics[4],
                "f1": metrics[5],
            }
            results.append(result_row)
            writer.writerow(result_row)

    



############ TESTBENCH #################

#mapping of model names to model (its weights)
#ex: pytorch_quantized_int_8 -> model1.pth
 #   trt_full_precision -> model1.engine

 
@dataclass
class ExperimentConfig:
    model_name: str
    model_type: str        # 'pytorch' | 'trt'
    model_quantization:str # 'FP16' | 'INT8' | ...
    batch_size: int
    num_workers:int


models = {}    # key: (model_name, model_type, model_quantization) -> model_path

#csv format
#    model_name,model_type,model_quantization,model_path
#    resnet18,cnn,int8,/models/resnet18_int8.onnx

# ---- Load models ----
with open("models.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows or rows without batch_size
            if not row.get("batch_size") or not row.get("model_name"):
                continue
            key = (
                row["model_name"],
                row["model_type"],
                row["model_quantization"],
                int(row["batch_size"]),
                int(row["num_workers"])
            )
            models[key] = row["model_path"]

    
#save names->path mapping to csv


configs = [
    ExperimentConfig(
        model_name=model_name,
        model_type=model_type,
        model_quantization=model_quantization,
        batch_size=batch_size,
        num_workers=num_workers
    )
    for model_name, model_type, model_quantization, batch_size,num_workers in models.keys()
]

#experiments_data(configs)
#model_path,data_test=prepare_model_dataset('resnet18','trt','FP16',64)


#benchmark_with_profiler(model_path,func=run_inference,**desc)


#train_model(full_data_train,1e-4,3600,0.9,'resnet18_dogs_80%_acc.pth')

experiments_data(configs)


