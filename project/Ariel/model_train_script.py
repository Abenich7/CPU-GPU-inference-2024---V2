# Train the model
model_path=os.path.join(PROJECT_ROOT,'resnet18_finetuned_stanford_dogs.pth')



full_data_train = datasets.ImageFolder(root=path, transform=train_transform)



    #model=load_model(model_path)
    
model=torch.hub.load('pytorch/vision', 'resnet18', weights='IMAGENET1K_V1')

in_ftr=model.fc.in_features
out_ftr=120
model.fc=torch.nn.Linear(in_ftr,out_ftr,bias=True)


    
 #   model.train()
#file_path=Path(model_path)

#if file_path.exists(): 
 #   print(f"using weights from {model_path}")
  #  model.load_state_dict(torch.load(model_path))

model.to(device)


# Option 1: Freeze all except the final layer (for fast adaptation)
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

model.eval()          # freeze BN stats
model.fc.train()      # train classifier only



#model.train()
train_model(full_data_train,1e-3,600,0.9,model_path)
