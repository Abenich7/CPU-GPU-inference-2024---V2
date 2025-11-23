import torch
import numpy as np
import matplotlib.pyplot as plt

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    device="cpu"
    X_0=[]
    X_1=[]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
                 # Loop over batch samples
            for i in range(X.size(0)):
                if pred[i].argmax() == 0:
                    X_0.append(X[i].cpu().numpy())  # Store the sample in X_0
                elif pred[i].argmax() == 1:
                    X_1.append(X[i].cpu().numpy())  # Store the sample in X_1

                
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
   
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)



 
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
   
  
    return pred, X_0, X_1,test_loss
