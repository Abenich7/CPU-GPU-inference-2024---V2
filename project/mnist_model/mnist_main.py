# ============================================================================
# file: mnist_main.py
# ----------------------------------------------------------------------------
# End‑to‑end training script.  Assumes dataset is already downloaded inside
# a folder named "mnist" located in the current project directory.
# ============================================================================


import time
from pathlib import Path
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from mnist_model import NeuralNetwork
from mnist_train import train_epoch, evaluate


# ------------------------------- 1. paths ----------------------------------
ROOT_DIR = Path(__file__).resolve().parent          # project root
DATA_DIR = ROOT_DIR / "mnist"                      # contains raw MNIST files
OUT_DIR  = ROOT_DIR                    # where artefacts go
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------- 2. data -----------------------------------
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=TRANSFORM)
test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=TRANSFORM)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,num_workers=32,pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=64,num_workers=32,pin_memory=True)

# ------------------------------- 3. model ----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
#print(model.net[1])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------- 4. train ----------------------------------
EPOCHS = 5
for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
    dt = time.time() - t0
    print(f"Epoch {epoch+1}/{EPOCHS} | train‑loss {train_loss:.4f} | ")

acc, val_loss = evaluate(test_loader, model, loss_fn, device)
print(   f"val‑loss {val_loss:.4f} | accuracy {acc*100:.2f}% | {dt:.1f}s")
# ------------------------------- 5. save -----------------------------------
MODEL_PATH = OUT_DIR / "mnist_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"[✓] saved weights → {MODEL_PATH}")



#plotting function
def plot_metrics(train_losses, val_losses, accuracies):
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, img, true_lbl, pred_lbl in zip(axes.flatten(),
                                           X_batch[:16],
                                           y_batch[:16],
                                           preds[:16]):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"T:{true_lbl} P:{pred_lbl}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sample_predictions.png", dpi=300)
    plt.show()
    print(f"[✓] sample plot saved → {OUT_DIR/'sample_predictions.png'}")


# ------------------------------- 6. quick demo -----------------------------


        

    
    
    
    
    
    