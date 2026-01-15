# ============================================================================
# file: mnist_train.py
# ----------------------------------------------------------------------------
# Training & evaluation helpers (one epoch each) – device‑agnostic
# ============================================================================
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Tuple


def train_epoch(loader: DataLoader,
                model: nn.Module,
                loss_fn: nn.Module,
                optimizer: optim.Optimizer,
                device: str = "cpu") -> float:
    """Run one epoch; re
    turn average training loss."""
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    #synchronize if using cuda
    if device == "cuda":
        torch.cuda.synchronize()
    return running_loss / len(loader)


def evaluate(loader: DataLoader,
             model: nn.Module,
             loss_fn: nn.Module,
             device: str = "cpu") -> Tuple[float, float]:
    """Return (accuracy, avg_loss) on *loader*."""
    # put the model in eval mode
    model.eval()
    correct = 0
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            running_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    avg_loss = running_loss / len(loader)
    return acc, avg_loss
