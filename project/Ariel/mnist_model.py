# ============================================================================
# file: mnist_model.py
# ----------------------------------------------------------------------------
# Simple fully‑connected network for MNIST (28×28 grayscale → 10 classes)
# ============================================================================
import torch
from torch import nn as nn


class NeuralNetwork(nn.Module):
    """A lightweight MLP that reaches ~95% accuracy after a few epochs."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 1×28×28 ⇒ 784
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 logits for 0‑9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)

if __name__ == "__main__":
    model = NeuralNetwork()
    print(model)
    