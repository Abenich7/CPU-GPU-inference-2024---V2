from collections import OrderedDict
import torch
from torch import nn as nn




class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(784, 256)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(128, 10)),
        ]))

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

model = NeuralNetwork()
print(len(model.net))        # ✅ 5
print(model.net[1])          # ✅ relu1
print(model.net['fc2'])      # ✅ Linear(256 → 128)
