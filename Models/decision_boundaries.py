import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Generate concentric circles dataset
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Define the model (same as user's)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 128),
            #nn.ReLU(),
            nn.Linear(128, 128),
            #nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model and collect decision boundaries
boundaries = []
epochs = [0, 50, 200, 500]
for epoch in range(501):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch in epochs:
        # Record decision boundary snapshot
        xx, yy = torch.meshgrid(torch.linspace(-1.5, 1.5, 100), torch.linspace(-1.5, 1.5, 100), indexing='xy')
        grid = torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=1)
        with torch.no_grad():
            zz = model(grid).argmax(dim=1).reshape(xx.shape)
        boundaries.append((epoch, xx, yy, zz))

# Plot boundaries
fig, axes = plt.subplots(1, len(boundaries), figsize=(14,3))
for ax, (epoch, xx, yy, zz) in zip(axes, boundaries):
    ax.contourf(xx, yy, zz, cmap='coolwarm', alpha=0.7)
    ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=8, edgecolors='k')
    ax.set_title(f"Epoch {epoch}")
plt.show()
