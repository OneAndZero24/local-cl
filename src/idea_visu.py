import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, TensorDataset

from model.layer.rbf import RBFLayer, rbf_gaussian, l_norm


X1, y1 = make_blobs(n_samples=1000, centers=2, random_state=42)
X1 = torch.tensor(X1, dtype=torch.float32)
y1 = torch.tensor(y1, dtype=torch.long)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.layers(x)
    
class LocalNet(nn.Module):
    def __init__(self):
        super(LocalNet, self).__init__()
        self.layers = nn.Sequential(
            RBFLayer(2, 4, 32, rbf_gaussian, l_norm, False),
        )

    def forward(self, x):
        return self.layers(x)


def train(model, X, y, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

def visualize(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        Z = model(grid)[:, 1].numpy()

    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.8)

    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='coolwarm', edgecolor='k')

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("NN Output")
    ax.set_title("3D Plot of Neural Network Function")

    plt.show()

model = Net()
model_local = LocalNet()

train(model, X1, y1, 50)
visualize(model, X1, y1)

train(model_local, X1, y1, 50)
visualize(model_local, X1, y1)