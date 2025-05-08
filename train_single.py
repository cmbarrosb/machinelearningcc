#!/usr/bin/env python3
"""
train_single.py

Trains a simple LeNet-style CNN on MNIST for 5 epochs on CPU.
Logs total training time, final test accuracy, and images/sec to train_single.log.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ---- Config ----
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.01
DEVICE = torch.device('cpu')
LOG_FILE = 'train_single.log'

# ---- Model ----
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---- Data ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST('.', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ---- Training & Evaluation ----
def train(model, device, loader, optimizer, criterion):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)

def main():
    model = LeNet().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, criterion)
    elapsed = time.time() - start_time

    accuracy = test(model, DEVICE, test_loader)
    images_per_sec = len(train_ds) * EPOCHS / elapsed

    # ---- Log results ----
    with open(LOG_FILE, 'a') as f:
        f.write(f"Time: {elapsed:.2f}s\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Images/sec: {images_per_sec:.2f}\n")

    print(f"Done. See {LOG_FILE} for metrics.")

if __name__ == '__main__':
    main()
