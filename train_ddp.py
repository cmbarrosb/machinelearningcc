

#!/usr/bin/env python3
"""
train_ddp.py

Trains a simple LeNet-style CNN on MNIST using PyTorch DistributedDataParallel across multiple nodes.
Logs total training time, final test accuracy, and images/sec to train_ddp.log (only on rank 0).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# Configuration
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.01
LOG_FILE = 'train_ddp.log'

def parse_env():
    """Parse required environment variables for DDP."""
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ.get('MASTER_PORT', '29500')
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    return master_addr, master_port, world_size, rank

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def main():
    master_addr, master_port, world_size, rank = parse_env()
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initialize process group
    torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device = torch.device('cpu')
    model = LeNet().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model)

    # Dataset and distributed sampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.SGD(ddp_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training
    if rank == 0:
        start_time = time.time()
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        ddp_model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation and logging (only on rank 0)
    if rank == 0:
        elapsed = time.time() - start_time
        ddp_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = ddp_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        accuracy = correct / len(test_ds)
        images_per_sec = len(train_ds) * EPOCHS / elapsed
        with open(LOG_FILE, 'a') as f:
            f.write(f"Time: {elapsed:.2f}s\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"Images/sec: {images_per_sec:.2f}\n")
        print(f"Done on rank 0. Metrics in {LOG_FILE}")

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()