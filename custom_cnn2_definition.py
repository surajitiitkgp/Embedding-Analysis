import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the custom CNN model
class CustomCNN2(nn.Module):
    def __init__(self):
        super(CustomCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)  # Adjusting for the input image size of 28x28
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)  # Flattening the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
