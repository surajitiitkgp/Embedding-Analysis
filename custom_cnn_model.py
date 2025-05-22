import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This creates a folder named saved_models to save the trained model.
os.makedirs("saved_models", exist_ok=True)

# Define a simple CNN Model, This is your custom CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Model Structure
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #Layer 1
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16, # Shrinks size from 32x32 to 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Layer 2
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 shrink size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), #into 1D
            nn.Linear(64 * 8 * 8, 128), #Dense layer with 128 units
            nn.ReLU(),
            nn.Linear(128, 10)  # CIFAR-10 has 10 classes, Final layer: 10 outputs
        )
    
    #  forward method tells the model how to process inputs through the layers.
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Transformations for training data
transform = transforms.Compose([
    transforms.ToTensor(), #convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss, optimizer
model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (very short for demo)
num_epochs = 3
print("Training started...")
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Save the entire model
save_path = "saved_models/custom_cnn.pth" #This saves the model's learned parameters (not the full model structure) to a file.
torch.save(model.state_dict(), "saved_models/custom_cnn.pth")
print(f"Model saved successfully to {save_path}")





"""
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 170M/170M [02:50<00:00, 997kB/s]
Training started...
Epoch 1/3, Loss: 1082.7830
Epoch 2/3, Loss: 792.4240
Epoch 3/3, Loss: 676.1203
Model saved successfully to saved_models/custom_cnn.pth
"""