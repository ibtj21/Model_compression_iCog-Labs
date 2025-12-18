import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch #Core PyTorch library
import torch.nn as nn #Tools for defining neural network layers
import torch.nn.functional as F #Activation functions and operations
from torchvision import datasets, transforms #Dataset and image transformation utilities
from torch.utils.data import DataLoader #Data loading utilities
import os
"""The file begins by configuring the console to properly display text and importing all required 
PyTorch and torchvision libraries. These libraries provide the tools needed to define neural networks,
load datasets, train models, and save learned parameters."""
# -----------------------------
# Step 1: Define  CNN(convolutional neural network)
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # reduces 28x28 -> 14x14 after first pool
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 256)  # 64 channels, 14x14 after pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all except batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""A small convolutional neural network (CNN) is defined. The model is designed to process grayscale
images and extract meaningful features using convolutional layers, followed by pooling to reduce spatial 
dimensions. The extracted features are then passed through multiple fully connected layers to perform 
digit classification. This architecture is intentionally larger than a minimal model so that the effects 
of pruning and quantization can be observed later."""
# -----------------------------
# Step 2: Prepare MNIST dataset
# -----------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
"""The MNIST dataset is downloaded and prepared for training and testing. Images are converted into 
tensors and loaded into DataLoaders, which manage batching and shuffling. This allows efficient and 
organized feeding of data into the model during training and evaluation."""
# -----------------------------
# Step 3: Initialize model, optimizer, loss
# -----------------------------
device = torch.device("cpu")  
model = SmallCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
"""The model is initialized on the selected device (CPU in this case). A loss function suitable for 
multi-class classification and an optimizer are defined. These components determine how prediction 
errors are measured and how model weights are updated during training."""

# -----------------------------
# Step 4: Train the model
# -----------------------------
num_epochs = 3  
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("✅ Model training complete")
"""The model is trained for a fixed number of epochs. During each epoch, training data is passed 
through the network, predictions are made, and errors are computed. The model learns by adjusting 
its weights through backpropagation, gradually reducing the loss as training progresses."""

# -----------------------------
# Step 5: Save the model
# -----------------------------
model_file = "mnist_cnn.pt"
torch.save(model.state_dict(), model_file)
print(f"Model saved as {model_file}")
