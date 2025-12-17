import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time

# -----------------------------
# Step 1: Define the same CNN architecture
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------
# Step 2: Load MNIST test dataset
# -----------------------------
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Step 3: Load trained model
# -----------------------------
device = torch.device("cpu")  # change to 'cuda' if GPU available
model = SmallCNN()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully")

# -----------------------------
# Step 4: Measure Model Size
# -----------------------------
torch.save(model.state_dict(), "temp_model.pt")
model_size = os.path.getsize("temp_model.pt") / (1024*1024)  # size in MB
os.remove("temp_model.pt")

# -----------------------------
# Step 5: Measure Accuracy & Inference Time
# -----------------------------
correct = 0
total = 0
start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

end_time = time.time()

accuracy = (correct / total) * 100
inference_time = ((end_time - start_time) / total) * 1000  # ms/sample

# -----------------------------
# Step 6: Print results
# -----------------------------
print(f"Model Size: {model_size:.2f} MB")
print(f"Inference Time: {inference_time:.2f} ms/sample")
print(f"Accuracy: {accuracy:.2f}%")
