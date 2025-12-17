import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import os
import time

# -----------------------------
# Step 1: Define the CNN model
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
# Step 2: Load datasets
# -----------------------------
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Step 3: Load pretrained model
# -----------------------------
device = torch.device("cpu")
model = SmallCNN()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.to(device)
model.train()
print("✅ Original model loaded successfully")

# -----------------------------
# Step 4: Apply structured pruning
# -----------------------------
# Prune 30% channels (L1-norm) from conv layers
prune.ln_structured(model.conv1, name='weight', amount=0.3, n=1, dim=0)
prune.ln_structured(model.conv2, name='weight', amount=0.3, n=1, dim=0)

# Remove pruning re-parametrization to make pruning permanent
prune.remove(model.conv1, 'weight')
prune.remove(model.conv2, 'weight')
print("✅ Structured pruning applied (30% channels pruned)")

# -----------------------------
# Step 5: Fine-tune to recover accuracy
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_finetune_epochs = 2  # just a few epochs to recover

for epoch in range(num_finetune_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Fine-tune Epoch [{epoch+1}/{num_finetune_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# -----------------------------
# Step 6: Measure model size
# -----------------------------
torch.save(model.state_dict(), "temp_pruned_model.pt")
model_size = os.path.getsize("temp_pruned_model.pt") / (1024*1024)
os.remove("temp_pruned_model.pt")

# -----------------------------
# Step 7: Measure accuracy & inference time
# -----------------------------
model.eval()
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
inference_time = ((end_time - start_time) / total) * 1000

# -----------------------------
# Step 8: Print metrics
# -----------------------------
print(f"Pruned + Fine-tuned Model Size: {model_size:.2f} MB")
print(f"Pruned + Fine-tuned Inference Time: {inference_time:.2f} ms/sample")
print(f"Pruned + Fine-tuned Accuracy: {accuracy:.2f}%")
