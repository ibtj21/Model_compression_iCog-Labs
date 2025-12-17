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
# Step 1: Define the original CNN model
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
# Step 2: Load test data
# -----------------------------
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Step 3: Load trained model
# -----------------------------
device = torch.device("cpu")
model = SmallCNN()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.to(device)
model.eval()
print("✅ Original model loaded successfully")

# -----------------------------
# Step 4: Identify channels to keep after pruning
# -----------------------------
# Prune 30% of channels in conv layers
amount = 0.3

# Conv1
weights = model.conv1.weight.data.abs().mean(dim=(1,2,3))
num_keep1 = int(weights.size(0)*(1-amount))
keep_idx1 = torch.topk(weights, num_keep1)[1]

# Conv2
weights2 = model.conv2.weight.data.abs().mean(dim=(1,2,3))
num_keep2 = int(weights2.size(0)*(1-amount))
keep_idx2 = torch.topk(weights2, num_keep2)[1]

# -----------------------------
# Step 5: Build a new smaller CNN
# -----------------------------
class SmallCNNPruned(nn.Module):
    def __init__(self, keep1, keep2):
        super(SmallCNNPruned, self).__init__()
        self.conv1 = nn.Conv2d(1, len(keep1), kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(len(keep1), len(keep2), kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(len(keep2) * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.keep1 = keep1
        self.keep2 = keep2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

pruned_model = SmallCNNPruned(keep_idx1, keep_idx2).to(device)

# -----------------------------
# Step 6: Copy weights from old model to pruned model
# -----------------------------
pruned_model.conv1.weight.data = model.conv1.weight.data[keep_idx1,:,:,:].clone()
pruned_model.conv1.bias.data = model.conv1.bias.data[keep_idx1].clone()

pruned_model.conv2.weight.data = model.conv2.weight.data[keep_idx2,:,:,:][:,keep_idx1,:,:].clone()
pruned_model.conv2.bias.data = model.conv2.bias.data[keep_idx2].clone()

pruned_model.fc1.weight.data = model.fc1.weight.data[:,keep_idx2.repeat_interleave(14*14)].clone()
pruned_model.fc1.bias.data = model.fc1.bias.data.clone()

pruned_model.fc2.weight.data = model.fc2.weight.data.clone()
pruned_model.fc2.bias.data = model.fc2.bias.data.clone()
pruned_model.fc3.weight.data = model.fc3.weight.data.clone()
pruned_model.fc3.bias.data = model.fc3.bias.data.clone()

pruned_model.eval()
print(f"✅ Structured pruning applied (30% channels removed)")

# -----------------------------
# Step 7: Measure model size
# -----------------------------
torch.save(pruned_model.state_dict(), "temp_pruned_model.pt")
model_size = os.path.getsize("temp_pruned_model.pt") / (1024*1024)
os.remove("temp_pruned_model.pt")

# -----------------------------
# Step 8: Measure accuracy & inference time
# -----------------------------
correct = 0
total = 0
start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = pruned_model(images)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

end_time = time.time()

accuracy = (correct / total) * 100
inference_time = ((end_time - start_time)/total)*1000

# -----------------------------
# Step 9: Print results
# -----------------------------
print(f"Pruned Model Size: {model_size:.2f} MB")
print(f"Pruned Inference Time: {inference_time:.2f} ms/sample")
print(f"Pruned Accuracy: {accuracy:.2f}%")
