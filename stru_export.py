import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import time
import os

# -----------------------------
# Step 1: Define original CNN
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, conv1_out=32, conv2_out=64):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(conv2_out * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
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
original_model = SmallCNN()
original_model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
original_model.to(device)
original_model.eval()
print("✅ Original model loaded successfully")

# -----------------------------
# Step 4: Apply structured pruning
# -----------------------------
# Prune 30% of channels in conv layers
prune.ln_structured(original_model.conv1, name="weight", amount=0.3, n=2, dim=0)
prune.ln_structured(original_model.conv2, name="weight", amount=0.3, n=2, dim=0)
print("✅ Structured pruning applied (30% channels pruned)")

# Remove pruning reparametrization to export actual smaller model
prune.remove(original_model.conv1, 'weight')
prune.remove(original_model.conv2, 'weight')

# -----------------------------
# Step 5: Export a smaller CNN reflecting pruning
# -----------------------------
# Count remaining channels
remaining_conv1 = int(original_model.conv1.out_channels)
remaining_conv2 = int(original_model.conv2.out_channels)

pruned_model = SmallCNN(conv1_out=remaining_conv1, conv2_out=remaining_conv2)

# Copy pruned weights (channels that remain)
pruned_model.conv1.weight.data = original_model.conv1.weight.data.clone()
pruned_model.conv2.weight.data = original_model.conv2.weight.data.clone()
pruned_model.fc1.weight.data = original_model.fc1.weight.data.clone()
pruned_model.fc2.weight.data = original_model.fc2.weight.data.clone()
pruned_model.fc3.weight.data = original_model.fc3.weight.data.clone()
pruned_model.to(device)
pruned_model.eval()

# -----------------------------
# Step 6: Measure metrics
# -----------------------------
# Model size
torch.save(pruned_model.state_dict(), "temp_pruned_model.pt")
model_size = os.path.getsize("temp_pruned_model.pt") / (1024*1024)
os.remove("temp_pruned_model.pt")

# Accuracy & Inference Time
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
accuracy = (correct/total)*100
inference_time = ((end_time-start_time)/total)*1000  # ms per sample

# -----------------------------
# Step 7: Print metrics
# -----------------------------
print(f"Pruned Model Size: {model_size:.2f} MB")
print(f"Pruned Inference Time: {inference_time:.2f} ms/sample")
print(f"Pruned Accuracy: {accuracy:.2f}%")
