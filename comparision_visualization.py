import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data from your experiments
# -----------------------------
models = ["Original", "Pruned", "Quantized"]

model_size = [12.46, 12.46, 3.17]          # MB
inference_time = [1.21, 1.14, 1.09]        # ms/sample
accuracy = [98.74, 98.63, 98.74]           # %

x = np.arange(len(models))
width = 0.6

# -----------------------------
# Plot Model Size
# -----------------------------
plt.figure()
plt.bar(x, model_size, width)
plt.xticks(x, models)
plt.ylabel("Model Size (MB)")
plt.title("Model Size Comparison")
plt.show()

# -----------------------------
# Plot Inference Time
# -----------------------------
plt.figure()
plt.bar(x, inference_time, width)
plt.xticks(x, models)
plt.ylabel("Inference Time (ms/sample)")
plt.title("Inference Time Comparison")
plt.show()

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure()
plt.bar(x, accuracy, width)
plt.xticks(x, models)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.ylim(95, 100)  # zoom for clarity
plt.show()
