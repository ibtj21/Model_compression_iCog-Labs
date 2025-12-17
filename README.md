```markdown
# Model_compression_iCog-Labs

## Project Description
This project explores model compression techniques using weight pruning and quantization on a trained neural network. The effects on model size, inference time, and accuracy are measured and compared to evaluate efficiency–performance trade-offs for real-world deployment.

## Repository Structure
```

data/
.gitignore
Accuracy.png
Inference_time.png
LICENSE
Model_size.png
README.md
comparision_visualization.py
metrics_baseline.py
metrics_pruning.py
metrics_quantization.py
mnist_cnn.pt
model_compression_report.pdf
train.py

```

## Features
- Trains and saves a model.
- Prints three metrics (model size, inference time, accuracy) for the baseline model.
- Applies pruning and prints the same metrics for the pruned model.
- Applies quantization and prints the same metrics for the quantized model.
- Draws an interactive bar chart to compare the three metrics across the original, pruned, and quantized models.

## Dataset / Model
The MNIST dataset was used for this project, consisting of 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing), each of size 28×28 pixels. A small convolutional neural network (CNN) was trained on this dataset, with two convolutional layers followed by three fully connected layers. This setup allows the effects of pruning and quantization to be observed while maintaining efficient experimentation.

## Requirements
```

torch==2.5.1
torchvision==0.16.1
numpy==1.26.0
matplotlib==3.8.1
plotly==6.1.0

```

## How to Run
1. Clone the repository.
2. Train the model by running `train.py`.
3. Run `metrics_baseline.py` to view the metrics of the original model.
4. Run `metrics_pruning.py` to view the metrics of the pruned model.
5. Run `metrics_quantization.py` to view the metrics of the quantized model.
6. Run `comparision_visualization.py` to visualize a model-wise comparison across all metrics.

## License
MIT License
```

