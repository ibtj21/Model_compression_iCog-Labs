import plotly.graph_objects as go

# -----------------------------
# Metrics (your real results)
# -----------------------------
models = ["Original", "Pruned", "Quantized"]

model_size = [12.46, 12.46, 3.17]      # MB
inference_time = [1.21, 1.14, 1.09]    # ms/sample
accuracy = [98.74, 98.63, 98.74]       # %

# -----------------------------
# Create figure
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Bar(
    name="Model Size (MB)",
    x=models,
    y=model_size,
    hovertemplate="Model: %{x}<br>Size: %{y} MB<extra></extra>"
))

fig.add_trace(go.Bar(
    name="Inference Time (ms/sample)",
    x=models,
    y=inference_time,
    hovertemplate="Model: %{x}<br>Time: %{y} ms<extra></extra>"
))

fig.add_trace(go.Bar(
    name="Accuracy (%)",
    x=models,
    y=accuracy,
    hovertemplate="Model: %{x}<br>Accuracy: %{y}%<extra></extra>"
))

# -----------------------------
# Layout
# -----------------------------
fig.update_layout(
    title="Model Compression Comparison",
    xaxis_title="Model Version",
    yaxis_title="Metric Value",
    barmode="group",
    template="plotly_white",
    legend_title="Metrics"
)

# -----------------------------
# Show plot
# -----------------------------
fig.show()
