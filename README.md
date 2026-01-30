# advanced_time_series_forecasting.py
Advanced Time Series Forecasting with Deep Learning and Explainability
# =========================================================
# Advanced Time Series Forecasting with Deep Learning
# Model: LSTM
# Explainability: Integrated Gradients
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# -----------------------------
# 1. Synthetic Data Generation
# -----------------------------
np.random.seed(42)

time_steps = 1200
t = np.arange(time_steps)

trend = 0.01 * t
seasonality_1 = 5 * np.sin(2 * np.pi * t / 24)
seasonality_2 = 2 * np.sin(2 * np.pi * t / 168)

feature_1 = trend + seasonality_1
feature_2 = seasonality_2
feature_3 = np.random.normal(0, 0.5, time_steps)
feature_4 = 0.5 * trend + np.random.normal(0, 0.2, time_steps)
feature_5 = np.sin(2 * np.pi * t / 365)

target = feature_1 + feature_2 + feature_3

data = np.vstack([
    feature_1,
    feature_2,
    feature_3,
    feature_4,
    feature_5,
    target
]).T

df = pd.DataFrame(
    data,
    columns=["f1", "f2", "f3", "f4", "f5", "target"]
)

# -----------------------------
# 2. Dataset Preparation
# -----------------------------
SEQ_LEN = 30

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len, :-1]
        y = self.data[idx+self.seq_len, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

dataset = TimeSeriesDataset(df.values, SEQ_LEN)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# -----------------------------
# 3. LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(input_size=5, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4. Training Loop
# -----------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(x_batch).squeeze()
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

# -----------------------------
# 5. Evaluation
# -----------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch).squeeze()
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.numpy())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")

# -----------------------------
# 6. Integrated Gradients
# -----------------------------
def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for scaled in scaled_inputs:
        scaled.requires_grad = True
        output = model(scaled)
        output.backward(torch.ones_like(output))
        grads.append(scaled.grad.detach().numpy())

    avg_grads = np.mean(grads, axis=0)
    integrated_grads = (input_tensor.detach().numpy() - baseline.detach().numpy()) * avg_grads
    return integrated_grads

sample_x, _ = dataset[100]
ig = integrated_gradients(model, sample_x.unsqueeze(0))

feature_importance = ig.mean(axis=(0,1))

for i, val in enumerate(feature_importance):
    print(f"Feature {i+1} importance: {val:.4f}")
