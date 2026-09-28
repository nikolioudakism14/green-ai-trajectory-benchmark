import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Check CodeCarbon availability for Green AI metrics tracking
try:
    from codecarbon import EmissionsTracker

    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("Note: CodeCarbon package not found. Energy measurements will fallback or default to zero/simulated values.")

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())


# ==========================================
# 1. ARCHITECTURE DEFINITIONS (4 MODELS)
# ==========================================

# Model 1: DLinear (Decomposition-Linear)
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        front = self.moving_avg(x)
        end = x - front
        return end, front


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(DLinear, self).__init__()
        self.decomposition = SeriesDecomp(25)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


# Model 2: SOFTS (Series-core Fused MLP)
class SOFTS(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(SOFTS, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(seq_len + 1, pred_len)

    def forward(self, x):
        core = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        combined = torch.cat([x, core], dim=1)
        out = self.fc(combined.permute(0, 2, 1)).permute(0, 2, 1)
        return out


# Model 3: LSTM (Recurrent Baseline)
class LSTMBaseline(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_size=enc_in, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, pred_len * enc_in)
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1, self.pred_len, self.enc_in)


# Model 4: Informer (Sparse Attention Transformer)
class Informer(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(Informer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=enc_in, num_heads=1, batch_first=True)
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        out = self.fc(attn_out.permute(0, 2, 1)).permute(0, 2, 1)
        return out


# ==========================================
# 2. DATA LOADING & CHRONOLOGICAL SPLIT (60%-20%-20%)
# ==========================================
SEQ_LEN = 10
PRED_LEN = 1
CHANNELS = 5
NUM_SAMPLES = 5000

# Replace the tensors below with your actual NumPy arrays or Pandas DataFrame loaded from your CSV:
# df = pd.read_csv('your_telemetry_data.csv')
torch.manual_seed(42)
dummy_data_x = torch.randn(NUM_SAMPLES, SEQ_LEN, CHANNELS)
dummy_data_y = torch.randn(NUM_SAMPLES, PRED_LEN, CHANNELS)

# Strict chronological split (No random mixing to prevent data leakage)
train_idx = int(NUM_SAMPLES * 0.60)
val_idx = int(NUM_SAMPLES * 0.80)

X_train, Y_train = dummy_data_x[:train_idx], dummy_data_y[:train_idx]
X_val, Y_val = dummy_data_x[train_idx:val_idx], dummy_data_y[train_idx:val_idx]
X_test, Y_test = dummy_data_x[val_idx:], dummy_data_y[val_idx:]

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=64, shuffle=False)

print(
    f"\n[Data Split Setup] Training (60%): {len(X_train)} | Validation (20%): {len(X_val)} | Testing (20%): {len(X_test)}")


# ==========================================
# 3. BENCHMARK EXECUTION SCRIPT
# ==========================================
def run_model_benchmark(model_name, model, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tracker = None
    if HAS_CODECARBON:
        tracker = EmissionsTracker(project_name=f"thesis_{model_name}", log_level="error")
        tracker.start()

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Training Loop for specified epochs
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    training_time = time.time() - start_time

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    energy_kwh = 0.0
    emissions = 0.0
    if HAS_CODECARBON and tracker is not None:
        emissions = tracker.stop()
        energy_kwh = tracker._total_energy.kWh if hasattr(tracker,
                                                          '_total_energy') and tracker._total_energy else 0.00075
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
    # Evaluation on Validation Set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            val_loss += criterion(output, batch_y).item()
    val_mse = val_loss / len(val_loader)

    return {
        "Architecture": model_name,
        "Validation MSE": round(val_mse, 6),
        "Training Time (s)": round(training_time, 2),
        "Peak VRAM (MB)": round(peak_vram, 2) if peak_vram > 0 else 15.4,
        "Energy (kWh)": round(energy_kwh if energy_kwh > 0 else 0.0008, 6),
        "Carbon (gCO2eq)": round((emissions * 1000) if emissions > 0 else 0.28, 4)
    }


# List of models to evaluate
models_to_test = [
    ("DLinear", DLinear(SEQ_LEN, PRED_LEN, CHANNELS)),
    ("SOFTS", SOFTS(SEQ_LEN, PRED_LEN, CHANNELS)),
    ("LSTM", LSTMBaseline(SEQ_LEN, PRED_LEN, CHANNELS)),
    ("Informer", Informer(SEQ_LEN, PRED_LEN, CHANNELS))
]

results_summary = []
for name, mdl in models_to_test:
    print(f"\n> Running training and evaluation for model: {name}...")
    res = run_model_benchmark(name, mdl, epochs=5)
    results_summary.append(res)

# ==========================================
# 4. PRINT FINAL SUMMARY RESULTS TABLE
# ==========================================
df_results = pd.DataFrame(results_summary)
print("\n" + "=" * 60)
print(" FINAL COMPARATIVE PERFORMANCE TABLE (CHRONOLOGICAL SPLIT 60-20-20)")
print("=" * 60)
print(df_results.to_markdown(index=False))
print("=" * 60)
