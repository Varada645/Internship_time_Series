# GRU Forecasting Model for Disease Time Series

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import math

# --- CONFIG ---
DATA_PATH = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/selected_diseases_time_series_with_seasonality.csv"
OUTPUT_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/gru_model"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
SCALER_DIR = os.path.join(OUTPUT_DIR, "scalers")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# --- Seasonality Features ---
def add_seasonality(df):
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df

# --- Lag Feature Engineering ---
def add_lag_features(df, target_col, lags=[1,2,3,4,5,6]):
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

# --- GRU Model ---
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# --- Create Sequences ---
def create_sequences(data, target_col, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i][target_col])
    return np.array(X), np.array(y)

# --- Forecast Function ---
def forecast_with_gru(disease, seq_len=12, epochs=100, hidden_size=128, learning_rate=0.001):
    print(f"\n📈 GRU Forecasting for {disease}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = add_seasonality(df)
    df = add_lag_features(df, disease)

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.drop(columns=['Date']))
    test_scaled = scaler.transform(test.drop(columns=['Date']))
    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{disease}_scaler.pkl"))

    feature_cols = train.drop(columns=['Date']).columns.tolist()
    target_idx = feature_cols.index(disease)
    X_train, y_train = create_sequences(train_scaled, target_idx, seq_len)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

    model = GRUNet(input_size=X_train.shape[2], hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    model.eval()
    test_true = test[disease].values
    history = list(train_scaled[-seq_len:])
    test_pred = []
    with torch.no_grad():
        for i in range(len(test_scaled)):
            inp = torch.tensor(np.array(history[-seq_len:])[np.newaxis], dtype=torch.float32).to(device)
            pred = model(inp).item()
            pred = max(0, pred)
            new_feat = test_scaled[i].copy()
            new_feat[target_idx] = pred
            history.append(new_feat)
            test_pred.append(pred)

    test_pred = scaler.inverse_transform(
        np.column_stack([test_pred] + [test_scaled[:, i] for i in range(1, test_scaled.shape[1])])
    )[:, 0]

    rmse = math.sqrt(mean_squared_error(test_true, test_pred))
    mae = mean_absolute_error(test_true, test_pred)
    mape = mean_absolute_percentage_error(test_true, test_pred)
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(test['Date'], test_true, label='Actual Test', color='blue')
    plt.plot(test['Date'], test_pred, label='GRU Prediction (Test)', linestyle=':', color='orange')
    plt.title(f'{disease} GRU Forecast (Test Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_gru_forecast.png"))
    plt.close()

    print(f"✅ Forecast plot saved for {disease}")

    # --- Forecast future until end of 2022 ---
    last_date = pd.to_datetime(test['Date'].iloc[-1])
    end_date = pd.to_datetime("2022-12-26")  # Last Monday of 2022
    future_weeks = max(1, ((end_date - last_date).days) // 7)
    future_forecasts = []
    future_dates = []
    week = int(test['Week'].iloc[-1])
    month = int(test['Month'].iloc[-1])
    history_future = history.copy()

    with torch.no_grad():
        for _ in range(future_weeks):
            inp = torch.tensor(np.array(history_future[-seq_len:])[np.newaxis], dtype=torch.float32).to(device)
            pred = model(inp).item()
            pred = max(0, pred)
            new_feat = history_future[-1].copy()
            new_feat[target_idx] = pred
            # Update seasonality features
            week = (week % 52) + 1
            month = (month % 12) + 1
            new_feat[-4] = np.sin(2 * np.pi * week / 52)
            new_feat[-3] = np.cos(2 * np.pi * week / 52)
            new_feat[-2] = np.sin(2 * np.pi * month / 12)
            new_feat[-1] = np.cos(2 * np.pi * month / 12)
            history_future.append(new_feat)
            future_forecasts.append(new_feat.copy())
            future_dates.append(last_date + pd.Timedelta(weeks=len(future_dates)+1))

    future_forecasts_arr = np.array(future_forecasts)
    future_forecast_values = scaler.inverse_transform(future_forecasts_arr)[:, target_idx]

    # --- Plot with future forecast ---
    plt.figure(figsize=(12, 6))
    plt.plot(test['Date'], test_true, label='Actual Test', color='blue')
    plt.plot(test['Date'], test_pred, label='GRU Prediction (Test)', linestyle=':', color='orange')
    plt.plot(future_dates, future_forecast_values, label='Forecast (Future)', linestyle='--', color='red')
    plt.title(f'{disease} GRU Forecast (Test + Future)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_gru_forecast_future.png"))
    plt.close()

    # Save future forecast to CSV
    pd.DataFrame({"Date": future_dates, "Forecast": future_forecast_values}).to_csv(
        os.path.join(OUTPUT_DIR, f"{disease}_gru_forecast_future.csv"), index=False
    )

    print(f"✅ Future forecast (till 2022) plot and CSV saved for {disease}")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

# --- Run Forecasts ---
forecast_with_gru("chickenpox")
forecast_with_gru("Giardiasis")