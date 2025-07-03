# LSTM Pipeline Upgrade for Disease Forecasting

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
OUTPUT_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/lstm_upgraded"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
SCALER_DIR = os.path.join(OUTPUT_DIR, "scalers")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

def add_seasonality(df):
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df

def add_lag_features(df, target_col, lags=[1,2,3,4,5,6]):
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    df = df.fillna(0)
    return df

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def create_sequences(df, target_col, seq_len):
    df = df.drop(columns=['Date'])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i][df.columns.get_loc(target_col)])
    return np.array(X), np.array(y), scaler

def forecast_disease(disease, seq_len=12, epochs=100, hidden_size=128, learning_rate=0.001):
    print(f"\n🚀 Forecasting {disease} using LSTM pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = add_seasonality(df)
    df = add_lag_features(df, disease)

    # Split: last 20% as test
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)

    # Fit scaler only on train, transform both
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.drop(columns=['Date']))
    test_scaled = scaler.transform(test.drop(columns=['Date']))
    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{disease}_scaler.pkl"))

    # Prepare sequences for training (from train only)
    X_train, y_train = [], []
    train_cols = train.drop(columns=['Date']).columns
    for i in range(seq_len, len(train_scaled)):
        X_train.append(train_scaled[i-seq_len:i])
        y_train.append(train_scaled[i][train_cols.get_loc(disease)])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

    # Model
    model = LSTM(input_size=X_train.shape[2], hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{disease}_best_model.pt"))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Forecast: start from end of test set until end of 2021
    model.eval()
    last_date = pd.to_datetime(test['Date'].iloc[-1])
    end_date = pd.to_datetime("2020-12-28")  # Last Monday of 2020
  
    future_weeks = max(1, ((end_date - last_date).days) // 7)
    history = list(test_scaled[-seq_len:])
    forecasts = []
    week = int(test['Week'].iloc[-1])
    month = int(test['Month'].iloc[-1])

    with torch.no_grad():
        for _ in range(future_weeks):
            inp = torch.tensor(np.array(history[-seq_len:])[np.newaxis], dtype=torch.float32).to(device)
            pred = model(inp).item()
            pred = max(0, pred)
            new_feat = history[-1].copy()
            new_feat[0] = pred  # update target
            week = (week % 52) + 1
            month = (month % 12) + 1
            new_feat[-4] = np.sin(2 * np.pi * week / 52)
            new_feat[-3] = np.cos(2 * np.pi * week / 52)
            new_feat[-2] = np.sin(2 * np.pi * month / 12)
            new_feat[-1] = np.cos(2 * np.pi * month / 12)
            history.append(new_feat)
            forecasts.append(new_feat.copy())

    forecasts_arr = np.array(forecasts)
    forecast_values = scaler.inverse_transform(forecasts_arr)[:, 0]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=future_weeks, freq='W-MON')

    # Evaluate on test set
    test_true = test[disease].values
    test_pred = []
    history_eval = list(train_scaled[-seq_len:])
    for i in range(len(test_scaled)):
        inp = torch.tensor(np.array(history_eval[-seq_len:])[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(inp).item()
        pred = max(0, pred)
        new_feat = test_scaled[i].copy()
        new_feat[0] = pred
        history_eval.append(new_feat)
        test_pred.append(pred)
    test_pred = scaler.inverse_transform(np.column_stack([test_pred] + [test_scaled[:, i] for i in range(1, test_scaled.shape[1])]))[:, 0]

    rmse = math.sqrt(mean_squared_error(test_true, test_pred))
    mae = mean_absolute_error(test_true, test_pred)
    mape = mean_absolute_percentage_error(test_true, test_pred)
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

    # Plot: only test and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(test['Date'], test_true, label='Actual Test', color='blue')
    plt.plot(test['Date'], test_pred, label='LSTM Prediction (Test)', linestyle=':', color='orange')
    plt.plot(future_dates, forecast_values, label='Forecast (Future)', linestyle='--', color='red')
    plt.title(f'{disease} LSTM Forecast (Test + Future)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_forecast.png"))

    # Residuals plot
    residuals = test_true - test_pred
    plt.figure(figsize=(12, 4))
    plt.plot(test['Date'], residuals, label='Residuals (Test - Forecast)', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Residuals of {disease} Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_forecast_residuals_realistic.png"))
    plt.close()

    pd.DataFrame({"Date": future_dates, "Forecast": forecast_values}).to_csv(
        os.path.join(OUTPUT_DIR, f"{disease}_forecast.csv"), index=False
    )
    print(f"✅ Forecast completed and saved for {disease}.")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

# --- Run Forecasts ---
forecast_disease("chickenpox")
forecast_disease("Giardiasis")

