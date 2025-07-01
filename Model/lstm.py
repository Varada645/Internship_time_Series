# LSTM Forecasting for Chickenpox & Giardiasis with Multivariate Features + Seasonality

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Paths
input_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/lstm_multivariate"
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Add seasonality features
def add_seasonality(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df

# Sequence creation
def create_sequences(df, target_col, seq_len):
    df = df.drop(columns=['Date'])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i][df.columns.get_loc(target_col)])
    return np.array(X), np.array(y), scaler

# LSTM Model
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

# Forecast function (generic for any disease)
def forecast_disease(disease, seq_len=12, future_weeks=52, epochs=100):
    print(f"\n🚀 Forecasting {disease} using multivariate LSTM")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = pd.read_csv(os.path.join(input_dir, f"{disease}_train.csv"))
    test = pd.read_csv(os.path.join(input_dir, f"{disease}_test.csv"))
    df = pd.concat([train, test], ignore_index=True)
    df = add_seasonality(df)

    # Limit forecasting only until end of 2024
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    end_date = pd.to_datetime("2024-12-30")
    remaining_weeks = ((end_date - last_date).days) // 7
    future_weeks = max(1, remaining_weeks)

    X, y, scaler = create_sequences(df.copy(), target_col=disease, seq_len=seq_len)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    model = LSTM(input_size=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Forecast
    model.eval()
    history = list(X[-1])  # last sequence
    forecasts = []
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    week = int(df['Week'].iloc[-1])
    month = int(df['Month'].iloc[-1])

    with torch.no_grad():
        for i in range(future_weeks):
            inp = torch.tensor(np.array(history[-seq_len:])[np.newaxis], dtype=torch.float32).to(device)
            pred = model(inp).item()
            pred = max(0, pred)
            # Update features for next step
            week = (week % 52) + 1
            month = (month % 12) + 1
            week_sin = np.sin(2 * np.pi * week / 52)
            week_cos = np.cos(2 * np.pi * week / 52)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            new_feat = history[-1].copy()
            new_feat[0] = pred
            # Update seasonality features
            new_feat[-4] = week_sin
            new_feat[-3] = week_cos
            new_feat[-2] = month_sin
            new_feat[-1] = month_cos
            history.append(new_feat)
            forecasts.append(pred)

    # Inverse transform
    pred_scaled = np.array(forecasts).reshape(-1, 1)
    zeros_padding = np.zeros((future_weeks, X.shape[2] - 1))
    inv_input = np.hstack([pred_scaled, zeros_padding])
    forecast_values = scaler.inverse_transform(inv_input)[:, 0]

    # Plot
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=future_weeks, freq='W-MON')

    full_df = pd.concat([train, test], ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    plt.figure(figsize=(12,6))
    plt.plot(full_df['Date'], full_df[disease], label='Historical (Train+Test)', color='blue')
    plt.plot(future_dates, forecast_values, label='Forecast (till 2024)', linestyle='--', color='red')
    plt.title(f'{disease} Forecast until 2024')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{disease}_forecast_lstm_multivariate.png"))
    plt.close()

    pd.DataFrame({"Date": future_dates, "Forecast": forecast_values}).to_csv(
        os.path.join(output_dir, f"{disease}_forecast_lstm_multivariate.csv"), index=False
    )
    print(f"✅ Forecast completed and saved for {disease}.")

# Forecast both diseases
forecast_disease("chickenpox")
forecast_disease("Giardiasis")
