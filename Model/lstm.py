import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import shutil
from datetime import timedelta

# Paths
base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
input_dir = f"{base_dir}/splits"
stationarity_dir = f"{base_dir}/stationarity"
output_dir = f"{base_dir}/lstm"
plot_dir = f"{base_dir}/forecasts"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Diseases and sequence length (weeks)
diseases = {'Giardiasis': 4, 'chickenpox': 4}
forecast_horizon = 104  # Forecast 104 weeks (2 years)

# Clean up old files in output_dir and plot_dir
for folder in [output_dir, plot_dir]:
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path) and item not in diseases:
            shutil.rmtree(item_path)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Updated LSTM Model (simplified architecture with single FC layer, dropout=0.3)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

# Prepare sequences
def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Reverse differencing
def reverse_differencing(history, predictions, d):
    if d == 0:
        return predictions
    result = [history[-d] + predictions[0]]
    for i in range(1, len(predictions)):
        result.append(result[i-1] + predictions[i])
    return np.array(result)

# MAPE function to handle zero values
def mean_absolute_percentage_error(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    non_zero_mask = actual != 0
    if non_zero_mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100

# Process each disease
for disease, seq_length in diseases.items():
    # Create subfolders for each disease
    disease_dir = os.path.join(output_dir, disease)
    disease_plot_dir = os.path.join(plot_dir, disease)
    os.makedirs(disease_dir, exist_ok=True)
    os.makedirs(disease_plot_dir, exist_ok=True)

    # Load data
    train_path = f"{input_dir}/{disease}_train.csv"
    test_path = f"{input_dir}/{disease}_test.csv"
    stationarity_path = f"{stationarity_dir}/{disease}_stationarity_results.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Error: {disease} train/test files missing.")
        continue
    
    train = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    test = pd.read_csv(test_path, index_col='Date', parse_dates=True)
    
    try:
        # Check data
        if train[disease].isna().any() or test[disease].isna().any():
            raise ValueError("Missing values found")
        if (train[disease] < 0).any() or (test[disease] < 0).any():
            raise ValueError("Negative values found")
        
        # Get differencing order
        d = 1 if not os.path.exists(stationarity_path) else pd.read_csv(stationarity_path)['Recommended_d'].iloc[0]
        if os.path.exists(stationarity_path):
            p_value = pd.read_csv(stationarity_path).loc[pd.read_csv(stationarity_path)['Test'] == 'Original', 'p_value'].iloc[0]
            if p_value > 0.05 and d == 0:
                print(f"{disease} not stationary (p={p_value:.4f}), using d=1")
                d = 1
        
        # Differencing
        train_data = train[disease].copy()
        test_data = test[disease].copy()
        if d > 0:
            train_data = train_data.diff(d).dropna()
            test_data = test_data.diff(d).dropna()
        
        # Scale
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
        test_scaled = scaler.transform(test_data.values.reshape(-1, 1))
        
        # Sequences
        X_train, y_train = prepare_sequences(train_scaled, seq_length)
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Model
        model = LSTMModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Train
        epochs = 200
        best_loss = float('inf')
        patience = 15
        counter = 0
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            scheduler.step(train_loss)
            print(f"{disease} Epoch {epoch+1}, Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if train_loss < best_loss:
                best_loss = train_loss
                counter = 0
                torch.save(model.state_dict(), f"{disease_dir}/lstm_best.pth")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f"{disease_dir}/lstm_best.pth"))
        
        # Forecasting (test set + future)
        model.eval()
        test_predictions = []
        future_predictions = []
        input_seq = train_scaled[-seq_length:].copy()
        input_seq = torch.tensor(input_seq, dtype=torch.float32).reshape(1, seq_length, 1).to(device)
        
        # Test set predictions
        with torch.no_grad():
            for _ in range(len(test)):
                pred = model(input_seq)
                test_predictions.append(pred.cpu().numpy()[0, 0])
                input_seq = torch.cat((input_seq[:, 1:, :], pred.reshape(1, 1, 1)), dim=1)
        
        # Future predictions
        with torch.no_grad():
            for _ in range(forecast_horizon):
                pred = model(input_seq)
                future_predictions.append(pred.cpu().numpy()[0, 0])
                input_seq = torch.cat((input_seq[:, 1:, :], pred.reshape(1, 1, 1)), dim=1)
        
        # Reverse scaling and differencing for test predictions
        test_preds = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
        if d > 0:
            test_preds = reverse_differencing(train[disease].values[-d:], test_preds, d)
        test_preds = np.maximum(test_preds, 0)
        
        # Reverse scaling and differencing for future predictions
        future_preds = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        if d > 0:
            future_preds = reverse_differencing(np.concatenate([train[disease].values, test[disease].values])[-d:], 
                                              future_preds, d)
        future_preds = np.maximum(future_preds, 0)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(test[disease].values[:len(test_preds)], test_preds))
        mae = mean_absolute_error(test[disease].values[:len(test_preds)], test_preds)
        mape = mean_absolute_percentage_error(test[disease].values[:len(test_preds)], test_preds)
        
        # Create date index for future predictions
        last_date = test.index[-1]
        future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_horizon+1)]
        
        # Save results
        with open(f"{disease_dir}/lstm_results.txt", "w") as f:
            f.write(f"LSTM RMSE: {rmse:.2f}\n")
            f.write(f"LSTM MAE: {mae:.2f}\n")
            f.write(f"LSTM MAPE: {mape:.2f}%\n")
            f.write(f"Differencing Order (d): {d}\n")
        
        # Save forecasts
        forecast_df = pd.DataFrame({
            'Date': list(test.index[:len(test_preds)]) + future_dates,
            'Forecast': np.concatenate([test_preds, future_preds])
        })
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        forecast_df.to_csv(f"{disease_dir}/lstm_forecast.csv", index=False)
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(train[disease][-52:], label='Train (last year)')
        plt.plot(test[disease], label='Actual')
        plt.plot(test.index[:len(test_preds)], test_preds, label='Test Forecast')
        plt.plot(future_dates, future_preds, label='Future Forecast', linestyle='--')
        plt.title(f'{disease} LSTM Forecast')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{disease_plot_dir}/lstm_forecast_plot.png")
        plt.close()
        
        print(f"{disease} LSTM RMSE: {rmse:.2f}, d={d}")
    
    except Exception as e:
        print(f"Error for {disease}: {e}")

print("✅ LSTM modeling completed.")
print(f"Saving forecast to: {output_dir}")