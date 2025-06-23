import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import shutil
from datetime import timedelta

# Paths
base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
input_dir = f"{base_dir}/splits"
stationarity_dir = f"{base_dir}/stationarity"
output_dir = f"{base_dir}/sarima"
plot_dir = f"{base_dir}/forecasts"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Diseases
diseases = ['Giardiasis', 'chickenpox']
forecast_horizon = 104  # Forecast 104 weeks (2 years)

# Clean up old files in output_dir and plot_dir
for folder in [output_dir, plot_dir]:
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path) and item not in diseases:
            shutil.rmtree(item_path)

# MAPE function to handle zero values
def mean_absolute_percentage_error(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    non_zero_mask = actual != 0
    if non_zero_mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100

# Process each disease
for disease in diseases:
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
        print(f"Error: {disease} train/test files missing. Skipping.")
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
        
        # Log transform
        train_data = np.log1p(train[disease])
        test_data = np.log1p(test[disease])
        
        # Differencing
        if d > 0:
            train_data = train_data.diff(d).dropna()
        
        # SARIMA model
        order = (1, d, 1)
        seasonal_order = (1, 1, 1, 52)
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        fit = model.fit(disp=False)
        
        # Forecast (test + future)
        forecast_steps = len(test) + forecast_horizon
        sarima_forecast = fit.forecast(steps=forecast_steps)
        
        # Reverse transformations
        if d > 0:
            last_values = train_data.iloc[-d:].values
            sarima_forecast = np.cumsum(np.concatenate([last_values, sarima_forecast]))[d:]
        sarima_forecast = np.expm1(sarima_forecast)
        sarima_forecast = np.maximum(sarima_forecast, 0)
        
        # RMSE for test set
        sarima_test_forecast = sarima_forecast[:len(test)]
        sarima_rmse = np.sqrt(mean_squared_error(test[disease].values, sarima_test_forecast))
        
        # Additional metrics: MAE and MAPE
        sarima_mae = mean_absolute_error(test[disease].values, sarima_test_forecast)
        sarima_mape = mean_absolute_percentage_error(test[disease].values, sarima_test_forecast)
        
        # Create date index for forecasts
        test_dates = test.index
        last_date = test.index[-1]
        future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_horizon+1)]
        all_dates = list(test_dates) + future_dates
        
        # Save results
        with open(f"{disease_dir}/sarima_results.txt", "w") as f:
            f.write(f"SARIMA RMSE: {sarima_rmse:.2f}\n")
            f.write(f"SARIMA MAE: {sarima_mae:.2f}\n")
            f.write(f"SARIMA MAPE: {sarima_mape:.2f}%\n")
            f.write(f"Differencing Order (d): {d}\n")
        
        # Save forecasts
        forecast_df = pd.DataFrame({
            'Date': all_dates,
            'Forecast': sarima_forecast
        })
        forecast_df.to_csv(f"{disease_dir}/sarima_forecast.csv", index=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(train[disease][-52:], label='Train (last year)')
        plt.plot(test[disease], label='Test')
        plt.plot(forecast_df.set_index('Date')['Forecast'][:len(test)], label='Test Forecast')
        plt.plot(forecast_df.set_index('Date')['Forecast'][len(test):], label='Future Forecast', linestyle='--')
        plt.title(f'{disease} SARIMA Forecast')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{disease_plot_dir}/sarima_forecast_plot.png")
        plt.close()
        
        print(f"{disease} SARIMA RMSE: {sarima_rmse:.2f}, d={d}")
    
    except Exception as e:
        print(f"Error for {disease}: {e}")

print("✅ SARIMA forecasts done.")
print(f"Saving forecast to: {output_dir}")