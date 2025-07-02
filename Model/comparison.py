import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- CONFIG ---
BASE_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
MODEL_DIRS = {
    'ARIMA': os.path.join(BASE_DIR, 'arima'),
    'SARIMA': os.path.join(BASE_DIR, 'sarima'),
    'LSTM': os.path.join(BASE_DIR, 'lstm_upgraded'),
    'GRU': os.path.join(BASE_DIR, 'gru_model'),
    'XGBoost': os.path.join(BASE_DIR, 'xgboost')
}
TEST_DATA_DIR = os.path.join(BASE_DIR, 'splits')
PLOT_DIR = os.path.join(BASE_DIR, 'model_comparison/plots')
os.makedirs(PLOT_DIR, exist_ok=True)

DISEASES = ['Giardiasis', 'chickenpox']

# --- Helper Function for MAPE ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

# --- Load Forecast and Metrics ---
def load_forecast_and_metrics(disease):
    forecasts = {}
    metrics = {}
    test_data = pd.read_csv(os.path.join(TEST_DATA_DIR, f"{disease}_test.csv"), index_col='Date', parse_dates=True)
    
    for model in MODEL_DIRS:
        # Load forecast
        if model == 'ARIMA':
            forecast_file = os.path.join(MODEL_DIRS[model], f"{disease}_forecast_2021.csv")
            forecast_col = disease
        elif model == 'SARIMA':
            forecast_file = os.path.join(MODEL_DIRS[model], f"{disease}_sarima_forecast_extended.csv")
            forecast_col = 'Forecast'
        elif model == 'LSTM':
            forecast_file = os.path.join(MODEL_DIRS[model], f"{disease}_forecast_realistic.csv")
            forecast_col = 'Forecast'
        elif model == 'GRU':
            forecast_file = os.path.join(MODEL_DIRS[model], f"{disease}_gru_forecast_future.csv")
            forecast_col = 'Forecast'
        elif model == 'XGBoost':
            forecast_file = os.path.join(MODEL_DIRS[model], f"{disease}_xgboost_predictions.csv")
            forecast_col = 'XGB_Pred'
        
        try:
            forecast_df = pd.read_csv(forecast_file, index_col='Date', parse_dates=True)
            # Align forecasts to test data index
            forecasts[model] = forecast_df[forecast_col].reindex(test_data.index, method='nearest').fillna(0)
        except Exception as e:
            print(f"Error loading forecast for {model} - {disease}: {e}")
            forecasts[model] = pd.Series(np.nan, index=test_data.index)
        
        # Load or compute metrics
        if model == 'ARIMA':
            metrics_file = os.path.join(MODEL_DIRS[model], f"{disease}_evaluation_metrics.csv")
            try:
                metrics_df = pd.read_csv(metrics_file)
                metrics[model] = {
                    'RMSE': metrics_df['RMSE'].iloc[0],
                    'MAE': metrics_df['MAE'].iloc[0],
                    'MAPE': mean_absolute_percentage_error(test_data[disease], forecasts[model])
                }
            except Exception as e:
                print(f"Error loading metrics for {model} - {disease}: {e}")
                metrics[model] = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        elif model == 'XGBoost':
            metrics_file = os.path.join(MODEL_DIRS[model], f"{disease}_xgboost_metrics.csv")
            try:
                metrics_df = pd.read_csv(metrics_file)
                metrics[model] = {
                    'RMSE': metrics_df['RMSE'].iloc[0],
                    'MAE': metrics_df['MAE'].iloc[0],
                    'MAPE': metrics_df['MAPE'].iloc[0]
                }
            except Exception as e:
                print(f"Error loading metrics for {model} - {disease}: {e}")
                metrics[model] = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        else:
            # Compute metrics for SARIMA, LSTM, GRU
            try:
                rmse = np.sqrt(mean_squared_error(test_data[disease], forecasts[model]))
                mae = mean_absolute_error(test_data[disease], forecasts[model])
                mape = mean_absolute_percentage_error(test_data[disease], forecasts[model])
                metrics[model] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
            except Exception as e:
                print(f"Error computing metrics for {model} - {disease}: {e}")
                metrics[model] = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
    
    return test_data[disease], forecasts, metrics

# --- Plot Comparison ---
def plot_model_comparison(disease, actual, forecasts, metrics):
    # Line Plot: Actual vs Predicted
    plt.figure(figsize=(14, 8))
    plt.plot(actual.index, actual, label='Actual Test Data', color='black', linewidth=2)
    colors = {'ARIMA': 'blue', 'SARIMA': 'green', 'LSTM': 'orange', 'GRU': 'red', 'XGBoost': 'purple'}
    for model, forecast in forecasts.items():
        plt.plot(forecast.index, forecast, label=f'{model} Forecast', linestyle='--', color=colors[model])
    plt.title(f'Model Forecast Comparison for {disease} - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_forecast_comparison.png"))
    plt.close()

    # Bar Plot: Metrics Comparison
    metrics_df = pd.DataFrame(metrics).T
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    metrics_df[['RMSE']].plot(kind='bar', ax=ax[0], color='skyblue')
    ax[0].set_title(f'RMSE Comparison - {disease}')
    ax[0].set_ylabel('RMSE')
    metrics_df[['MAE']].plot(kind='bar', ax=ax[1], color='lightgreen')
    ax[1].set_title(f'MAE Comparison - {disease}')
    ax[1].set_ylabel('MAE')
    metrics_df[['MAPE']].plot(kind='bar', ax=ax[2], color='salmon')
    ax[2].set_title(f'MAPE Comparison - {disease}')
    ax[2].set_ylabel('MAPE (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_metrics_comparison.png"))
    plt.close()

    # Residual Plot
    plt.figure(figsize=(14, 8))
    for model, forecast in forecasts.items():
        residuals = actual - forecast
        plt.plot(actual.index, residuals, label=f'{model} Residuals', linestyle='-', alpha=0.6, color=colors[model])
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Residuals Comparison for {disease}')
    plt.xlabel('Date')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_residuals_comparison.png"))
    plt.close()

# --- Summary Table ---
def save_summary_table(disease, metrics):
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(PLOT_DIR, f"{disease}_metrics_summary.csv"))
    print(f"\nMetrics Summary for {disease}:")
    print(metrics_df)

# --- Run Comparison ---
for disease in DISEASES:
    print(f"\nProcessing {disease}...")
    actual, forecasts, metrics = load_forecast_and_metrics(disease)
    plot_model_comparison(disease, actual, forecasts, metrics)
    save_summary_table(disease, metrics)

print(f"\n✅ Model comparison completed. Plots and summary tables saved in {PLOT_DIR}")