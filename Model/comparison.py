import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Paths
base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
input_dir = f"{base_dir}/splits"
plot_dir = f"{base_dir}/forecasts"
os.makedirs(plot_dir, exist_ok=True)

# Diseases and models
diseases = ['Giardiasis', 'chickenpox']
models = ['arima', 'sarima', 'gru', 'lstm']

# Initialize results dictionary
results = []

# Function to calculate MAPE (handle zero actuals)
def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0  # Avoid division by zero
    if not mask.any():
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Process each disease
for disease in diseases:
    # Load test data
    test_path = f"{input_dir}/{disease}_test.csv"
    if not os.path.exists(test_path):
        print(f"Error: {disease} test file missing at {test_path}. Skipping.")
        continue
    test = pd.read_csv(test_path, index_col='Date', parse_dates=True)
    actual = test[disease].values

    # Disease-specific plot
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, actual, label='Actual', linewidth=2, color='black')

    # Process each model
    for model in models:
        try:
            # Load results (RMSE, MAE, MAPE, d)
            result_file = f"{base_dir}/{model}/{disease}/{model}_results.txt"
            if not os.path.exists(result_file):
                print(f"Warning: {model} results file missing for {disease} at {result_file}.")
                continue
            with open(result_file, 'r') as f:
                lines = f.readlines()
                rmse = float(lines[0].split(': ')[1].split()[0])
                mae = float(lines[1].split(': ')[1].split()[0])
                mape = float(lines[2].split(': ')[1].split('%')[0])
                d = float(lines[3].split(': ')[1])

            # Load forecasts
            forecast_file = f"{base_dir}/{model}/{disease}/{model}_forecast.csv"
            if not os.path.exists(forecast_file):
                print(f"Warning: {model} forecast file missing for {disease} at {forecast_file}.")
                continue
            forecast_df = pd.read_csv(forecast_file)
            
            # Handle missing 'Date' column
            if 'Date' not in forecast_df.columns:
                print(f"Error: 'Date' column not found in {forecast_file}. Available columns: {list(forecast_df.columns)}")
                if 'Unnamed: 0' in forecast_df.columns:
                    forecast_df['Date'] = pd.to_datetime(forecast_df['Unnamed: 0'])
                    forecast_df = forecast_df.drop(columns=['Unnamed: 0'])
                elif len(forecast_df.columns) == 1:
                    print(f"Assuming single unnamed column as 'Forecast' for {model} in {disease}.")
                    forecast_df.columns = ['Forecast']
                    forecast_df['Date'] = pd.date_range(start=test.index[0], periods=len(forecast_df), freq='W-MON')
                else:
                    print(f"Cannot infer 'Date' column for {forecast_file}. Skipping {model} for {disease}.")
                    continue
            else:
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

            # Filter forecasts to match test period
            test_forecast = forecast_df[forecast_df['Date'].isin(test.index)]['Forecast'].values

            # Ensure forecast length matches actual
            if len(test_forecast) != len(actual):
                print(f"Warning: {model} forecast length ({len(test_forecast)}) does not match test data ({len(actual)}) for {disease}. Truncating.")
                test_forecast = test_forecast[:len(actual)]
                actual_truncated = actual[:len(test_forecast)]
            else:
                actual_truncated = actual

            # Verify metrics by recalculating
            calc_rmse = np.sqrt(mean_squared_error(actual_truncated, test_forecast))
            calc_mae = mean_absolute_error(actual_truncated, test_forecast)
            calc_mape = mean_absolute_percentage_error(actual_truncated, test_forecast)

            # Store results
            results.append({
                'Disease': disease,
                'Model': model.upper(),
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Differencing Order': d,
                'Calculated RMSE': calc_rmse,
                'Calculated MAE': calc_mae,
                'Calculated MAPE': calc_mape
            })

            # Plot forecasts
            plt.plot(forecast_df['Date'], forecast_df['Forecast'], 
                     label=f'{model.upper()} Forecast', 
                     linestyle='--' if model in ['lstm', 'gru'] else '-')

        except Exception as e:
            print(f"Error processing {model} for {disease}: {str(e)}")

    # Finalize plot
    plt.title(f'{disease} Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Incidence')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/{disease}/comparison_plot.png")
    plt.close()

# Create results DataFrame
results_df = pd.DataFrame(results)

# Pivot tables for each metric
rmse_pivot = results_df.pivot(index='Disease', columns='Model', values='RMSE').round(2)
mae_pivot = results_df.pivot(index='Disease', columns='Model', values='MAE').round(2)
mape_pivot = results_df.pivot(index='Disease', columns='Model', values='MAPE').round(2)
d_pivot = results_df.pivot(index='Disease', columns='Model', values='Differencing Order')
calc_rmse_pivot = results_df.pivot(index='Disease', columns='Model', values='Calculated RMSE').round(2)
calc_mae_pivot = results_df.pivot(index='Disease', columns='Model', values='Calculated MAE').round(2)
calc_mape_pivot = results_df.pivot(index='Disease', columns='Model', values='Calculated MAPE').round(2)

# Print comparison tables
print("\nRMSE Comparison (from results files):")
print(rmse_pivot)
print("\nMAE Comparison (from results files):")
print(mae_pivot)
print("\nMAPE Comparison (%, from results files):")
print(mape_pivot)
print("\nDifferencing Order (d):")
print(d_pivot)
print("\nCalculated RMSE Comparison:")
print(calc_rmse_pivot)
print("\nCalculated MAE Comparison:")
print(calc_mae_pivot)
print("\nCalculated MAPE Comparison (%):")
print(calc_mape_pivot)

# Save results to CSV
results_df.to_csv(f"{base_dir}/model_comparison_results.csv", index=False)
print(f"\nComparison results saved to: {base_dir}/model_comparison_results.csv")
print(f"Comparison plots saved to: {plot_dir}/Giardiasis/comparison_plot.png and {plot_dir}/chickenpox/comparison_plot.png")