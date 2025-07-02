import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/arima"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/correlation"
os.makedirs(output_dir, exist_ok=True)

# Load ARIMA forecast data
giardiasis = pd.read_csv(f"{base_dir}/Giardiasis_forecast_2021.csv", usecols=["Date", "Giardiasis"], parse_dates=["Date"])
chickenpox = pd.read_csv(f"{base_dir}/chickenpox_forecast_2021.csv", usecols=["Date", "chickenpox"], parse_dates=["Date"])

# Merge on Date
df = pd.merge(giardiasis, chickenpox, on="Date").set_index("Date")

# Compute correlations
results = []
for data_type, series in [
    ('Raw', df),
    ('Differenced', df.diff().dropna())
]:
    pearson_corr, pearson_p = pearsonr(series['Giardiasis'], series['chickenpox'])
    spearman_corr, spearman_p = spearmanr(series['Giardiasis'], series['chickenpox'])
    results.append({
        'Data': data_type,
        'Pearson Correlation': pearson_corr,
        'Pearson p-value': pearson_p,
        'Spearman Correlation': spearman_corr,
        'Spearman p-value': spearman_p
    })

# Cross-correlation
max_lags = 52  # 1 year of weekly data
cross_corr = []
lags = range(-max_lags, max_lags + 1)
for lag in lags:
    shifted = df['chickenpox'].shift(lag)
    corr = df['Giardiasis'].corr(shifted)
    cross_corr.append(corr)

# Save correlation results
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/arima_forecast_correlation_results.csv", index=False)

# Visualizations
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Giardiasis'], label='Giardiasis Forecast', alpha=0.7)
plt.plot(df.index, df['chickenpox'], label='Chickenpox Forecast', alpha=0.7)
plt.title('ARIMA Forecasts: Giardiasis and Chickenpox')
plt.xlabel('Date')
plt.ylabel('Forecasted Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/arima_forecast_time_series.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix (ARIMA Forecasts)')
plt.savefig(f"{output_dir}/arima_forecast_correlation_heatmap.png")
plt.close()

# Cross-correlation plot with max lag marked
plt.figure(figsize=(12, 6))
plt.stem(lags, cross_corr, basefmt=" ")
max_idx = np.argmax(np.abs(cross_corr))
plt.axvline(lags[max_idx], color='red', linestyle='--', label=f'Max Corr Lag: {lags[max_idx]}')
plt.title('Cross-Correlation: Giardiasis vs. Chickenpox (ARIMA Forecasts)')
plt.xlabel('Lag (weeks)')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/arima_forecast_cross_correlation_max.png")
plt.close()

# Print results
print("\nARIMA Forecast Correlation Results:")
print(results_df)
print(f"\nSaved results to: {output_dir}/arima_forecast_correlation_results.csv")
print(f"Saved plots to: {output_dir}/arima_forecast_time_series.png, arima_forecast_correlation_heatmap.png, arima_forecast_cross_correlation_max.png")

print("✅ ARIMA forecast correlation analysis completed.")