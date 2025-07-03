# program_11_correlation.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
input_dir = f"{base_dir}/splits"
output_dir = f"{base_dir}/correlation"
os.makedirs(output_dir, exist_ok=True)

# Diseases
diseases = {'Giardiasis': 'giardiasis', 'chickenpox': 'chickenpox'}

# Load data
data = {}
for disease_name, file_name in diseases.items():
    try:
        train_path = f"{input_dir}/{file_name}_train.csv"
        test_path = f"{input_dir}/{file_name}_test.csv"
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print(f"Error: {disease_name} train/test files missing at {train_path}, {test_path}. Skipping.")
            continue
        
        train = pd.read_csv(train_path, encoding='utf-8-sig', index_col='Date', parse_dates=['Date'])
        test = pd.read_csv(test_path, encoding='utf-8-sig', index_col='Date', parse_dates=['Date'])
        full_data = pd.concat([train, test])
        data[disease_name] = full_data[disease_name]
    except Exception as e:
        print(f"Error loading {disease_name}: {str(e)}")

# Create combined DataFrame
df = pd.DataFrame(data).dropna()

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
results_df.to_csv(f"{output_dir}/disease_correlation_results.csv", index=False)

# Visualizations
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Giardiasis'], label='Giardiasis', alpha=0.7)
plt.plot(df.index, df['chickenpox'], label='chickenpox', alpha=0.7)
plt.title('Time Series of Giardiasis and chickenpox (2006-2021)')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/disease_time_series.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix (Giardiasis vs. chickenpox)')
plt.savefig(f"{output_dir}/disease_correlation_heatmap.png")
plt.close()

# Cross-correlation plot with max lag marked
plt.figure(figsize=(12, 6))
plt.stem(lags, cross_corr, basefmt=" ")
max_idx = np.argmax(np.abs(cross_corr))
plt.axvline(lags[max_idx], color='red', linestyle='--', label=f'Max Corr Lag: {lags[max_idx]}')
plt.title('Cross-Correlation: Giardiasis vs. chickenpox')
plt.xlabel('Lag (weeks)')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/disease_cross_correlation_max.png")
plt.close()

# Print results
print("\nCorrelation Results:")
print(results_df)
print(f"\nSaved results to: {output_dir}/disease_correlation_results.csv")
print(f"Saved plots to: {output_dir}/disease_time_series.png, disease_correlation_heatmap.png, disease_cross_correlation_max.png")

print("✅ Correlation analysis completed.")