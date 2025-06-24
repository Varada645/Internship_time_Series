# stationarity_test.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import os

# Define paths
input_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/stationarity"
plot_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Diseases and seasonal periods (weeks)
diseases = {
    'Giardiasis': 26,  # Biannual
    'chickenpox': 52   # Yearly
}

def adf_test(series, name):
    """Perform ADF test and return p-value and statistic."""
    result = adfuller(series.dropna())
    return result[0], result[1]

def stationarity_analysis(series, name, m, output_dir, plot_dir):
    """Analyze stationarity and recommend differencing orders."""
    results = {'Disease': name, 'Test': [], 'ADF_Statistic': [], 'p_value': [], 'Stationary': []}
    
    # Original series
    stat, p_val = adf_test(series, f"{name} Original")
    results['Test'].append('Original')
    results['ADF_Statistic'].append(stat)
    results['p_value'].append(p_val)
    results['Stationary'].append(p_val < 0.05)
    
    # First differencing
    diff1 = series.diff(1).dropna()
    stat, p_val = adf_test(diff1, f"{name} 1st Diff")
    results['Test'].append('1st Differencing')
    results['ADF_Statistic'].append(stat)
    results['p_value'].append(p_val)
    results['Stationary'].append(p_val < 0.05)
    
    # Second differencing
    diff2 = diff1.diff(1).dropna()
    stat, p_val = adf_test(diff2, f"{name} 2nd Diff")
    results['Test'].append('2nd Differencing')
    results['ADF_Statistic'].append(stat)
    results['p_value'].append(p_val)
    results['Stationary'].append(p_val < 0.05)
    
    # Seasonal differencing
    diff_seasonal = series.diff(m).dropna()
    stat, p_val = adf_test(diff_seasonal, f"{name} Seasonal Diff (m={m})")
    results['Test'].append(f'Seasonal Differencing (m={m})')
    results['ADF_Statistic'].append(stat)
    results['p_value'].append(p_val)
    results['Stationary'].append(p_val < 0.05)
    
    # Recommend differencing orders
    d = 0
    if results['p_value'][0] > 0.05:  # Original non-stationary
        d = 1
        if results['p_value'][1] > 0.05:  # 1st diff non-stationary
            d = 2
    D = 1 if results['p_value'][3] > 0.05 else 1  # Enforce D=1 for SARIMA if seasonal
    
    results['Recommended_d'] = [d, '', '', '']
    results['Recommended_D'] = [D if m else 0, '', '', '']
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/{name}_stationarity_results.csv", index=False)
    
    # Plot diagnostics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(series, label='Original')
    plt.title(f'{name} Original Series')
    plt.subplot(2, 2, 2)
    plot_acf(series.dropna(), ax=plt.gca(), title='ACF Original')
    plt.subplot(2, 2, 3)
    plt.plot(diff1, label='1st Diff')
    plt.title('1st Differencing')
    plt.subplot(2, 2, 4)
    plot_acf(diff1, ax=plt.gca(), title='ACF 1st Diff')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_stationarity_diagnostics.png")
    plt.close()
    
    # Seasonal differencing plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(diff_seasonal, label=f'Seasonal Diff (m={m})')
    plt.title(f'Seasonal Differencing (m={m})')
    plt.subplot(1, 2, 2)
    plot_acf(diff_seasonal, ax=plt.gca(), title='ACF Seasonal Diff')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_seasonal_diagnostics.png")
    plt.close()
    
    print(f"{name} Stationarity Results:")
    print(results_df)
    print(f"Recommended d: {d}, Recommended D: {D}")
    print(f"Saved results to {output_dir}/{name}_stationarity_results.csv")
    print(f"Saved plots to {plot_dir}/{name}_stationarity_diagnostics.png and {name}_seasonal_diagnostics.png")

# Process each disease
for disease, m in diseases.items():
    train_path = f"{input_dir}/{disease}_train.csv"
    if not os.path.exists(train_path):
        print(f"Warning: Train file for {disease} not found. Skipping.")
        continue
    
    train = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    
    try:
        if train[disease].isna().any():
            raise ValueError(f"Missing values in {disease} data")
        if (train[disease] < 0).any():
            raise ValueError(f"Negative values in {disease} data")
        
        stationarity_analysis(train[disease], disease, m, output_dir, plot_dir)
    
    except Exception as e:
        print(f"Stationarity test failed for {disease}: {str(e)}")

print("✅ Stationarity testing completed.")