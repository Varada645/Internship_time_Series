import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from warnings import catch_warnings, simplefilter
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Define paths
input_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/stationarity"
plot_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Diseases and their seasonal periods
diseases = {
    'Giardiasis': 26,
    'chickenpox': 52
}

def safe_adfuller(series):
    with catch_warnings():
        simplefilter("ignore", InterpolationWarning)
        result = adfuller(series.dropna())
    return result[0], result[1]

def safe_kpss(series):
    with catch_warnings():
        simplefilter("ignore", InterpolationWarning)
        result = kpss(series.dropna(), nlags='auto')
    return result[0], result[1]

def enhanced_stationarity_test(series, name, m=52):
    results = []

    # Original series
    adf_stat, adf_p = safe_adfuller(series)
    kpss_stat, kpss_p = safe_kpss(series)
    results.append({
        'Test': 'Original',
        'ADF_Statistic': adf_stat,
        'ADF_p-value': adf_p,
        'KPSS_Statistic': kpss_stat,
        'KPSS_p-value': kpss_p,
        'Stationary_ADF': adf_p < 0.05,
        'Stationary_KPSS': kpss_p > 0.05
    })

    # First differencing
    diff1 = series.diff().dropna()
    adf_stat, adf_p = safe_adfuller(diff1)
    kpss_stat, kpss_p = safe_kpss(diff1)
    results.append({
        'Test': 'First Difference',
        'ADF_Statistic': adf_stat,
        'ADF_p-value': adf_p,
        'KPSS_Statistic': kpss_stat,
        'KPSS_p-value': kpss_p,
        'Stationary_ADF': adf_p < 0.05,
        'Stationary_KPSS': kpss_p > 0.05
    })

    # Seasonal differencing
    diff_seasonal = series.diff(m).dropna()
    if len(diff_seasonal) > 0:
        adf_stat, adf_p = safe_adfuller(diff_seasonal)
        kpss_stat, kpss_p = safe_kpss(diff_seasonal)
        results.append({
            'Test': f'Seasonal Difference (m={m})',
            'ADF_Statistic': adf_stat,
            'ADF_p-value': adf_p,
            'KPSS_Statistic': kpss_stat,
            'KPSS_p-value': kpss_p,
            'Stationary_ADF': adf_p < 0.05,
            'Stationary_KPSS': kpss_p > 0.05
        })

    # Decomposition residuals
    decomposition = seasonal_decompose(series.dropna(), period=m)
    resid = decomposition.resid.dropna()
    adf_stat, adf_p = safe_adfuller(resid)
    kpss_stat, kpss_p = safe_kpss(resid)
    results.append({
        'Test': 'Residuals',
        'ADF_Statistic': adf_stat,
        'ADF_p-value': adf_p,
        'KPSS_Statistic': kpss_stat,
        'KPSS_p-value': kpss_p,
        'Stationary_ADF': adf_p < 0.05,
        'Stationary_KPSS': kpss_p > 0.05
    })

    # Auto-suggest differencing
    recommended_d = 0 if results[0]['Stationary_ADF'] else 1
    recommended_D = 1 if len(results) > 2 and not results[2]['Stationary_ADF'] else 0

    return pd.DataFrame(results), recommended_d, recommended_D

def process_disease(disease, m):
    path = f"{input_dir}/{disease}_train.csv"
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return

    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    series = df[disease]

    if series.isna().any():
        print(f"Missing values in {disease}")
        return
    if (series < 0).any():
        print(f"Negative values in {disease}")
        return

    results_df, d, D = enhanced_stationarity_test(series, disease, m)
    results_df.to_csv(f"{output_dir}/{disease}_enhanced_stationarity.csv", index=False)

    # ACF/PACF diagnostics
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plot_acf(series.dropna(), ax=plt.gca(), title=f"ACF - {disease}")
    plt.subplot(1, 2, 2)
    plot_pacf(series.dropna(), ax=plt.gca(), title=f"PACF - {disease}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{disease}_acf_pacf.png")
    plt.close()

    # Plot original and transformed series
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(series)
    plt.title("Original")

    plt.subplot(2, 2, 2)
    plt.plot(series.diff().dropna())
    plt.title("First Difference")

    plt.subplot(2, 2, 3)
    plt.plot(series.diff(m).dropna())
    plt.title(f"Seasonal Difference (m={m})")

    decomposition = seasonal_decompose(series, period=m)
    plt.subplot(2, 2, 4)
    plt.plot(decomposition.resid)
    plt.title("Residuals from Decomposition")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{disease}_enhanced_stationarity_plot.png")
    plt.close()

    print(f"\n✅ {disease} Analysis Done")
    print(f"Recommended d: {d}, Recommended D: {D}\n")

# Run for each disease
for disease, m in diseases.items():
    process_disease(disease, m)

print("✅ All enhanced stationarity + order suggestion completed.")
