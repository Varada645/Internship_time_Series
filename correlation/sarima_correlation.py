import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
BASE_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/sarima"
OUTPUT_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/correlation/sarima"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD FORECASTS ---
def load_forecast(disease):
    path = os.path.join(BASE_DIR, f"{disease}_sarima_forecast_extended.csv")
    df = pd.read_csv(path, usecols=["Date", "Forecast"], parse_dates=["Date"])
    df.rename(columns={"Forecast": disease}, inplace=True)
    return df

giardiasis_df = load_forecast("Giardiasis")
chickenpox_df = load_forecast("chickenpox")

# --- MERGE DATA ---
df = pd.merge(giardiasis_df, chickenpox_df, on="Date").set_index("Date")

# --- CALCULATE CORRELATIONS ---
def get_correlation_stats(df, label):
    pearson_corr, pearson_p = pearsonr(df['Giardiasis'], df['chickenpox'])
    spearman_corr, spearman_p = spearmanr(df['Giardiasis'], df['chickenpox'])
    return {
        "Data Type": label,
        "Pearson Correlation": pearson_corr,
        "Pearson p-value": pearson_p,
        "Spearman Correlation": spearman_corr,
        "Spearman p-value": spearman_p
    }

results = [
    get_correlation_stats(df, "Raw"),
    get_correlation_stats(df.diff().dropna(), "Differenced")
]

# --- CROSS-CORRELATION ---
max_lags = 52
lags = range(-max_lags, max_lags + 1)
cross_corr = [df['Giardiasis'].corr(df['chickenpox'].shift(lag)) for lag in lags]
max_lag = lags[np.argmax(np.abs(cross_corr))]

# --- SAVE RESULTS ---
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/sarima_forecast_correlation_results.csv", index=False)

# --- PLOT 1: Time Series Forecasts ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Giardiasis'], label="Giardiasis Forecast")
plt.plot(df.index, df['chickenpox'], label="Chickenpox Forecast")
plt.title("SARIMA Forecasts: Giardiasis and Chickenpox")
plt.xlabel("Date")
plt.ylabel("Forecasted Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sarima_forecast_time_series.png")
plt.close()

# --- PLOT 2: Correlation Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix (SARIMA Forecasts)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sarima_forecast_correlation_heatmap.png")
plt.close()

# --- PLOT 3: Cross-Correlation ---
plt.figure(figsize=(12, 6))
plt.stem(lags, cross_corr, basefmt=" ")
plt.axvline(max_lag, color='red', linestyle='--', label=f'Max Corr Lag: {max_lag}')
plt.title("Cross-Correlation: Giardiasis vs. Chickenpox (SARIMA Forecasts)")
plt.xlabel("Lag (weeks)")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sarima_forecast_cross_correlation_max.png")
plt.close()

# --- PRINT SUMMARY ---
print("\n📊 SARIMA Forecast Correlation Summary:\n")
print(results_df.round(4))
print(f"\n✅ Results saved in: {OUTPUT_DIR}")
