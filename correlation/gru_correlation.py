import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
BASE_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/gru_model"
CHICKENPOX_PATH = os.path.join(BASE_DIR, "chickenpox_gru_forecast_future.csv")
GIARDIASIS_PATH = os.path.join(BASE_DIR, "Giardiasis_gru_forecast_future.csv")

# === LOAD FORECAST DATA ===
chickenpox = pd.read_csv(CHICKENPOX_PATH)
giardiasis = pd.read_csv(GIARDIASIS_PATH)

# Convert date column
chickenpox['Date'] = pd.to_datetime(chickenpox['Date'])
giardiasis['Date'] = pd.to_datetime(giardiasis['Date'])

# === MERGE FORECASTS ON DATE ===
chickenpox.rename(columns={'Forecast': 'Chickenpox'}, inplace=True)
giardiasis.rename(columns={'Forecast': 'Giardiasis'}, inplace=True)
merged_df = pd.merge(chickenpox[['Date', 'Chickenpox']], giardiasis[['Date', 'Giardiasis']], on='Date')

# === PLOT THE FORECASTS ===
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Date'], merged_df['Chickenpox'], label='Chickenpox Forecast', color='blue')
plt.plot(merged_df['Date'], merged_df['Giardiasis'], label='Giardiasis Forecast', color='green')
plt.title("GRU Forecasts: Chickenpox vs Giardiasis (Future until 2020)")
plt.xlabel("Date")
plt.ylabel("Predicted Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "forecast_comparison_plot.png"))
plt.show()

# === CALCULATE CORRELATION ===
correlation = merged_df['Chickenpox'].corr(merged_df['Giardiasis'])
print(f"📊 Pearson Correlation between GRU forecasts: {correlation:.4f}")

# === HEATMAP VISUALIZATION ===
plt.figure(figsize=(6, 4))
sns.heatmap(merged_df[['Chickenpox', 'Giardiasis']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: GRU Forecasts")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "forecast_correlation_heatmap.png"))
plt.show()
