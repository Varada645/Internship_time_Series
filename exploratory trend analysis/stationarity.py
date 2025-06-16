import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import os

# Load updated dataset
df = pd.read_csv("C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/us_cleaned_complete.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Disease columns
exclude_cols = ['Year', 'Week', 'Month', 'Quarter']
disease_cols = [col for col in df.columns if col not in exclude_cols]

# ADF test function
def adf_check(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'Disease': name,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Is Stationary': result[1] <= 0.05
    }

# Run ADF test
report = []
for col in disease_cols:
    transformed = np.log(df[col].replace(0, 1)).diff()  # Log + differencing
    result = adf_check(transformed, col)
    report.append(result)

# Compile and save report
report_df = pd.DataFrame(report)
print("\n📄 ADF Stationarity Report (Complete Data):\n")
print(report_df.sort_values(by='Is Stationary', ascending=False))

output_path = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/adf_stationarity_report_complete.csv"
report_df.to_csv(output_path, index=False)
print(f"\n📁 Saved ADF report to: {output_path}")