import pandas as pd
import os
# Sample code to read and display forecast results from different models
base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
# Base directory for model results
# Models: arima, sarima, gru, lstm



for model in ['arima', 'sarima', 'gru', 'lstm']:
    for disease in ['Giardiasis', 'chickenpox']:
        df = pd.read_csv(f"{base_dir}/{model}/{disease}/{model}_forecast.csv")
        print(f"\n{model} {disease} Forecast Sample:")
        print(df.head())