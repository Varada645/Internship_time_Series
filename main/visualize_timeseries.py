# This script visualizes time series data for selected diseases and saves the plots.
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define input and output paths
input_path = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/selected_diseases_time_series.csv"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/plots"
os.makedirs(output_dir, exist_ok=True)

# List of diseases
diseases = ['Giardiasis', 'chickenpox']

# Check if input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found at {input_path}")

# Load combined time series
df = pd.read_csv(input_path, index_col='Date', parse_dates=True)

# Process each disease
for disease in diseases:
    if disease not in df.columns:
        print(f"Warning: {disease} not found in input data. Skipping.")
        continue
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df[disease], label='Actual')
    plt.title(f'{disease} Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f"{output_dir}/{disease}_time_series_plot.png")
    plt.close()
    
    print(f"Saved time series plot for {disease} to {output_dir}")

print("✅ Time series visualization completed.")