import pandas as pd
import os
from datetime import datetime

# Define input and output paths
input_path = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/preprocess/preprocessed_dataset.csv"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
os.makedirs(output_dir, exist_ok=True)

# Check if input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found at {input_path}")

# Load preprocessed dataset
df = pd.read_csv(input_path)

# Ensure Date is datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Add seasonal features
df['Week'] = df.index.isocalendar().week  # Week number (1-52/53)
df['Month'] = df.index.month              # Month number (1-12)
df['DayOfWeek'] = df.index.dayofweek      # Day of week (0-6)

# List of diseases to analyze
diseases = ['Giardiasis', 'chickenpox']  # Modify to include other diseases

# Select columns to save (diseases + seasonal features)
columns_to_save = diseases + ['Week', 'Month', 'DayOfWeek']

# Save combined time series with seasonal features
output_path = f"{output_dir}/selected_diseases_time_series_with_seasonality.csv"
df[columns_to_save].to_csv(output_path)
print(f"Saved combined time series with seasonal features for {', '.join(diseases)} to {output_path}")

print("✅ Data loaded and prepared for analysis with seasonal features.")