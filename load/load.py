# program_1_load_prepare_data.py
import pandas as pd
import os

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

# List of diseases to analyze
diseases = ['Giardiasis', 'chickenpox']  # Modify to include other diseases

# Save combined time series for selected diseases
output_path = f"{output_dir}/selected_diseases_time_series.csv"
df[diseases].to_csv(output_path)
print(f"Saved combined time series for {', '.join(diseases)} to {output_path}")

print("✅ Data loaded and prepared for analysis.")