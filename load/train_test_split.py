# program_2_train_test_split.py
import pandas as pd
import os

# Define input and output paths
input_path = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/selected_diseases_time_series.csv"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
os.makedirs(output_dir, exist_ok=True)

# List of diseases (use lowercase for consistency)
diseases = ['Giardiasis', 'chickenpox']

# Check if input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found at {input_path}")

# Load combined time series
df = pd.read_csv(input_path, index_col='Date', parse_dates=True)

def train_test_split(series, split_date='2016-12-31'):
    train = series.loc[:split_date]
    test = series.loc[pd.Timestamp(split_date) + pd.Timedelta(days=7):]
    return train, test

# Process each disease
for disease in diseases:
    if disease not in df.columns:
        print(f"Warning: {disease} not found in input data. Skipping.")
        continue
    
    # Split into train and test
    train, test = train_test_split(df[disease])
    
    # Save train and test sets (lowercase filenames)
    train.to_csv(f"{output_dir}/{disease}_train.csv")
    test.to_csv(f"{output_dir}/{disease}_test.csv")
    
    print(f"{disease} - Train size: {len(train)}, Test size: {len(test)}")
    print(f"Saved train and test sets for {disease} to {output_dir}")

print("✅ Train-test split completed.")