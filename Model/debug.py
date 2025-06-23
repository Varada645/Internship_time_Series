# debug_csv.py
import pandas as pd
import os

base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
files = ['giardiasis_train.csv', 'giardiasis_test.csv', 'chickenpox_train.csv', 'chickenpox_test.csv']

for file in files:
    path = f"{base_dir}/{file}"
    print(f"\nChecking {path}:")
    if not os.path.exists(path):
        print(f"File does not exist.")
        continue
    
    # Check first few lines
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:3]
        print(f"First 3 lines:\n{lines}")
    
    # Try reading without parse_dates
    try:
        df = pd.read_csv(path)
        print(f"Columns: {df.columns}")
        print(f"Head:\n{df.head(2)}")
    except Exception as e:
        print(f"Error reading without parse_dates: {e}")
    
    # Try reading with parse_dates
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
        print(f"Parsed successfully. Head:\n{df.head(2)}")
    except Exception as e:
        print(f"Error reading with parse_dates: {e}")