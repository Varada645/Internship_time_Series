import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, RobustScaler

def create_features(df, disease, lags=4):
    """Create lag features, rolling statistics, and utilize existing seasonal features"""
    df = df.copy()
    
    # Lag features
    for lag in range(1, lags+1):
        df[f'{disease}_lag_{lag}'] = df[disease].shift(lag)
    
    # Rolling statistics
    df[f'{disease}_rolling_mean'] = df[disease].rolling(window=4).mean()
    df[f'{disease}_rolling_std'] = df[disease].rolling(window=4).std()
    
    # Utilize existing seasonal features (week, month, quarter)
    # Note: These are already in the input file, so we don't need to recreate them
    # Just ensure they're included in the output
    
    # Drop initial rows with NaN values from lag features
    df = df.dropna()
    
    return df

def train_test_split_enhanced(input_path, output_dir, diseases, split_date='2016-12-31'):
    """Enhanced version with feature engineering using seasonal features"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    # Load the data with seasonal features
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    
    for disease in diseases:
        if disease not in df.columns:
            print(f"Warning: {disease} not found in dataset. Skipping.")
            continue
        
        # Get all columns relevant for this disease (disease column + seasonal features)
        cols_to_keep = [disease, 'Week', 'Month', 'DayOfWeek']  # Using the seasonal features we added
        disease_df = df[cols_to_keep].copy()
        
        # Create features (this will add lags and rolling stats)
        disease_df = create_features(disease_df, disease)
        
        # Split data
        train = disease_df.loc[:split_date]
        test = disease_df.loc[pd.Timestamp(split_date) + pd.Timedelta(days=7):]
        
        # Save
        train.to_csv(f"{output_dir}/{disease}_train.csv")
        test.to_csv(f"{output_dir}/{disease}_test.csv")
        
        print(f"{disease} - Train size: {len(train)}, Test size: {len(test)}")
        print(f"Features used: {', '.join(train.columns)}")
    
    print("✅ Enhanced train-test split with seasonality completed.")

# Usage with the seasonal features file
input_path = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/selected_diseases_time_series_with_seasonality.csv"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
diseases = ['Giardiasis', 'chickenpox']
train_test_split_enhanced(input_path, output_dir, diseases)
print("Sample 5 rows from Giardiasis train set:")
df = pd.read_csv(f"{output_dir}/Giardiasis_train.csv")
print(df.head())
print("Sample 5 rows from chickenpox train set:")
df = pd.read_csv(f"{output_dir}/chickenpox_train.csv")
print(df.head())
print("Sample 5 rows from Giardiasis test set:")
df = pd.read_csv(f"{output_dir}/Giardiasis_test.csv")
print(df.head())
print("Sample 5 rows from chickenpox test set:")
df = pd.read_csv(f"{output_dir}/chickenpox_test.csv")
print(df.head())