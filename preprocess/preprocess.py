import pandas as pd
import numpy as np
import os


# Output directory

output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/preprocess"

# Load the dataset
df = pd.read_csv("C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/US_ID (1).csv")

df = df.drop(columns=['Reporting Area'])

# ------------------ 1. Validate and Summarize Missing Data ------------------
print("Missing Data Summary:")
missing_summary = df[['Dengue', 'HepatitisC', 'Lyme disease']].isna().sum()
print(missing_summary)
print("\nMissing Percentage:")
print((missing_summary / len(df) * 100).round(2))

# ------------------ 2. Handle Duplicates ------------------
# Check for duplicate dates
df['Temp_Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Week'].astype(str) + '-1', format='%Y-%W-%w', errors='coerce')
duplicates = df['Temp_Date'].duplicated().sum()
print(f"\nDuplicate Dates: {duplicates}")
if duplicates > 0:
    print("Aggregating duplicates by sum for case counts.")
    # Sum disease counts, keep first for non-numeric (e.g., Reporting Area)
    agg_dict = {col: 'sum' if col in ['Dengue', 'HepatitisC', 'Lyme disease'] else 'first' for col in df.columns if col != 'Temp_Date'}
    df = df.groupby('Temp_Date').agg(agg_dict).reset_index()
    df = df.rename(columns={'Temp_Date': 'Date'})

# ------------------ 3. Impute Missing Data ------------------
# Dengue: Zero imputation (rarity, median=0, 2006–2009 missing)
df['Dengue'] = df['Dengue'].fillna(0)

# Hepatitis C: Seasonal imputation (monthly mean, 2006–2009 missing)
df['Month'] = df['Date'].dt.month
hepatitis_monthly_mean = df.groupby('Month')['HepatitisC'].mean()
df['HepatitisC'] = df.apply(
    lambda x: hepatitis_monthly_mean[x['Month']] if pd.isna(x['HepatitisC']) else x['HepatitisC'],
    axis=1
)

# Lyme disease: Seasonal imputation (monthly mean from 2006–2016, 2017–2019 missing)
lyme_monthly_mean = df[df['Year'] <= 2016].groupby('Month')['Lyme disease'].mean()
df['Lyme disease'] = df.apply(
    lambda x: lyme_monthly_mean[x['Month']] if pd.isna(x['Lyme disease']) else x['Lyme disease'],
    axis=1
)

# Round to integers (disease counts)
for col in ['Dengue', 'HepatitisC', 'Lyme disease']:
    df[col] = df[col].round().astype(int)

# ------------------ 4. Create Date Column ------------------
# Ensure Date is correctly formatted (already computed as Temp_Date)
if 'Date' not in df.columns:
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Week'].astype(str) + '-1', format='%Y-%W-%w', errors='coerce')

# ------------------ 5. Add Week and Month Columns ------------------
# Use calendar month from Date
df['Week'] = df['Date'].dt.isocalendar().week
df['Month'] = df['Date'].dt.month

# ------------------ 6. Enforce Weekly Frequency ------------------
# Set Date as index and enforce weekly frequency (W-MON)
df = df.set_index('Date').asfreq('W-MON', method='ffill').reset_index()
print("\nData Frequency:", df.index.freq if hasattr(df.index, 'freq') else "Set to W-MON")

# ------------------ 7. Validate Data ------------------
# Check for remaining missing values
print("\nMissing Data After Imputation:")
print(df[['Dengue', 'HepatitisC', 'Lyme disease']].isna().sum())

# ...existing code...

# Ensure 'Date' column exists and is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add or update 'Month' and 'WeekOfYear' columns next to 'Date'
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Optional: Reorder columns so 'Date', 'Month', 'WeekOfYear' are next to each other
cols = list(df.columns)
for col in ['Month', 'WeekOfYear']:
    cols.remove(col)
date_idx = cols.index('Date')
for i, col in enumerate(['Month', 'WeekOfYear']):
    cols.insert(date_idx + 1 + i, col)
df = df[cols]

# ...existing code...
# Summary stats
print("\nDescriptive Stats After Preprocessing:")
print(df[['Dengue', 'HepatitisC', 'Lyme disease']].describe())

# Example: Display the first 5 rows of the cleaned DataFrame
print(df.head())

# ------------------ 8. Save Preprocessed Dataset ------------------
output_path = f"{output_dir}/preprocessed_dataset.csv"
df.to_csv(output_path, index=False)
print(f"✅ Preprocessing complete. Saved as '{output_path}'.")