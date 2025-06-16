import pandas as pd
import os

# ------------------ 1. Load Dataset ------------------
df = pd.read_csv("US_ID (1).csv")  # Update path if needed
df = df.dropna()
# ------------------ 2. Drop Static Columns ------------------
if df['Reporting Area'].nunique() == 1:
    df.drop('Reporting Area', axis=1, inplace=True)

# ------------------ 3. Missing Value Imputation ------------------
missing_cols = ['Dengue', 'HepatitisC', 'Lyme disease']
df[missing_cols] = df[missing_cols].fillna(0.0)

# ------------------ 4. Create 'Date' Column ------------------
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + df['Week'].astype(str) + '-1', format='%Y-W%W-%w')

# ------------------ 5. Extract 'Month' from 'Date' ------------------
df['Month'] = df['Date'].dt.month

# ------------------ 6. Reorder Columns ------------------
reordered = ['Year', 'Week', 'Month'] + [col for col in df.columns if col not in ['Year', 'Week', 'Month', 'Date']] + ['Date']
df = df[reordered]

# ------------------ 7. Final Checks ------------------
assert df.duplicated().sum() == 0, "⚠️ Duplicate rows detected!"
assert df.isnull().sum().sum() == 0, "⚠️ Missing values remain!"

# ------------------ 8. Save Cleaned Weekly Dataset ------------------
output_path = os.path.expanduser("~/us_cleaned.csv")
df.to_csv(output_path, index=False)

# ------------------ 9. Display Sample Data ------------------
print(f"✅ Preprocessing complete. Cleaned dataset saved to:\n{output_path}")
print("\n📌 Sample of cleaned data:")
print(df.head())

# ------------------ 10. Display Dataset Information ------------------
print("\n🔍 Dataset Information:")
print(df.info())

# ------------------ 11. Optional: Aggregate and Save Monthly and Yearly ------------------

# Set 'Date' as index for resampling
df.set_index('Date', inplace=True)

# Monthly Aggregation
monthly_df = df.resample('M').sum().reset_index()
monthly_df['Year'] = monthly_df['Date'].dt.year
monthly_df['Month'] = monthly_df['Date'].dt.month
monthly_path = os.path.expanduser("~/us_monthly_diseases.csv")
monthly_df.to_csv(monthly_path, index=False)
print(f"\n📆 Monthly aggregated data saved to: {monthly_path}")

# Yearly Aggregation
yearly_df = df.resample('Y').sum().reset_index()
yearly_df['Year'] = yearly_df['Date'].dt.year
yearly_path = os.path.expanduser("~/us_yearly_diseases.csv")
yearly_df.to_csv(yearly_path, index=False)
print(f"📅 Yearly aggregated data saved to: {yearly_path}")

# Reset index if needed later
df.reset_index(inplace=True)
# Display final dataset information
print("\n🔍 Final Dataset Information:" )
print(df.info())
