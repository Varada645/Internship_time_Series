import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/us_cleaned_complete.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Check shape
print(f"Dataset shape: {df.shape} (expected ~733 rows)")

# Check for missing weeks
expected_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='W-MON')
missing_dates = expected_dates.difference(df['Date'])
print(f"Missing weeks: {'✅ None' if len(missing_dates) == 0 else missing_dates}")

# View imputed rows
imputed_dates = ['2007-12-31', '2012-12-31', '2018-12-31']
print("\nImputed rows:")
print(df[df['Date'].isin(imputed_dates)][['Date', 'Year', 'Week', 'Chlamydia', 'Dengue', 'Lyme disease']])