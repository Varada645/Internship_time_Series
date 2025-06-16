import pandas as pd

# Load the cleaned dataset
df = pd.read_csv(r"C:\Users\VARADA S NAIR\OneDrive\Desktop\inten_disease\us_cleaned.csv")

# Ensure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Generate expected date range (weekly, starting Monday)
expected_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='W-MON')

# Find missing weeks
missing_dates = expected_dates.difference(df['Date'])
print("Missing Week(s):")
for date in missing_dates:
    print(f"Year: {date.year}, Week: {date.isocalendar().week}, Date: {date}")
if not missing_dates.empty:
    print(f"⚠️ {len(missing_dates)} week(s) missing.")
else:
    print("✅ No missing weeks found after rechecking.")