import pandas as pd
import os

# Load the cleaned dataset
df = pd.read_csv("C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/us_cleaned.csv")
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])

# Generate expected date range (weekly, starting Monday)
expected_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='W-MON')
missing_dates = expected_dates.difference(df['Date'])
print(f"Missing Week(s): {len(missing_dates)} found: {missing_dates}")

# Create missing data with zero counts
disease_cols = ['Chlamydia', 'Dengue', 'Giardiasis', 'Gonorrhea', 'Haemophilus', 
                'HepatitisA', 'HepatitisB', 'HepatitisC', 'Legionellosis', 
                'Lyme disease', 'Malaria', 'Syphilis', 'chickenpox']
missing_data = []
for date in missing_dates:
    missing_data.append({
        'Year': date.year,
        'Week': date.isocalendar().week,
        'Month': date.month,
        'Quarter': (date.month - 1) // 3 + 1,  # Derive quarter
        'Date': date,
        **{col: 0.0 for col in disease_cols}
    })

# Append and sort
missing_df = pd.DataFrame(missing_data)
df = pd.concat([df, missing_df], ignore_index=True).sort_values('Date').reset_index(drop=True)

# Verify no missing weeks
missing_dates_after = expected_dates.difference(df['Date'])
print(f"After imputation, missing weeks: {'✅ None' if len(missing_dates_after) == 0 else len(missing_dates_after)}")

# Save updated dataset
output_path = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/us_cleaned_complete.csv"
df.to_csv(output_path, index=False)
print(f"✅ Updated dataset saved to: {output_path}")

# Confirm new shape
print(f"New dataset shape: {df.shape} (expected: 733 rows)")
# Display sample data
print("\n📌 Sample of updated data:" )