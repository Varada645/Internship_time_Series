import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output directory
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/eda/investigate_missing_data.csv"

os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("US_ID (1).csv")  # Change filename if needed

# 1. Quantify missing data
print("Missing Data Summary:")
print(df[['Dengue', 'HepatitisC', 'Lyme disease']].isna().sum())
print("\nMissing Percentage:")
print(df[['Dengue', 'HepatitisC', 'Lyme disease']].isna().mean() * 100)

# 2. Temporal distribution of missing data
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Week'].astype(str) + '-1', format='%Y-%W-%w')
missing_by_year = df.groupby(df['Date'].dt.year)[['Dengue', 'HepatitisC', 'Lyme disease']].apply(lambda x: x.isna().sum())
print("\nMissing Data by Year:")
print(missing_by_year)

# Plot missing data over time
plt.figure(figsize=(12, 6))
for col in ['Dengue', 'HepatitisC', 'Lyme disease']:
    missing = df[col].isna().astype(int)
    plt.plot(df['Date'], missing, label=col, alpha=0.6)
plt.title('Missing Data Over Time')
plt.xlabel('Date')
plt.ylabel('Missing (1) / Present (0)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/missing_data_timeline.png")
plt.close()

# 3. Regional distribution (if Reporting Area exists)
if 'Reporting Area' in df.columns:
    missing_by_region = df.groupby('Reporting Area')[['Dengue', 'HepatitisC', 'Lyme disease']].apply(lambda x: x.isna().sum())
    print("\nMissing Data by Reporting Area:")
    print(missing_by_region)
    # Save to CSV
    missing_by_region.to_csv(f"{output_dir}/missing_by_region.csv")

# 4. Check for co-occurring missingness
missing_corr = df[['Dengue', 'HepatitisC', 'Lyme disease']].isna().corr()
plt.figure(figsize=(8, 6))
sns.heatmap(missing_corr, annot=True, cmap='coolwarm')
plt.title('Correlation of Missing Data Between Diseases')
plt.tight_layout()
plt.savefig(f"{output_dir}/missing_data_correlation.png")
plt.close()

# 5. Compare missing vs. non-missing periods for Dengue
dengue_missing = df[df['Dengue'].isna()][['Date', 'Year', 'Week']]
dengue_non_missing = df[df['Dengue'].notna()][['Date', 'Year', 'Week', 'Dengue']]
print("\nDengue Missing Periods (Sample):")
print(dengue_missing.head())
print("\nDengue Non-Missing Stats:")
print(dengue_non_missing['Dengue'].describe())