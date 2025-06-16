import pandas as pd
import matplotlib.pyplot as plt

# ------------------ 1. Load Data ------------------
df = pd.read_csv("C:/Users/VARADA S NAIR/us_cleaned.csv", parse_dates=['Date'])

print("📂 Dataset Loaded Successfully")
print("🧾 Shape:", df.shape)
print("🔍 Columns:", df.columns.tolist())

# ------------------ 2. Schema Validation ------------------
print("\n🔎 Data Types:")
print(df.dtypes)

print("\n🚨 Missing Values:")
print(df.isnull().sum())

# ------------------ 3. Range Validation ------------------
print("\n📆 Year Range:", df['Year'].min(), "to", df['Year'].max())
print("📅 Week Range:", df['Week'].min(), "to", df['Week'].max())
print("📅 Month Range:", df['Month'].min(), "to", df['Month'].max())

# Check for negative disease counts
disease_cols = df.columns[3:-1]  # Skipping Year, Week, Month, Date
negatives = (df[disease_cols] < 0).sum()
print("\n❗ Negative Values Detected (should be 0):")
print(negatives[negatives > 0])

# ------------------ 4. Duplicates Check ------------------
duplicate_count = df.duplicated().sum()
print(f"\n📛 Duplicates Found: {duplicate_count}")

# ------------------ 5. Temporal Consistency ------------------
df_check = df.copy()
df_check['Reconstructed_Date'] = pd.to_datetime(df_check['Year'].astype(str) + '-W' + df_check['Week'].astype(str) + '-1', format='%Y-W%W-%w')
date_consistency = (df_check['Date'] == df_check['Reconstructed_Date']).all()
print("\n⏳ Date Consistency Check:", "✅ Passed" if date_consistency else "❌ Failed")

# ------------------ 6. Outlier Detection ------------------
print("\n📊 Plotting Boxplot for Disease Counts (detect outliers)...")
plt.figure(figsize=(15, 6))
df[disease_cols].boxplot(rot=90)
plt.title("Disease Count Distribution (Outlier Check)")
plt.tight_layout()
plt.show()

# ------------------ 7. Summary of Totals ------------------
print("\n📈 Total Disease Counts:")
print(df[disease_cols].sum().sort_values(ascending=False))

print("\n✅ Data validation complete.")
