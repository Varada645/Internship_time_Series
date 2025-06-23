import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("US_ID (1).csv")

# ------------------ 1. Structure Check ------------------
print("✅ Dataset Dimensions:", df.shape)
print("\n📌 Column Names:", df.columns.tolist())
print("\n🔍 Data Types & Non-Null Counts:")
print(df.info())

# ------------------ 2. First 5 Rows ------------------
print("\n🧾 First 5 Rows:")
print(df.head())

# ------------------ 3. Missing Values ------------------
print("\n🚨 Missing Values:")
print(df.isnull().sum().sort_values(ascending=False))

# ------------------ 4. Descriptive Statistics ------------------
print("\n📊 Descriptive Statistics:")
print(df.describe(include='all'))

# ------------------ 5. Total Cases Per Disease ------------------
disease_cols = df.columns[3:]  # Assuming Area, Year, Week are first 3
print("\n💉 Total Cases Per Disease:")
print(df[disease_cols].sum().sort_values(ascending=False))

# ------------------ 6. Create a Date Column ------------------
# Use ISO week format for epidemiological data      

df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Week'].astype(str) + '1', format='%G%V%u')

# ------------------ 7. Plot Disease Trends ------------------
# Exclude non-disease columns from plotting
non_disease_cols = ['Reporting Area', 'Year', 'Week', 'Date']
disease_columns = [col for col in df.columns if col not in non_disease_cols]

for disease in disease_columns:
    plt.figure(figsize=(10, 4))
    df.groupby('Date')[disease].sum().plot()
    plt.title(f'{disease} Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ 8. Correlation Heatmap ------------------
plt.figure(figsize=(12, 10))
corr_matrix = df[disease_columns].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Between Diseases")
plt.tight_layout()
plt.show()