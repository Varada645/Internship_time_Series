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
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + df['Week'].astype(str) + '-1', format='%Y-W%W-%w')

# ------------------ 7. Plot Disease Trend (e.g. Dengue) ------------------
disease_to_plot = 'Dengue'

plt.figure(figsize=(10, 4))
df.groupby('Date')[disease_to_plot].sum().plot()
plt.title(f'{disease_to_plot} Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ 8. Correlation Heatmap ------------------
plt.figure(figsize=(12, 10))
corr_matrix = df[disease_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Between Diseases")
plt.tight_layout()
plt.show()




