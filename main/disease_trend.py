import pandas as pd

df = pd.read_csv(r'C:\Users\VARADA S NAIR\OneDrive\Desktop\inter_disease\preprocess\preprocessed_dataset.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

import matplotlib.pyplot as plt

df[['Giardiasis', 'chickenpox']].plot(figsize=(12,5))
plt.title("Disease Trends")
plt.grid(True)
plt.show()

def split_series(series, ratio=0.8):
    split_point = int(len(series) * ratio)
    return series[:split_point], series[split_point:]

g_train, g_test = split_series(df['Giardiasis'])
c_train, c_test = split_series(df['chickenpox'])
