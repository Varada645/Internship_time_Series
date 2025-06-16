import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# Load updated dataset
df = pd.read_csv("us_cleaned_complete.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Disease columns
disease_columns = [
    'Chlamydia', 'Dengue', 'Giardiasis', 'Gonorrhea', 'Haemophilus',
    'HepatitisA', 'HepatitisB', 'HepatitisC', 'Legionellosis',
    'Lyme disease', 'Malaria', 'Syphilis', 'chickenpox'
]

# Output folder
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/seasonality_plots_complete"
os.makedirs(output_dir, exist_ok=True)

# Seasonal decomposition
for disease in disease_columns:
    print(f"🔍 Decomposing: {disease}")
    try:
        result = seasonal_decompose(df[disease], model='additive', period=52)
        fig = result.plot()
        fig.suptitle(f'Seasonal Decomposition of {disease} (Complete Data)', fontsize=14)
        plt.tight_layout()
        fig.savefig(f"{output_dir}/{disease}_seasonal_decomposition_complete.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not decompose {disease}: {e}")

print("✅ Seasonal decomposition plots saved to folder:")
print(output_dir)