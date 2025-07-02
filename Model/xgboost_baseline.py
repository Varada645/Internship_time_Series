import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# --- CONFIG ---
input_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/xgboost"
plot_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/xgboost/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Diseases to process
diseases = ['Giardiasis', 'chickenpox']

# --- Helper Function for MAPE ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

# --- Create Lagged Features ---
def make_lagged_features(df, target, lags=[1, 2, 3, 4, 5, 6]):
    for lag in lags:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)
    df = df.dropna()
    return df

# --- XGBoost Baseline Function ---
def run_xgboost_baseline(disease, lags=[1, 2, 3, 4, 5, 6]):
    print(f"\n🚀 XGBoost ML Baseline for {disease}")
    train_path = os.path.join(input_dir, f"{disease}_train.csv")
    test_path = os.path.join(input_dir, f"{disease}_test.csv")
    
    # Check if input files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Missing train/test CSV for {disease}")
        return

    # Load and concatenate train+test for lag features
    train = pd.read_csv(train_path, index_col="Date", parse_dates=True)
    test = pd.read_csv(test_path, index_col="Date", parse_dates=True)
    df = pd.concat([train, test])
    df = make_lagged_features(df, disease, lags=lags)

    # Split back to train/test
    train_idx = train.index.intersection(df.index)
    test_idx = test.index.intersection(df.index)
    X_cols = [col for col in df.columns if 'lag' in col]
    X_train, y_train = df.loc[train_idx, X_cols], df.loc[train_idx, disease]
    X_test, y_test = df.loc[test_idx, X_cols], df.loc[test_idx, disease]

    # Fit XGBoost
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"XGBoost {disease} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Save predictions and metrics
    pred_df = pd.DataFrame({"Date": X_test.index, "Actual": y_test, "XGB_Pred": y_pred}).set_index("Date")
    pred_df.to_csv(os.path.join(output_dir, f"{disease}_xgboost_predictions.csv"))
    metrics_df = pd.DataFrame([{"Disease": disease, "RMSE": rmse, "MAE": mae, "MAPE": mape}])
    metrics_df.to_csv(os.path.join(output_dir, f"{disease}_xgboost_metrics.csv"), index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual", color='blue')
    plt.plot(y_test.index, y_pred, label="XGBoost Prediction", color='orange', linestyle='--')
    plt.title(f"XGBoost Baseline - {disease} (RMSE: {rmse:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{disease}_xgboost_baseline_plot.png"))
    plt.close()
    print(f"Plot saved to {plot_dir}/{disease}_xgboost_baseline_plot.png")
    print(f"Predictions saved to {output_dir}/{disease}_xgboost_predictions.csv")
    print(f"Metrics saved to {output_dir}/{disease}_xgboost_metrics.csv")

# --- Run for each disease ---
for disease in diseases:
    run_xgboost_baseline(disease)

print("\n✅ XGBoost baseline forecasting completed for all diseases.")