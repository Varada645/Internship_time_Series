import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
DATA_PATH = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
OUTPUT_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/sarima"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# --- PARAMETERS ---
SARIMA_PARAMS = {
    "chickenpox": (2, 1, 1, 1, 1, 1, 52),  # (p,d,q,P,D,Q,m)
    "Giardiasis": (1, 1, 1, 1, 1, 1, 26)
}

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def forecast_with_sarima(disease):
    print(f"\n📈 Running SARIMA for {disease}")
    train_path = os.path.join(DATA_PATH, f"{disease}_train.csv")
    test_path = os.path.join(DATA_PATH, f"{disease}_test.csv")

    # Load data
    train = pd.read_csv(train_path, index_col="Date", parse_dates=True)
    test = pd.read_csv(test_path, index_col="Date", parse_dates=True)
    series = train[disease]

    # SARIMA order
    p, d, q, P, D, Q, m = SARIMA_PARAMS[disease]
    order = (p, d, q)
    seasonal_order = (P, D, Q, m)

    # Fit model
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    print(results.summary())

    # Forecast for test + future (to 2022)
    last_test_date = test.index[-1]
    end_date = pd.to_datetime("2022-12-31")
    steps = ((end_date - last_test_date).days // 7)
    steps = max(steps, 1)
    total_steps = len(test) + steps

    # Prepare date index for forecast
    all_dates = test.index.tolist()
    last_date = all_dates[-1]
    for i in range(1, steps + 1):
        all_dates.append(last_date + pd.Timedelta(weeks=i))

    forecast = results.forecast(steps=total_steps)
    forecast = np.clip(forecast, 0, None)

    # Evaluation on test set only
    test_forecast = forecast[:len(test)]
    rmse = np.sqrt(mean_squared_error(test[disease], test_forecast))
    mae = mean_absolute_error(test[disease], test_forecast)
    mape = mean_absolute_percentage_error(test[disease], test_forecast)
    print(f"✅ {disease} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Save extended forecast
    forecast_df = pd.DataFrame({"Date": all_dates, "Forecast": forecast}).set_index("Date")
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, f"{disease}_sarima_forecast_extended.csv"))

    # Save evaluation metrics
    eval_metrics = pd.DataFrame({
        "Disease": [disease],
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE (%)": [mape],
        "Test_Start": [test.index[0]],
        "Test_End": [test.index[-1]],
        "Forecast_End": [forecast_df.index[-1]],
        "Test_Observations": [len(test)],
        "Forecast_Observations": [len(forecast_df)]
    })
    eval_path = os.path.join(OUTPUT_DIR, f"{disease}_sarima_evaluation_metrics.csv")
    eval_metrics.to_csv(eval_path, index=False)
    print(f"📁 Evaluation metrics saved to {eval_path}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train[disease], label="Train", color='blue')
    plt.plot(test[disease], label="Test", color='green')
    plt.plot(forecast_df.index, forecast_df["Forecast"], label="SARIMA Forecast (Test+Future)", linestyle='--', color='red')
    plt.title(f"SARIMA Forecast - {disease} (RMSE: {rmse:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{disease}_sarima_forecast_plot_extended.png"))
    plt.close()

# --- Run for both diseases ---
forecast_with_sarima("chickenpox")
forecast_with_sarima("Giardiasis")

print("\n📊 SARIMA forecasting completed for all diseases.")
