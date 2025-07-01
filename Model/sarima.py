import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from warnings import catch_warnings, simplefilter
from statsmodels.tools.sm_exceptions import InterpolationWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import shutil

# Define paths
base_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results"
input_dir = f"{base_dir}/splits"
stationarity_dir = f"{base_dir}/stationarity"
output_dir = f"{base_dir}/sarima"
plot_dir = f"{base_dir}/forecasts"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Diseases and their seasonal periods
diseases = {
    'Giardiasis': 52,
    'chickenpox': 52
}

# Specific dates provided
specific_dates = pd.to_datetime([
    '2017-01-08', '2017-01-15', '2017-01-22', '2017-01-29',
    '2017-02-05', '2017-02-12', '2017-02-19'
])

# Forecast horizon
forecast_horizon = 104  # 2 years

# MAPE function to handle zero values
def mean_absolute_percentage_error(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    non_zero_mask = actual != 0
    if non_zero_mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100

# Stationarity tests
def safe_adfuller(series):
    with catch_warnings():
        simplefilter("ignore", InterpolationWarning)
        result = adfuller(series.dropna())
    return result[0], result[1]

def safe_kpss(series):
    with catch_warnings():
        simplefilter("ignore", InterpolationWarning)
        result = kpss(series.dropna(), nlags='auto')
    return result[0], result[1]

def enhanced_stationarity_test(series, name, m=52):
    results = []

    # Original series
    adf_stat, adf_p = safe_adfuller(series)
    kpss_stat, kpss_p = safe_kpss(series)
    results.append({
        'Test': 'Original',
        'ADF_Statistic': adf_stat,
        'ADF_p-value': adf_p,
        'KPSS_Statistic': kpss_stat,
        'KPSS_p-value': kpss_p,
        'Stationary_ADF': adf_p < 0.05,
        'Stationary_KPSS': kpss_p > 0.05
    })

    # First differencing
    diff1 = series.diff().dropna()
    adf_stat, adf_p = safe_adfuller(diff1)
    kpss_stat, kpss_p = safe_kpss(diff1)
    results.append({
        'Test': 'First Difference',
        'ADF_Statistic': adf_stat,
        'ADF_p-value': adf_p,
        'KPSS_Statistic': kpss_stat,
        'KPSS_p-value': kpss_p,
        'Stationary_ADF': adf_p < 0.05,
        'Stationary_KPSS': kpss_p > 0.05
    })

    # Seasonal differencing
    diff_seasonal = series.diff(m).dropna()
    if len(diff_seasonal) > 0:
        adf_stat, adf_p = safe_adfuller(diff_seasonal)
        kpss_stat, kpss_p = safe_kpss(diff_seasonal)
        results.append({
            'Test': f'Seasonal Difference (m={m})',
            'ADF_Statistic': adf_stat,
            'ADF_p-value': adf_p,
            'KPSS_Statistic': kpss_stat,
            'KPSS_p-value': kpss_p,
            'Stationary_ADF': adf_p < 0.05,
            'Stationary_KPSS': kpss_p > 0.05
        })

    # Decomposition residuals
    decomposition = seasonal_decompose(series.dropna(), period=m)
    resid = decomposition.resid.dropna()
    adf_stat, adf_p = safe_adfuller(resid)
    kpss_stat, kpss_p = safe_kpss(resid)
    results.append({
        'Test': 'Residuals',
        'ADF_Statistic': adf_stat,
        'ADF_p-value': adf_p,
        'KPSS_Statistic': kpss_stat,
        'KPSS_p-value': kpss_p,
        'Stationary_ADF': adf_p < 0.05,
        'Stationary_KPSS': kpss_p > 0.05
    })

    # Auto-suggest differencing
    recommended_d = 0 if results[0]['Stationary_ADF'] else 1
    recommended_D = 1 if len(results) > 2 and not results[2]['Stationary_ADF'] else 0

    return pd.DataFrame(results), recommended_d, recommended_D

# Clean up old files
for folder in [output_dir, plot_dir]:
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path) and item not in diseases:
            shutil.rmtree(item_path)

# Process each disease
for disease, m in diseases.items():
    # Create subfolders
    disease_dir = os.path.join(output_dir, disease)
    disease_plot_dir = os.path.join(plot_dir, disease)
    os.makedirs(disease_dir, exist_ok=True)
    os.makedirs(disease_plot_dir, exist_ok=True)

    # Load data
    train_path = f"{input_dir}/{disease}_train.csv"
    test_path = f"{input_dir}/{disease}_test.csv"
    stationarity_path = f"{stationarity_dir}/{disease}_stationarity_results.csv"

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Error: {disease} train/test files missing. Skipping.")
        continue

    train = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    test = pd.read_csv(test_path, index_col='Date', parse_dates=True)
    series = train[disease]

    try:
        # Check data
        if series.isna().any() or test[disease].isna().any():
            raise ValueError("Missing values found")
        if (series < 0).any() or (test[disease] < 0).any():
            raise ValueError("Negative values found")

        # Debug: Print data range and statistics
        print(f"\n📊 {disease} Training Data:")
        print(f"Start Date: {series.index[0]}")
        print(f"End Date: {series.index[-1]}")
        print(f"Number of Observations: {len(series)}")
        print(f"Mean: {series.mean():.2f}, Std: {series.std():.2f}, Min: {series.min():.2f}, Max: {series.max():.2f}")

        # Plot training data
        plt.figure(figsize=(12, 6))
        plt.plot(series, label='Training Data', color='blue')
        plt.title(f"{disease} Training Data")
        plt.xlabel("Date")
        plt.ylabel("Cases")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{disease_plot_dir}/{disease}_training_data_plot.png")
        plt.close()
        print(f"Training data plot saved to {disease_plot_dir}/{disease}_training_data_plot.png")

        # Stationarity tests
        stationarity_results, d, D = enhanced_stationarity_test(series, disease, m)
        stationarity_results.to_csv(f"{disease_dir}/{disease}_enhanced_stationarity.csv", index=False)

        # ACF/PACF diagnostics
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plot_acf(series.dropna(), ax=plt.gca(), title=f"ACF - {disease}")
        plt.subplot(1, 2, 2)
        plot_pacf(series.dropna(), ax=plt.gca(), title=f"PACF - {disease}")
        plt.tight_layout()
        plt.savefig(f"{disease_plot_dir}/{disease}_acf_pacf.png")
        plt.close()

        # Plot original and transformed series
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(series)
        plt.title("Original")
        plt.subplot(2, 2, 2)
        plt.plot(series.diff().dropna())
        plt.title("First Difference")
        plt.subplot(2, 2, 3)
        plt.plot(series.diff(m).dropna())
        plt.title(f"Seasonal Difference (m={m})")
        decomposition = seasonal_decompose(series, period=m)
        plt.subplot(2, 2, 4)
        plt.plot(decomposition.resid)
        plt.title("Residuals from Decomposition")
        plt.tight_layout()
        plt.savefig(f"{disease_plot_dir}/{disease}_enhanced_stationarity_plot.png")
        plt.close()

        # Log transform
        train_data = np.log1p(series)
        test_data = np.log1p(test[disease])

        # Fit SARIMA model
        print(f"\n🔍 Running SARIMA for {disease}...")
        model = None
        try:
            if disease == 'Giardiasis':
                model = auto_arima(
                    train_data,
                    start_p=0, start_q=0,
                    max_p=1, max_q=1,
                    d=d,
                    start_P=1, start_Q=1,
                    max_P=1, max_Q=1,
                    D=D,
                    seasonal=True,
                    m=m,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
            else:
                model = auto_arima(
                    train_data,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2,
                    d=None,
                    start_P=0, start_Q=0,
                    max_P=1, max_Q=1,
                    D=None,
                    seasonal=True,
                    m=m,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
            print(model.summary())
        except Exception as e:
            print(f"❌ auto_arima failed for {disease}: {e}")
            if disease == 'Giardiasis':
                print(f"Using fallback SARIMA(0,{d},1)(1,{D},1)[{m}] model...")
                try:
                    model = SARIMAX(
                        train_data,
                        order=(0, d, 1),
                        seasonal_order=(1, D, 1, m),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                    print(model.summary())
                except Exception as e:
                    print(f"❌ Fallback SARIMA model failed for {disease}: {e}")
                    continue

        # Forecast
        end_date = pd.to_datetime('2021-12-31')
        last_date = series.index[-1]
        if last_date >= end_date:
            print(f"Data for {disease} already extends beyond 2021. No forecast needed.")
            continue

        forecast_steps = len(test) + forecast_horizon
        print(f"Forecasting {forecast_steps} weeks from {last_date} to {end_date}")

        try:
            if isinstance(model, SARIMAX):
                forecast = model.forecast(steps=forecast_steps)
            else:
                forecast = model.predict(n_periods=forecast_steps)
            forecast = np.expm1(np.clip(forecast, 0, None))  # Reverse log transform
            index = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_steps, freq='W-MON')
            forecast_series = pd.Series(forecast, index=index)

            # Debug: Print forecast details
            print(f"Forecast Length: {len(forecast_series)}")
            print(f"Forecast Start: {forecast_series.index[0]}")
            print(f"Forecast End: {forecast_series.index[-1]}")
            print(f"Forecast Values (first 5): {forecast_series.head().to_list()}")

            # Check specific 2017 dates in forecast
            overlapping_forecast_dates = specific_dates[specific_dates.isin(forecast_series.index)]
            if len(overlapping_forecast_dates) > 0:
                print(f"Specific 2017 Dates in Forecast: {len(overlapping_forecast_dates)} found")
                print(forecast_series.loc[overlapping_forecast_dates].to_string())

            # Save forecast
            forecast_df = pd.DataFrame({'Date': forecast_series.index, 'Forecast': forecast_series})
            forecast_df.to_csv(f"{disease_dir}/sarima_forecast.csv", index=False)
            print(f"Forecast saved to {disease_dir}/sarima_forecast.csv")
        except Exception as e:
            print(f"❌ Forecast failed for {disease}: {e}")
            continue

        # Calculate metrics
        evaluation_results = []
        if len(test) > 0:
            common_dates = forecast_series.index.intersection(test.index)
            if len(common_dates) > 0:
                forecast_aligned = forecast_series[common_dates]
                test_aligned = test[disease][common_dates]
                rmse = np.sqrt(mean_squared_error(test_aligned, forecast_aligned))
                mae = mean_absolute_error(test_aligned, forecast_aligned)
                mape = mean_absolute_percentage_error(test_aligned, forecast_aligned)
                evaluation_results.append({
                    'Disease': disease,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'Number_of_Observations': len(common_dates),
                    'Evaluation_Start': str(common_dates[0]),
                    'Evaluation_End': str(common_dates[-1])
                })
                print(f"Evaluation for {disease}:")
                print(f"RMSE: {rmse:.2f}")
                print(f"MAE: {mae:.2f}")
                print(f"MAPE: {mape:.2f}%")
                print(f"Evaluated over {len(common_dates)} dates from {common_dates[0]} to {common_dates[-1]}")
            else:
                print(f"No overlapping dates between forecast and test data for {disease}")
        else:
            print(f"No test data available for evaluation of {disease}")

        # Save evaluation results
        if evaluation_results:
            evaluation_df = pd.DataFrame(evaluation_results)
            evaluation_df.to_csv(f"{disease_dir}/sarima_evaluation_metrics.csv", index=False)
            print(f"Evaluation metrics saved to {disease_dir}/sarima_evaluation_metrics.csv")

        # Plot forecast
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(series[-52:], label='Train (last year)', color='blue')
            plt.plot(test[disease], label='Test', color='green')
            plt.plot(forecast_series[:len(test)], label='Test Forecast', linestyle='--', color='red')
            plt.plot(forecast_series[len(test):], label='Future Forecast', linestyle=':', color='purple')
            for date in specific_dates:
                if date in forecast_series.index:
                    plt.axvline(x=date, color='black', linestyle=':', alpha=0.5, label='2017 Dates' if date == specific_dates[0] else '')
            plt.title(f"{disease} SARIMA Forecast (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)")
            plt.xlabel("Date")
            plt.ylabel("Cases")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{disease_plot_dir}/sarima_forecast_plot.png")
            plt.close()
            print(f"Forecast plot saved to {disease_plot_dir}/sarima_forecast_plot.png")
        except Exception as e:
            print(f"❌ Forecast plot failed for {disease}: {e}")

        print(f"\n✅ {disease} Analysis Done")
        print(f"Recommended d: {d}, Recommended D: {D}")

    except Exception as e:
        print(f"Error for {disease}: {e}")

print("✅ SARIMA forecasts done.")
print(f"Saving forecast to: {output_dir}")