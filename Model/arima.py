import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from warnings import catch_warnings, simplefilter
from statsmodels.tools.sm_exceptions import InterpolationWarning
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define paths
input_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/arima"
plot_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/arima/plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Diseases and their seasonal periods
diseases = {
    'Giardiasis': 26,  # Try 52 if 26 fails
    'chickenpox': 52
}

# Specific dates provided
specific_dates = pd.to_datetime([
    '2017-01-08', '2017-01-15', '2017-01-22', '2017-01-29',
    '2017-02-05', '2017-02-12', '2017-02-19'
])

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

def process_disease(disease, m):
    path = f"{input_dir}/{disease}_train.csv"
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return

    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    series = df[disease]

    if series.isna().any():
        print(f"Missing values in {disease}")
        return
    if (series < 0).any():
        print(f"Negative values in {disease}")
        return

    # Debug: Print data range and statistics
    print(f"\n📊 {disease} Training Data:")
    print(f"Start Date: {series.index[0]}")
    print(f"End Date: {series.index[-1]}")
    print(f"Number of Observations: {len(series)}")
    print(f"Mean: {series.mean():.2f}, Std: {series.std():.2f}, Min: {series.min():.2f}, Max: {series.max():.2f}")

    # Plot training data to check seasonality
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Training Data', color='blue')
    plt.title(f"{disease} Training Data")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/{disease}_training_data_plot.png")
    plt.close()
    print(f"Training data plot saved to {plot_dir}/{disease}_training_data_plot.png")

    # Check specific dates in training data
    overlapping_train_dates = specific_dates[specific_dates.isin(series.index)]
    if len(overlapping_train_dates) > 0:
        print(f"Specific 2017 Dates in Training Set: {len(overlapping_train_dates)} found")
        print(series.loc[overlapping_train_dates].to_string())

    results_df, d, D = enhanced_stationarity_test(series, disease, m)
    results_df.to_csv(f"{output_dir}/{disease}_enhanced_stationarity.csv", index=False)

    # ACF/PACF diagnostics
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plot_acf(series.dropna(), ax=plt.gca(), title=f"ACF - {disease}")
    plt.subplot(1, 2, 2)
    plot_pacf(series.dropna(), ax=plt.gca(), title=f"PACF - {disease}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{disease}_acf_pacf.png")
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
    plt.savefig(f"{plot_dir}/{disease}_enhanced_stationarity_plot.png")
    plt.close()

    # Fit auto_arima model with adjusted parameters for Giardiasis
    print(f"\n🔍 Running auto_arima for {disease}...")
    model = None
    try:
        if disease == 'Giardiasis':
            model = auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=1, max_q=1,  # Reduced to minimize memory
                d=1,  # Force non-seasonal differencing
                start_P=1, start_Q=1,  # Encourage seasonal terms
                max_P=1, max_Q=1,
                D=1,  # Force seasonal differencing
                seasonal=True,
                m=m,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
        else:
            model = auto_arima(
                series,
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
            print(f"Using fallback SARIMA(0,1,1)(1,1,1)[{m}] model...")
            try:
                model = SARIMAX(
                    series,
                    order=(0, 1, 1),
                    seasonal_order=(1, 1, 1, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                print(model.summary())
            except Exception as e:
                print(f"❌ Fallback SARIMA model failed for {disease}: {e}")
                return

    # Plot diagnostics
    try:
        model.plot_diagnostics(figsize=(12, 6))
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{disease}_auto_arima_diagnostics.png")
        plt.close()
    except Exception as e:
        print(f"❌ Diagnostics plot failed for {disease}: {e}")

    # Forecast to 2021
    end_date = pd.to_datetime('2021-12-31')
    last_date = series.index[-1]
    if last_date >= end_date:
        print(f"Data for {disease} already extends beyond 2021. No forecast needed.")
        return

    n_periods = ((end_date - last_date).days // 7) + 1
    print(f"Forecasting {n_periods} weeks from {last_date} to {end_date}")

    try:
        if disease == 'Giardiasis' and isinstance(model, SARIMAX):
            forecast = model.forecast(steps=n_periods)
        else:
            forecast = model.predict(n_periods=n_periods)
        forecast = np.clip(forecast, 0, None)
        index = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=n_periods, freq='W-MON')
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
        forecast_df = pd.DataFrame({disease: forecast_series})
        forecast_df.to_csv(f"{output_dir}/{disease}_forecast_2021.csv")
        print(f"Forecast saved to {output_dir}/{disease}_forecast_2021.csv")
    except Exception as e:
        print(f"❌ Forecast failed for {disease}: {e}")
        return

    # Load test data
    test_path = f"{input_dir}/{disease}_test.csv"
    test_series = None
    if os.path.exists(test_path):
        try:
            df_test = pd.read_csv(test_path, index_col='Date', parse_dates=True)
            test_series = df_test[disease]
            print(f"Test Data Loaded: {len(test_series)} observations from {test_series.index[0]} to {test_series.index[-1]}")
            
            # Check specific 2017 dates in test data
            overlapping_test_dates = specific_dates[specific_dates.isin(test_series.index)]
            if len(overlapping_test_dates) > 0:
                print(f"Specific 2017 Dates in Test Set: {len(overlapping_test_dates)} found")
                print(test_series.loc[overlapping_test_dates].to_string())
            else:
                print(f"No specific 2017 dates found in test set. Expected start: 2017-01-09")
        except Exception as e:
            print(f"❌ Failed to load test data for {disease}: {e}")
            test_series = None
    else:
        print(f"No test data found at {test_path}")

    # Calculate RMSE and MAE
    evaluation_results = []
    if test_series is not None:
        common_dates = forecast_series.index.intersection(test_series.index)
        if len(common_dates) > 0:
            forecast_aligned = forecast_series[common_dates]
            test_aligned = test_series[common_dates]
            rmse = np.sqrt(mean_squared_error(test_aligned, forecast_aligned))
            mae = mean_absolute_error(test_aligned, forecast_aligned)
            evaluation_results.append({
                'Disease': disease,
                'RMSE': rmse,
                'MAE': mae,
                'Number_of_Observations': len(common_dates),
                'Evaluation_Start': str(common_dates[0]),
                'Evaluation_End': str(common_dates[-1])
            })
            print(f"Evaluation for {disease}:")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"Evaluated over {len(common_dates)} dates from {common_dates[0]} to {common_dates[-1]}")
        else:
            print(f"No overlapping dates between forecast and test data for {disease}")
    else:
        print(f"No test data available for evaluation of {disease}")

    # Save evaluation results
    if evaluation_results:
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv(f"{output_dir}/{disease}_evaluation_metrics.csv", index=False)
        print(f"Evaluation metrics saved to {output_dir}/{disease}_evaluation_metrics.csv")

    # Plot forecast
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(series, label='Historical', color='blue')
        plt.plot(forecast_series, label='Forecast', linestyle='--', color='red')
        if test_series is not None:
            plt.plot(test_series, label='Test', linestyle=':', color='green')

        # Mark specific 2017 dates
        for date in specific_dates:
            if date in forecast_series.index:
                plt.axvline(x=date, color='purple', linestyle=':', alpha=0.5, label='2017 Dates' if date == specific_dates[0] else '')

        # Set axis limits
        all_dates = series.index
        if test_series is not None:
            all_dates = all_dates.union(test_series.index)
        all_dates = all_dates.union(forecast_series.index)
        plt.xlim([all_dates[0], all_dates[-1]])

        all_values = np.concatenate([
            series.values,
            forecast_series.values,
            test_series.values if test_series is not None else []
        ])
        plt.ylim([0, max(all_values.max() * 1.1, 1)])

        plt.title(f"Forecast to 2021 - {disease} (RMSE: {rmse:.2f}, MAE: {mae:.2f})" if evaluation_results else f"Forecast to 2021 - {disease}")
        plt.xlabel("Date")
        plt.ylabel("Cases")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{disease}_forecast_2021_plot.png")
        plt.close()
        print(f"Forecast plot saved to {plot_dir}/{disease}_forecast_2021_plot.png")
    except Exception as e:
        print(f"❌ Forecast plot failed for {disease}: {e}")

    print(f"\n✅ {disease} Analysis Done")
    print(f"Recommended d: {d}, Recommended D: {D}")

# Run for each disease
for disease, m in diseases.items():
    process_disease(disease, m)

print("✅ All enhanced stationarity + order suggestion + auto_arima + forecast to 2021 completed.")