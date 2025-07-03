import pandas as pd
import numpy as np
import os
from datetime import datetime

# Define paths
DATA_PATH = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/splits"
OUTPUT_DIR = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inter_disease/time_series_results/sarima"

# Specific dates provided (from earlier context)
specific_dates = pd.to_datetime([
    '2017-01-08', '2017-01-15', '2017-01-22', '2017-01-29',
    '2017-02-05', '2017-02-12', '2017-02-19'
])

# Diseases and their SARIMA parameters (from the provided script)
SARIMA_PARAMS = {
    "chickenpox": (2, 1, 1, 1, 1, 1, 52),  # (p,d,q,P,D,Q,m)
    "Giardiasis": (1, 1, 1, 1, 1, 1, 26)
}
diseases = list(SARIMA_PARAMS.keys())

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def get_sarima_summary():
    summary_data = []
    
    for disease in diseases:
        summary_entry = {
            'Disease': disease,
            'Training_Data_Start': 'N/A',
            'Training_Data_End': 'N/A',
            'Training_Observations': 'N/A',
            'Test_Data_Start': 'N/A',
            'Test_Data_End': 'N/A',
            'Test_Observations': 'N/A',
            'Test_Data_Available': 'No',
            'Forecast_Start': 'N/A',
            'Forecast_End': 'N/A',
            'Forecast_Observations': 'N/A',
            'RMSE': 'N/A',
            'MAE': 'N/A',
            'MAPE': 'N/A',
            'Evaluation_Observations': 'N/A',
            'Evaluation_Start': 'N/A',
            'Evaluation_End': 'N/A',
            'SARIMA_Order_p_d_q': 'N/A',
            'SARIMA_Seasonal_Order_P_D_Q_m': 'N/A',
            '2017_Dates_in_Train': 'N/A',
            '2017_Dates_in_Test': 'N/A',
            '2017_Dates_in_Forecast': 'N/A'
        }

        # Load training data
        train_path = os.path.join(DATA_PATH, f"{disease}_train.csv")
        if os.path.exists(train_path):
            try:
                df_train = pd.read_csv(train_path, index_col='Date', parse_dates=True)
                series = df_train[disease]
                summary_entry['Training_Data_Start'] = series.index[0].strftime('%Y-%m-%d')
                summary_entry['Training_Data_End'] = series.index[-1].strftime('%Y-%m-%d')
                summary_entry['Training_Observations'] = len(series)
                summary_entry['2017_Dates_in_Train'] = len(specific_dates[specific_dates.isin(series.index)])
            except Exception as e:
                print(f"❌ Failed to load training data for {disease}: {e}")

        # Load test data
        test_path = os.path.join(DATA_PATH, f"{disease}_test.csv")
        test_series = None
        if os.path.exists(test_path):
            try:
                df_test = pd.read_csv(test_path, index_col='Date', parse_dates=True)
                test_series = df_test[disease]
                summary_entry['Test_Data_Available'] = 'Yes'
                summary_entry['Test_Data_Start'] = test_series.index[0].strftime('%Y-%m-%d')
                summary_entry['Test_Data_End'] = test_series.index[-1].strftime('%Y-%m-%d')
                summary_entry['Test_Observations'] = len(test_series)
                summary_entry['2017_Dates_in_Test'] = len(specific_dates[specific_dates.isin(test_series.index)])
            except Exception as e:
                print(f"❌ Failed to load test data for {disease}: {e}")

        # Load forecast data
        forecast_path = os.path.join(OUTPUT_DIR, f"{disease}_sarima_forecast_extended.csv")
        if os.path.exists(forecast_path):
            try:
                df_forecast = pd.read_csv(forecast_path, index_col='Date', parse_dates=True)
                forecast_series = df_forecast['Forecast']
                summary_entry['Forecast_Start'] = forecast_series.index[0].strftime('%Y-%m-%d')
                summary_entry['Forecast_End'] = forecast_series.index[-1].strftime('%Y-%m-%d')
                summary_entry['Forecast_Observations'] = len(forecast_series)
                summary_entry['2017_Dates_in_Forecast'] = len(specific_dates[specific_dates.isin(forecast_series.index)])
                
                # Calculate evaluation metrics if test data is available
                if test_series is not None:
                    common_dates = test_series.index.intersection(forecast_series.index)
                    if len(common_dates) > 0:
                        test_forecast = forecast_series[common_dates]
                        test_actual = test_series[common_dates]
                        summary_entry['RMSE'] = np.sqrt(mean_squared_error(test_actual, test_forecast))
                        summary_entry['MAE'] = mean_absolute_error(test_actual, test_forecast)
                        summary_entry['MAPE'] = mean_absolute_percentage_error(test_actual, test_forecast)
                        summary_entry['Evaluation_Observations'] = len(common_dates)
                        summary_entry['Evaluation_Start'] = common_dates[0].strftime('%Y-%m-%d')
                        summary_entry['Evaluation_End'] = common_dates[-1].strftime('%Y-%m-%d')
            except Exception as e:
                print(f"❌ Failed to load forecast data for {disease}: {e}")

        # Add SARIMA parameters
        p, d, q, P, D, Q, m = SARIMA_PARAMS[disease]
        summary_entry['SARIMA_Order_p_d_q'] = f"({p},{d},{q})"
        summary_entry['SARIMA_Seasonal_Order_P_D_Q_m'] = f"({P},{D},{Q},{m})"

        summary_data.append(summary_entry)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Print to terminal
    print("\n📋 Summary of SARIMA Model Results:")
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "sarima_summary.csv"), index=False)
    print(f"\n✅ Summary saved to {os.path.join(OUTPUT_DIR, 'sarima_summary.csv')}")

if __name__ == "__main__":
    get_sarima_summary()