import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Ensure output directory exists
output_dir = "C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/forecasts"
os.makedirs(output_dir, exist_ok=True)

# Load and clean dataset
try:
    df = pd.read_csv("C:/Users/VARADA S NAIR/OneDrive/Desktop/inten_disease/us_cleaned.csv", parse_dates=['Date'])
    df = df[['Date', 'Lyme disease']]
    
    # Check for duplicate dates
    duplicates = df['Date'].duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate dates. Aggregating by mean.")
        df = df.groupby('Date').mean().reset_index()
    
    # Set index and enforce weekly frequency
    df = df.set_index('Date')
    df = df.asfreq('W-MON', method='ffill')
    
    print("Lyme disease Data Summary:")
    print(df['Lyme disease'].describe())
    nan_count = df['Lyme disease'].isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaNs in Lyme disease. Filling with forward fill.")
        df['Lyme disease'] = df['Lyme disease'].ffill()
    df['Lyme disease'] = df['Lyme disease'].clip(lower=0)
    freq = df.index.freq
    print(f"Inferred Data Frequency: {freq}")
except Exception as e:
    print(f"Data Loading Failed: {e}")
    exit()

# Train-test split
train = df.iloc[:-52]
test = df.iloc[-52:]

# Define future dates globally
future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=7), periods=52, freq=freq or 'W-MON')

# Function to evaluate forecasts
def evaluate_forecast(actual, predicted, model_name):
    predicted = pd.Series(predicted, index=actual.index).ffill().clip(lower=0).values
    if np.any(np.isnan(predicted)):
        raise ValueError(f"{model_name} forecast contains NaNs after cleaning.")
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{model_name} Metrics:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}")
    return mae, rmse

# Store metrics and forecasts
metrics = {}
forecasts = {}
conf_intervals = {}

# --- ARIMA ---
try:
    print("Running ARIMA...")
    train_log = np.log1p(train['Lyme disease'] + 1)
    arima_model = auto_arima(
        train_log, start_p=0, start_q=0, max_p=2, max_q=2,
        d=None, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True,
        stepwise=True, maxiter=50, enforce_stationarity=True, enforce_invertibility=True
    )
    arima_forecast, conf_int = arima_model.predict(n_periods=52, return_conf_int=True)
    arima_forecast = np.expm1(pd.Series(arima_forecast, index=test.index).ffill() - 1).clip(lower=0)
    if np.any(np.isnan(arima_forecast)):
        raise ValueError("ARIMA forecast contains NaNs after reverse transform.")
    arima_conf_df = pd.DataFrame(np.expm1(conf_int - 1), index=test.index, columns=['lower', 'upper']).clip(lower=0)
    arima_mae, arima_rmse = evaluate_forecast(test['Lyme disease'], arima_forecast, "ARIMA")
    arima_future = np.expm1(pd.Series(arima_model.predict(n_periods=52), index=future_dates) - 1).ffill().clip(lower=0)
    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(arima_forecast, label='ARIMA Forecast (Test)')
    plt.plot(arima_future, label='ARIMA Forecast (2020)', linestyle='--')
    plt.fill_between(arima_forecast.index, arima_conf_df['lower'], arima_conf_df['upper'], color='blue', alpha=0.1)
    plt.title('ARIMA Forecast - Lyme Disease')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/arima_lyme_forecast.png")
    plt.close()
    metrics['ARIMA'] = {'MAE': arima_mae, 'RMSE': arima_rmse}
    forecasts['ARIMA'] = arima_forecast
    conf_intervals['ARIMA'] = arima_conf_df
except Exception as e:
    print(f"ARIMA Failed: {e}")

# --- SARIMA ---
try:
    print("Running SARIMA...")
    train_log = np.log1p(train['Lyme disease'] + 1)
    sarima_model = auto_arima(
        train_log, seasonal=True, m=52, start_P=0, start_Q=0, max_P=0, max_Q=0,
        d=None, D=None, trace=True, error_action='ignore', suppress_warnings=True,
        stepwise=True, maxiter=20, enforce_stationarity=True, enforce_invertibility=True
    )
    sarima_fit = SARIMAX(
        train['Lyme disease'], order=sarima_model.order, seasonal_order=sarima_model.seasonal_order,
        freq=freq
    ).fit(disp=False, maxiter=50)
    sarima_forecast = sarima_fit.get_forecast(steps=52)
    sarima_pred = pd.Series(sarima_forecast.predicted_mean, index=test.index).ffill().clip(lower=0)
    if np.any(np.isnan(sarima_pred)):
        raise ValueError("SARIMA forecast contains NaNs after cleaning.")
    sarima_conf = sarima_forecast.conf_int()
    sarima_conf.index = test.index
    sarima_mae, sarima_rmse = evaluate_forecast(test['Lyme disease'], sarima_pred, "SARIMA")
    sarima_future = pd.Series(sarima_fit.get_forecast(steps=104).predicted_mean.tail(52), index=future_dates).ffill().clip(lower=0)
    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(sarima_pred, label='SARIMA Forecast (Test)')
    plt.plot(sarima_future, label='SARIMA Forecast (2020)', linestyle='--')
    plt.fill_between(sarima_pred.index, sarima_conf.iloc[:, 0], sarima_conf.iloc[:, 1], color='green', alpha=0.1)
    plt.title('SARIMA Forecast - Lyme Disease')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sarima_lyme_forecast.png")
    plt.close()
    metrics['SARIMA'] = {'MAE': sarima_mae, 'RMSE': sarima_rmse}
    forecasts['SARIMA'] = sarima_pred
    conf_intervals['SARIMA'] = sarima_conf
except Exception as e:
    print(f"SARIMA Failed: {e}")

# --- Prophet ---
try:
    print("Running Prophet...")
    prophet_df = train.reset_index().rename(columns={'Date': 'ds', 'Lyme disease': 'y'})
    prophet_model = Prophet(yearly_seasonality=10, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=104, freq=freq or 'W-MON')
    prophet_forecast = prophet_model.predict(future)
    prophet_test = prophet_forecast[prophet_forecast['ds'].isin(test.index)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
    prophet_future = prophet_forecast.tail(52)[['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'Lyme disease'})
    prophet_test['yhat'] = prophet_test['yhat'].ffill().clip(lower=0)
    prophet_mae, prophet_rmse = evaluate_forecast(test['Lyme disease'], prophet_test['yhat'], "Prophet")
    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(prophet_test['yhat'], label='Prophet Forecast (Test)')
    plt.plot(prophet_future, label='Prophet Forecast (2020)', linestyle='--')
    plt.fill_between(prophet_test.index, prophet_test['yhat_lower'], prophet_test['yhat_upper'], color='orange', alpha=0.1)
    plt.title('Prophet Forecast - Lyme Disease')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prophet_lyme_forecast.png")
    plt.close()
    metrics['Prophet'] = {'MAE': prophet_mae, 'RMSE': prophet_rmse}
    forecasts['Prophet'] = prophet_test['yhat']
    conf_intervals['Prophet'] = prophet_test[['yhat_lower', 'yhat_upper']]
except Exception as e:
    print(f"Prophet Failed: {e}")

# --- LSTM ---
try:
    print("Running LSTM...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[['Lyme disease']])
    test_scaled = scaler.transform(test[['Lyme disease']])

    def create_sequences(data, lookback=26):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)

    lookback = 26
    X_train, y_train = create_sequences(train_scaled, lookback)
    X_test, y_test = create_sequences(test_scaled, lookback)
    print(f"LSTM Input Shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    lstm_model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50, activation='tanh', return_sequences=True),
        LSTM(50, activation='tanh'),
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    lstm_pred = []
    current_seq = np.concatenate([train_scaled[-lookback:], test_scaled[:1]])[:-1].reshape(1, lookback, 1)
    for _ in range(len(test)):
        pred = lstm_model.predict(current_seq, verbose=0)
        lstm_pred.append(pred[0, 0])
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = pred[0, 0]
    lstm_pred = scaler.inverse_transform(np.array(lstm_pred).reshape(-1, 1)).flatten()
    lstm_pred = pd.Series(np.maximum(lstm_pred, 0), index=test.index).ffill()

    lstm_mae, lstm_rmse = evaluate_forecast(test['Lyme disease'], lstm_pred, "LSTM")

    lstm_future = []
    current_seq = test_scaled[-lookback:].reshape(1, lookback, 1)
    for _ in range(52):
        pred = lstm_model.predict(current_seq, verbose=0)
        lstm_future.append(pred[0, 0])
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = pred[0, 0]
    lstm_future = scaler.inverse_transform(np.array(lstm_future).reshape(-1, 1)).flatten()
    lstm_future = pd.Series(np.maximum(lstm_future, 0), index=future_dates)

    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(lstm_pred, label='LSTM Forecast (Test)')
    plt.plot(lstm_future, label='LSTM Forecast (2020)', linestyle='--')
    plt.title('LSTM Forecast - Lyme Disease')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lstm_lyme_forecast.png")
    plt.close()
    metrics['LSTM'] = {'MAE': lstm_mae, 'RMSE': lstm_rmse}
    forecasts['LSTM'] = lstm_pred
except Exception as e:
    print(f"LSTM Failed: {e}")

# --- GRU ---
try:
    print("Running GRU...")
    gru_model = Sequential([
        Input(shape=(lookback, 1)),
        GRU(50, activation='tanh', return_sequences=True),
        GRU(50, activation='tanh'),
        Dense(1)
    ])
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    gru_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    gru_pred = []
    current_seq = np.concatenate([train_scaled[-lookback:], test_scaled[:1]])[:-1].reshape(1, lookback, 1)
    for _ in range(len(test)):
        pred = gru_model.predict(current_seq, verbose=0)
        gru_pred.append(pred[0, 0])
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = pred[0, 0]
    gru_pred = scaler.inverse_transform(np.array(gru_pred).reshape(-1, 1)).flatten()
    gru_pred = pd.Series(np.maximum(gru_pred, 0), index=test.index).ffill()

    gru_mae, gru_rmse = evaluate_forecast(test['Lyme disease'], gru_pred, "GRU")

    gru_future = []
    current_seq = test_scaled[-lookback:].reshape(1, lookback, 1)
    for _ in range(52):
        pred = gru_model.predict(current_seq, verbose=0)
        gru_future.append(pred[0, 0])
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = pred[0, 0]
    gru_future = scaler.inverse_transform(np.array(gru_future).reshape(-1, 1)).flatten()
    gru_future = pd.Series(np.maximum(gru_future, 0), index=future_dates)

    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(gru_pred, label='GRU Forecast (Test)')
    plt.plot(gru_future, label='GRU Forecast (2020)', linestyle='--')
    plt.title('GRU Forecast - Lyme Disease')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gru_lyme_forecast.png")
    plt.close()
    metrics['GRU'] = {'MAE': gru_mae, 'RMSE': gru_rmse}
    forecasts['GRU'] = gru_pred
except Exception as e:
    print(f"GRU Failed: {e}")

# --- Model Comparison ---
try:
    metrics_df = pd.DataFrame(metrics).T
    print("\n📊 Model Comparison - Lyme Disease:")
    print(metrics_df)
    metrics_df.to_csv(f"{output_dir}/lyme_model_comparison.csv")
    print(f"✅ Metrics saved to: {output_dir}/lyme_model_comparison.csv")

    plt.figure(figsize=(8, 5))
    metrics_df.plot(kind='bar')
    plt.title('Model Comparison - Lyme Disease (MAE and RMSE)')
    plt.ylabel('Error')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lyme_model_comparison_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    for model_name, forecast in forecasts.items():
        plt.plot(forecast, label=f'{model_name}')
    plt.title('Model Forecasts Comparison - Lyme Disease (Test Period)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lyme_forecasts_comparison.png")
    plt.show()
except Exception as e:
    print(f"Comparison Failed: {e}")

# Save confidence intervals
try:
    for model_name, conf_df in conf_intervals.items():
        conf_df.to_csv(f"{output_dir}/{model_name.lower()}_lyme_conf_intervals.csv")
        print(f"✅ {model_name} confidence intervals saved to: {output_dir}/{model_name.lower()}_lyme_conf_intervals.csv")
except Exception as e:
    print(f"Confidence Intervals Saving Failed: {e}")