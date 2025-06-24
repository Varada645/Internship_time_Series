

```
inter_disease/
├── data/
│   └── US_ID (1).csv
├── eda/
│   ├── eda_disease_analysis.py
│   ├── investigate_missing_data.py
│   └── outputs/
│       ├── missing_data_timeline.png
│       ├── missing_by_region.csv
│       └── missing_data_correlation.png
├── preprocess/
│   ├── preprocess.py
│   └── preprocessed_dataset.csv
├── time_series_results/
│   ├── splits/
│   │   ├── Giardiasis_train.csv
│   │   ├── Giardiasis_test.csv
│   │   ├── chickenpox_train.csv
│   │   └── chickenpox_test.csv
│   ├── stationarity/
│   │   ├── Giardiasis_stationarity_results.csv
│   │   └── chickenpox_stationarity_results.csv
│   ├── arima/
│   │   ├── Giardiasis/
│   │   │   ├── arima_results.txt
│   │   │   └── arima_forecast.csv
│   │   └── chickenpox/
│   ├── sarima/
│   ├── lstm/
│   ├── gru/
│   ├── forecasts/
│   │   ├── Giardiasis/
│   │   │   ├── arima_forecast_plot.png
│   │   │   ├── sarima_forecast_plot.png
│   │   │   ├── lstm_forecast_plot.png
│   │   │   ├── gru_forecast_plot.png
│   │   │   └── comparison_plot.png
│   │   └── chickenpox/
│   └── model_comparison_results.csv
├── stationarity_test.py
├── visualize_timeseries.py
├── arima.py
├── sarima.py
├── lstm.py
├── gru.py
├── comparison.py
└── README.md
```
