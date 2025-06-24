inter_disease/
├── Model/
│ ├── lstm.py
│ ├── gru.py
│ ├── sarima.py
│ ├── sample.py
│ ├── arima.txt
│ ├── arima.py
│ ├── debug.py
│ └── comparison.py
├── main/
│ ├── stationarity.py
│ ├── visualize_timeseries.py
│ ├── disease_trend.py
│ └── disease_trend.txt
├── time_series_results/
│ ├── sarima/
│ │ ├── Giardiasis/
│ │ │ ├── sarima_results.txt
│ │ │ └── sarima_forecast.csv
│ │ └── chickenpox/
│ │ ├── sarima_results.txt
│ │ └── sarima_forecast.csv
│ ├── plots/
│ │ └── stationarity/
│ │ ├── Giardiasis_seasonal_diagnostics.png
│ │ ├── Giardiasis_stationarity_diagnostics.png
│ │ ├── chickenpox_seasonal_diagnostics.png
│ │ └── chickenpox_stationarity_diagnostics.png
│ ├── stationarity/
│ │ ├── Giardiasis_stationarity_results.csv
│ │ └── chickenpox_stationarity_results.csv
│ ├── splits/
│ │ ├── Giardiasis_test.csv
│ │ ├── Giardiasis_train.csv
│ │ ├── chickenpox_test.csv
│ │ └── chickenpox_train.csv
│ ├── lstm/
│ │ ├── Giardiasis/
│ │ │ ├── lstm_results.txt
│ │ │ ├── lstm_forecast.csv
│ │ │ └── lstm_best.pth
│ │ └── chickenpox/
│ │ ├── lstm_results.txt
│ │ ├── lstm_forecast.csv
│ │ └── lstm_best.pth
│ ├── correlation/
│ │ ├── disease_time_series.png
│ │ ├── disease_cross_correlation_max.png
│ │ ├── main.md
│ │ ├── heatmap.md
│ │ ├── disease_correlation_heatmap.png
│ │ ├── cross_correlation.md
│ │ ├── disease_cross_correlation.png
│ │ └── disease_correlation_results.csv
│ ├── arima/
│ │ ├── Giardiasis/
│ │ │ ├── arima_results.txt
│ │ │ └── arima_forecast.csv
│ │ └── chickenpox/
│ │ ├── arima_results.txt
│ │ └── arima_forecast.csv
│ ├── gru/
│ │ ├── Giardiasis/
│ │ │ ├── gru_results.txt
│ │ │ ├── gru_forecast.csv
│ │ │ └── gru_best.pth
│ │ └── chickenpox/
│ │ ├── gru_results.txt
│ │ ├── gru_forecast.csv
│ │ └── gru_best.pth
│ ├── forecasts/
│ │ ├── Giardiasis/
│ │ │ ├── lstm_forecast_plot.png
│ │ │ ├── sarima_forecast_plot.png
│ │ │ ├── gru_forecast_plot.png
│ │ │ ├── arima_forecast_plot.png
│ │ │ └── comparison_plot.png
│ │ └── chickenpox/
│ │ ├── lstm_forecast_plot.png
│ │ ├── sarima_forecast_plot.png
│ │ ├── gru_forecast_plot.png
│ │ ├── arima_forecast_plot.png
│ │ └── comparison_plot.png
│ ├── selected_diseases_time_series.csv
│ ├── model_comparison_results.csv
│ └── disease_trends.png
├── preprocess/
│ ├── Untitled-1.ipynb
│ ├── Untitled-2.ipynb
│ ├── preprocessed_dataset.csv
│ ├── preprocess.py
│ └── preprocess.txt
├── correlation/
│ └── correlation.py
├── .pytest_cache/
│ ├── v/
│ │ └── cache/
│ │ └── nodeids
│ ├── README.md
│ └── CACHEDIR.TAG
├── load/
│ ├── train_test_split.py
│ └── load.py
├── eda/
│ ├── HepatutisC.png
│ ├── investigate_missing_data.py
│ ├── hepatitusa.png
│ ├── hepatitusB.png
│ ├── investigate_missing_data.txt
│ ├── malaria.png
│ ├── syphilis.png
│ ├── Legionelles.png
│ ├── lyme.png
│ ├── dengue.png
│ ├── dengue.txt
│ ├── chickenpox.png
│ ├── chlamydia.png
│ ├── eda.py
│ ├── gonorrhea.png
│ ├── haemophilus.png
│ ├── eda.txt
│ └── giardisis.png
├── tree_structure.txt
├── US_ID (1).csv
├── report.md
└── requirement.txt
