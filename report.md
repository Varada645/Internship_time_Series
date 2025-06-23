
1. Data Preprocessing
preprocess.py
Cleans the raw dataset.
Drops unnecessary columns (e.g., Reporting Area).
Handles missing data (analyzes, imputes with 0 or monthly mean as appropriate).
Aggregates duplicate dates.
Constructs a proper Date column from Year and Week.
Adds time features (Month, WeekOfYear).
Ensures weekly frequency and saves the cleaned data.
preprocess.txt
Text summary of all preprocessing steps, decisions, and justifications.
preprocessed_dataset.csv
The cleaned, imputed, and feature-enriched dataset used for all downstream analysis.
2. Exploratory Data Analysis (EDA)
eda.py
Loads the preprocessed data.
Computes descriptive statistics.
Visualizes time series for each disease.
Checks for missing values and trends.
eda.txt
Written summary of EDA findings: dataset shape, missingness, disease totals, and key insights.
dengue.txt
Focused analysis and notes on the Dengue disease time series.
Disease Plots (e.g., chickenpox.png, eda/dengue.png)
Time series plots for each disease, showing trends and seasonality.
3. Correlation Analysis
correlation.py
Calculates pairwise correlations between diseases.
Generates and saves correlation heatmaps and time series plots.
Outputs CSVs and images for further analysis.
4. Data Loading and Splitting
load.py
Loads the preprocessed dataset for use in modeling and analysis.
train_test_split.py
Splits the data into training and testing sets for each disease.
Ensures reproducibility and proper time series splitting.
5. Trend and Stationarity Analysis
disease_trend.py
Plots and analyzes long-term trends for each disease.
disease_trend.txt
Text summary of observed trends and seasonality.
stationarity.py
Tests each disease time series for stationarity (e.g., using ADF test).
Reports which diseases require differencing for modeling.
6. Time Series Modeling
arima.py
Automates ARIMA modeling for each disease.
Handles log transforms, auto-ARIMA order selection, diagnostics, forecasting, and RMSE calculation.
Saves results and plots for each run.
arima.txt
Documentation of the ARIMA modeling process, including steps, outputs, and summary.
sarima.py
(If used) Implements SARIMA modeling for diseases with seasonality.
gru.py
Implements a GRU (Gated Recurrent Unit) neural network for time series forecasting.
Trains, validates, and saves the best model for each disease.
lstm.py
Implements an LSTM (Long Short-Term Memory) neural network for time series forecasting.
Trains, validates, and saves the best model for each disease.
comparison.py
Compares the performance (e.g., RMSE) of ARIMA, GRU, and LSTM models across diseases.
debug.py
Utility script for checking the integrity and format of train/test CSV splits.
7. Results and Outputs
time_series_results
Contains all results, plots, and model checkpoints.
Example contents:
disease_trends.png: Combined plot of disease trends.
selected_diseases_time_series.csv: CSV of selected disease time series.
arima/, gru/, lstm/: Subfolders with model-specific results and best model weights (e.g., gru_best.pth).
8. Workflow Order
Preprocess raw data (preprocess/preprocess.py).
Explore data (eda/eda.py), generate plots and reports.
Analyze correlations (correlation/correlation.py).
Split data for modeling (load/train_test_split.py).
Analyze trends and stationarity (disease_trend.py, main/stationarity.py).
Model time series with ARIMA (Model/arima.py), GRU (Model/gru.py), and LSTM (Model/lstm.py).
Compare models (Model/comparison.py).
Debug and validate (Model/debug.py).
Review results in time_series_results.
9. Summary
All scripts are modular and well-documented.
All steps from raw data to advanced forecasting and model comparison are covered.
Outputs include cleaned data, EDA reports, correlation matrices, trend plots, model forecasts, and performance metrics.
get the report of this project workflow
