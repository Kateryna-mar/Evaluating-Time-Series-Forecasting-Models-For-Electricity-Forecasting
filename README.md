# Evaluating-Time-Series-Forecasting-Models-For-Electricity-Forecasting
Comparative benchmark of ML and Statistical models for daily electricity consumption forecasting using Darts &amp; Optuna.

## Dataset
The dataset is sourced from [Kaggle: Electricity Consumption and Production](https://www.kaggle.com/datasets/stefancomanita/hourly-electricity-consumption-and-production).

## Methodology & Pipeline

1. **Exploratory Data Analysis (EDA):**
   * Additive seasonal decomposition (30-day period) and autocorrelation analysis (ACF/PACF).
   * Augmented Dickey-Fuller (ADF) stationarity testing across consumption and generation features.
2. **Hyperparameter Optimization:**
   * Automated hyperparameter search using **Optuna** minimizing MAPE.
3. **Multi-Horizon Evaluation Engine:**
   * Automated backtesting across multiple forecast horizons (15, 30, 45 days).
   * Runtime profiling separating model fitting (`fit_time`) from inference (`predict_time`).
4. **Residual Diagnostics:**
   * Residual distribution, bias checks, and residual autocorrelation (ACF) tracking to ensure white-noise error behavior.


## Key Findings & Insights
* **Short vs Long Horizons:** Gradient boosting performs best on short-term horizons, while models accounting for global trend retain stability over 45-day horizons.
* **Baseline Comparison** The baseline Linear Regression model yielded moderate results across all horizons, underscoring the necessity of non-linear tree-based or decomposition approaches for complex load dynamics.
* **Horizon Dynamics** Error metrics peaked on the 15-day horizon due to sharp seasonal drops in demand, whereas performance stabilized on longer windows (30 and 45 days).
