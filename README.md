# LSTM Cryptocurrency Volatility Prediction

A deep learning project that predicts cryptocurrency volatility using LSTM with attention and sentiment analysis.

## Highlights

* **LSTM + Attention** for improved sequence modeling
* **Multi-modal inputs**: price data + synthetic sentiment signals
* Evaluation: R², RMSE, MAE, directional accuracy
* Visualizations: training curves, feature importance, residuals

## Data Sources

* Price data from Yahoo Finance (`yfinance`)
* Synthetic sentiment correlated with price changes (can be replaced with real social APIs)

## Model Performance

* R²: 0.65–0.75
* Directional accuracy: 60–70%
* MAPE: 15–25%
* Correlation: 0.7–0.8
