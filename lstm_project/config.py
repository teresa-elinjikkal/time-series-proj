import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"

for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data config
DATA_CONFIG = {
    'crypto_symbol': 'BTC-USD',
    'period': '2y',  # yfinance period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    'raw_data_path': DATA_DIR / "raw_data.csv",
    'processed_data_path': DATA_DIR / "processed_data.csv",
    'lookback_period': 60,
    'target_column': 'Volatility',
    'test_size': 0.2,
    'validation_split': 0.2,
    'random_state': 42
}

# Feature config
FEATURE_CONFIG = {
    'price_features': [
        'Close', 'Volume', 'Returns', 'SMA_20', 'SMA_50', 
        'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
        'Volume_ratio'
    ],
    'sentiment_features': [
        'sentiment_score', 'sentiment_ma_7', 'sentiment_ma_30',
        'sentiment_volatility', 'news_volume'
    ],
    'technical_indicators': {
        'sma_periods': [20, 50],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'volatility_window': 20
    }
}

# Model architecture config
MODEL_CONFIG = {
    'lstm_layers': [
        {'units': 128, 'return_sequences': True, 'dropout': 0.2, 'recurrent_dropout': 0.2},
        {'units': 64, 'return_sequences': True, 'dropout': 0.2, 'recurrent_dropout': 0.2},
        {'units': 32, 'return_sequences': False, 'dropout': 0.3, 'recurrent_dropout': 0.2}
    ],
    'dense_layers': [
        {'units': 64, 'activation': 'relu', 'dropout': 0.4},
        {'units': 32, 'activation': 'relu', 'dropout': 0.3}
    ],
    'output_activation': 'linear',
    'use_attention': True,
    'use_batch_norm': True
}

# Training config
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'huber_loss',
    'metrics': ['mae', 'mse'],
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 15,
        'restore_best_weights': True
    },
    'lr_scheduler': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 7,
        'min_lr': 1e-6
    }
}

# Evaluation config
EVALUATION_CONFIG = {
    'metrics': ['mse', 'mae', 'r2', 'mape'],
    'plot_predictions': True,
    'plot_residuals': True,
    'plot_training_history': True,
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300
}

# Synthetic data generation config
SYNTHETIC_CONFIG = {
    'days': 730,
    'initial_price': 40000,
    'daily_return_mean': 0.0008,
    'daily_return_std': 0.02,
    'base_volume': 1000000,
    'volume_std': 0.5,
    'sentiment_correlation': 0.4,
    'random_seed': 42
}

# Logging config
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': LOGS_DIR / "lstm_training.log"
}

# Model saving config
MODEL_SAVE_CONFIG = {
    'model_path': MODELS_DIR / "lstm_volatility_model.h5",
    'scaler_path': MODELS_DIR / "scalers.pkl",
    'history_path': MODELS_DIR / "training_history.pkl",
    'config_path': MODELS_DIR / "model_config.json",
    'save_best_only': True,
    'save_weights_only': False
}