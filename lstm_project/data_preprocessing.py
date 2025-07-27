"""
Data preprocessing module
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from typing import Tuple, Dict, Any
from config import DATA_CONFIG, FEATURE_CONFIG, SYNTHETIC_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for cryptocurrency data.
    """
    
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.sentiment_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.price_data = None
        self.sentiment_data = None
        
    def fetch_crypto_data(self, symbol: str = None, period: str = None) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Yahoo Finance.
        
        Args:
            symbol: Cryptocurrency symbol (default from config)
            period: Time period (default from config)
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol or DATA_CONFIG['crypto_symbol']
        period = period or DATA_CONFIG['period']
        
        logger.info(f"Fetching {symbol} data for period {period}")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning("No data retrieved from Yahoo Finance. Using synthetic data.")
                return self._generate_synthetic_price_data()
            
            logger.info(f"Successfully fetched {len(data)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}. Using synthetic data.")
            return self._generate_synthetic_price_data()
    
    def _generate_synthetic_price_data(self) -> pd.DataFrame:
        """Generate synthetic cryptocurrency data for demonstration."""
        logger.info("Generating synthetic price data")
        np.random.seed(SYNTHETIC_CONFIG['random_seed'])
        
        days = SYNTHETIC_CONFIG['days']
        dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
        
        # Generate realistic price movement (geometric Brownian motion)
        initial_price = SYNTHETIC_CONFIG['initial_price']
        returns = np.random.normal(
            SYNTHETIC_CONFIG['daily_return_mean'], 
            SYNTHETIC_CONFIG['daily_return_std'], 
            days
        )
        
        prices = [initial_price]
        for return_rate in returns[1:]:
            prices.append(prices[-1] * (1 + return_rate))
        
        base_volume = SYNTHETIC_CONFIG['base_volume']
        volume = np.random.lognormal(
            np.log(base_volume), 
            SYNTHETIC_CONFIG['volume_std'], 
            days
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': np.array(prices) * (1 + np.random.normal(0, 0.001, days)),
            'High': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.01, days))),
            'Low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.01, days))),
            'Close': prices,
            'Volume': volume
        }, index=dates)
        
        data['High'] = np.maximum(data['High'], data['Close'])
        data['Low'] = np.minimum(data['Low'], data['Close'])
        
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the dataset.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        logger.info("Calculating technical indicators")
        
        df = data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(
            window=FEATURE_CONFIG['technical_indicators']['volatility_window']
        ).std() * np.sqrt(252)
        
        # Simple Moving Averages
        for period in FEATURE_CONFIG['technical_indicators']['sma_periods']:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        
        df['RSI'] = self._calculate_rsi(
            df['Close'], 
            FEATURE_CONFIG['technical_indicators']['rsi_period']
        )
        
        df['MACD'], df['MACD_signal'] = self._calculate_macd(df['Close'])
        
        df['BB_upper'], df['BB_lower'] = self._calculate_bollinger_bands(df['Close'])
        
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        config = FEATURE_CONFIG['technical_indicators']
        exp1 = prices.ewm(span=config['macd_fast']).mean()
        exp2 = prices.ewm(span=config['macd_slow']).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=config['macd_signal']).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        config = FEATURE_CONFIG['technical_indicators']
        sma = prices.rolling(window=config['bb_period']).mean()
        std = prices.rolling(window=config['bb_period']).std()
        upper_band = sma + (std * config['bb_std'])
        lower_band = sma - (std * config['bb_std'])
        return upper_band, lower_band
    
    def generate_sentiment_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """      
        Args:
            price_data: DataFrame with price data
            
        Returns:
            DataFrame with sentiment features
        """
        logger.info("Generating synthetic sentiment data")
        np.random.seed(SYNTHETIC_CONFIG['random_seed'])
        
        dates = price_data.index
        price_changes = price_data['Returns'].fillna(0)
        
        base_sentiment = np.random.normal(0, 0.3, len(dates))
        
        sentiment_scores = []
        correlation = SYNTHETIC_CONFIG['sentiment_correlation']
        
        for i, (date, price_change) in enumerate(zip(dates, price_changes)):
            if i > 0:
                sentiment = (base_sentiment[i] + 
                           correlation * price_changes.iloc[i-1] + 
                           0.2 * np.random.normal())
            else:
                sentiment = base_sentiment[i]
            sentiment_scores.append(np.clip(sentiment, -1, 1))
        
        sentiment_df = pd.DataFrame({
            'sentiment_score': sentiment_scores,
            'sentiment_ma_7': pd.Series(sentiment_scores).rolling(7).mean(),
            'sentiment_ma_30': pd.Series(sentiment_scores).rolling(30).mean(),
            'sentiment_volatility': pd.Series(sentiment_scores).rolling(14).std(),
            'news_volume': np.random.poisson(50, len(dates)) + 20 * np.abs(price_changes)
        }, index=dates)
        
        return sentiment_df.fillna(method='bfill').fillna(method='ffill')
    
    def create_sequences(self, data: pd.DataFrame, 
                        lookback_period: int = None,
                        target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Combined DataFrame with all features
            lookback_period: Number of time steps to look back
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        lookback_period = lookback_period or DATA_CONFIG['lookback_period']
        target_col = target_col or DATA_CONFIG['target_column']
        
        logger.info(f"Creating sequences with lookback period: {lookback_period}")
        
        # Get feature columns
        price_features = FEATURE_CONFIG['price_features']
        sentiment_features = FEATURE_CONFIG['sentiment_features']
        
        available_price_features = [f for f in price_features if f in data.columns]
        available_sentiment_features = [f for f in sentiment_features if f in data.columns]
        
        if len(available_price_features) != len(price_features):
            missing = set(price_features) - set(available_price_features)
            logger.warning(f"Missing price features: {missing}")
        
        if len(available_sentiment_features) != len(sentiment_features):
            missing = set(sentiment_features) - set(available_sentiment_features)
            logger.warning(f"Missing sentiment features: {missing}")
        
        # Prepare feature data
        price_data = data[available_price_features].fillna(method='ffill').fillna(method='bfill')
        sentiment_data = data[available_sentiment_features].fillna(method='ffill').fillna(method='bfill')
        target_data = data[target_col].fillna(method='ffill').fillna(method='bfill')
        
        price_data_scaled = self.price_scaler.fit_transform(price_data)
        sentiment_data_scaled = self.sentiment_scaler.fit_transform(sentiment_data)
        target_scaled = self.target_scaler.fit_transform(target_data.values.reshape(-1, 1)).flatten()
        
        all_features = np.concatenate([price_data_scaled, sentiment_data_scaled], axis=1)
        
        X, y = [], []
        for i in range(lookback_period, len(all_features)):
            X.append(all_features[i-lookback_period:i])
            y.append(target_scaled[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = None, 
                   random_state: int = None) -> Tuple[np.ndarray, ...]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature sequences
            y: Target values
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or DATA_CONFIG['test_size']
        random_state = random_state or DATA_CONFIG['random_state']
        
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, shuffle=False)
    
    def save_scalers(self, filepath: str = None):
        """Save fitted scalers for later use."""
        filepath = filepath or str(DATA_CONFIG['processed_data_path']).replace('.csv', '_scalers.pkl')
        
        scalers = {
            'price_scaler': self.price_scaler,
            'sentiment_scaler': self.sentiment_scaler,
            'target_scaler': self.target_scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scalers, f)
        
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str = None):
        """Load previously fitted scalers."""
        filepath = filepath or str(DATA_CONFIG['processed_data_path']).replace('.csv', '_scalers.pkl')
        
        with open(filepath, 'rb') as f:
            scalers = pickle.load(f)
        
        self.price_scaler = scalers['price_scaler']
        self.sentiment_scaler = scalers['sentiment_scaler']
        self.target_scaler = scalers['target_scaler']
        
        logger.info(f"Scalers loaded from {filepath}")
    
    def preprocess_pipeline(self) -> Tuple[np.ndarray, ...]:
        """
        Complete preprocessing pipeline.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Fetch and process price data
        raw_data = self.fetch_crypto_data()
        price_data = self.calculate_technical_indicators(raw_data)
        
        sentiment_data = self.generate_sentiment_data(price_data)
        
        combined_data = pd.concat([price_data, sentiment_data], axis=1)
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        combined_data.to_csv(DATA_CONFIG['processed_data_path'])
        logger.info(f"Processed data saved to {DATA_CONFIG['processed_data_path']}")
        
        X, y = self.create_sequences(combined_data)
        
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        self.save_scalers()
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training targets shape: {y_train.shape}")
    print(f"Test targets shape: {y_test.shape}")