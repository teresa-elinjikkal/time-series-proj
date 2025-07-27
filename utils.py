import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# DATA UTILITY FUNCTIONS
# ================================

def validate_data_quality(data: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required columns
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
        'data_types': data.dtypes.to_dict(),
        'duplicate_rows': data.duplicated().sum(),
        'date_range': None,
        'quality_issues': []
    }
    
    if isinstance(data.index, pd.DatetimeIndex):
        quality_report['date_range'] = {
            'start': data.index.min(),
            'end': data.index.max(),
            'total_days': (data.index.max() - data.index.min()).days
        }
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            quality_report['quality_issues'].append(f"Missing required columns: {missing_columns}")
    
    # Check excessive missing values
    high_missing = {col: pct for col, pct in quality_report['missing_percentage'].items() if pct > 20}
    if high_missing:
        quality_report['quality_issues'].append(f"Columns with >20% missing values: {high_missing}")

    # Check constant columns
    constant_columns = []
    for col in data.select_dtypes(include=[np.number]).columns:
        if data[col].nunique() <= 1:
            constant_columns.append(col)
    
    if constant_columns:
        quality_report['quality_issues'].append(f"Constant columns detected: {constant_columns}")
    
    # Check outliers
    outlier_info = {}
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outlier_info[col] = {
            'count': outliers,
            'percentage': (outliers / len(data)) * 100 if len(data) > 0 else 0
        }
    
    quality_report['outliers'] = outlier_info
    
    return quality_report

def print_data_quality_report(quality_report: Dict[str, Any]):
    """Print formatted data quality report."""
    print("📊 DATA QUALITY REPORT")
    print("=" * 50)
    print(f"Total Rows: {quality_report['total_rows']:,}")
    print(f"Total Columns: {quality_report['total_columns']}")
    print(f"Duplicate Rows: {quality_report['duplicate_rows']}")
    
    if quality_report['date_range']:
        print(f"Date Range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
        print(f"Total Days: {quality_report['date_range']['total_days']}")
    
    missing_summary = {k: v for k, v in quality_report['missing_percentage'].items() if v > 0}
    if missing_summary:
        print("\n🔍 MISSING VALUES:")
        for col, pct in missing_summary.items():
            print(f"  {col}: {pct:.1f}%")
    else:
        print("\n✅ No missing values detected")
    
    if quality_report['quality_issues']:
        print("\n⚠️ QUALITY ISSUES:")
        for issue in quality_report['quality_issues']:
            print(f"  • {issue}")
    else:
        print("\n✅ No major quality issues detected")
    
    high_outliers = {k: v for k, v in quality_report['outliers'].items() if v['percentage'] > 5}
    if high_outliers:
        print("\n🎯 COLUMNS WITH >5% OUTLIERS:")
        for col, info in high_outliers.items():
            print(f"  {col}: {info['count']} ({info['percentage']:.1f}%)")

def handle_missing_values(data: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        data: Input DataFrame
        strategy: Strategy for handling missing values
                 ('forward_fill', 'backward_fill', 'interpolate', 'drop', 'mean')
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values using strategy: {strategy}")
    
    data_clean = data.copy()
    
    if strategy == 'forward_fill':
        data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'backward_fill':
        data_clean = data_clean.fillna(method='bfill').fillna(method='ffill')
    elif strategy == 'interpolate':
        data_clean = data_clean.interpolate(method='linear').fillna(method='bfill')
    elif strategy == 'drop':
        data_clean = data_clean.dropna()
    elif strategy == 'mean':
        for col in data_clean.select_dtypes(include=[np.number]).columns:
            data_clean[col].fillna(data_clean[col].mean(), inplace=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    logger.info(f"Missing values handled. Shape: {data.shape} -> {data_clean.shape}")
    
    return data_clean

def detect_outliers(data: pd.DataFrame, columns: List[str] = None, 
                   method: str = 'iqr', threshold: float = 1.5) -> Dict[str, np.ndarray]:
    """
    Detect outliers in numerical columns.
    
    Args:
        data: Input DataFrame
        columns: Columns to check (default: all numerical)
        method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary with outlier indices for each column
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_mask = z_scores > threshold
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_mask = iso_forest.fit_predict(data[[col]].values) == -1
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outliers[col] = data.index[outlier_mask].values
    
    return outliers

# ================================
# VISUALIZATION UTILITIES
# ================================

def plot_correlation_matrix(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                           save_path: str = None, method: str = 'pearson'):
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: Input DataFrame
        figsize: Figure size
        save_path: Path to save the plot
        method: Correlation method ('pearson', 'spearman', 'kendall')
    """
    correlation_matrix = data.corr(method=method)
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title(f'Feature Correlation Matrix ({method.capitalize()})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")
    
    plt.show()

def plot_time_series_with_events(data: pd.DataFrame, y_column: str, 
                                events: Dict[str, datetime] = None,
                                figsize: Tuple[int, int] = (15, 8),
                                save_path: str = None):
    """
    Plot time series with optional event markers.
    
    Args:
        data: DataFrame with datetime index
        y_column: Column to plot
        events: Dictionary of event_name: event_date
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    # Plot main time series
    plt.plot(data.index, data[y_column], linewidth=1.5, alpha=0.8)
    
    # Add event markers
    if events:
        for event_name, event_date in events.items():
            if event_date in data.index:
                plt.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
                plt.text(event_date, data[y_column].max() * 0.9, event_name, 
                        rotation=90, fontsize=10)
    
    plt.title(f'{y_column} Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel(y_column)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time series plot saved to {save_path}")
    
    plt.show()

def plot_distribution_comparison(data1: np.ndarray, data2: np.ndarray,
                               labels: List[str] = ['Data 1', 'Data 2'],
                               title: str = 'Distribution Comparison',
                               figsize: Tuple[int, int] = (12, 6)):
    """
    Plot comparison of two distributions.
    
    Args:
        data1: First dataset
        data2: Second dataset
        labels: Labels for the datasets
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histograms
    axes[0].hist(data1, bins=30, alpha=0.7, label=labels[0], density=True)
    axes[0].hist(data2, bins=30, alpha=0.7, label=labels[1], density=True)
    axes[0].set_title('Histogram Comparison')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plots
    box_data = [data1, data2]
    axes[1].boxplot(box_data, labels=labels)
    axes[1].set_title('Box Plot Comparison')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ================================
# PERFORMANCE UTILITIES
# ================================

def calculate_portfolio_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        returns: Portfolio returns series
        benchmark_returns: Benchmark returns for comparison
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
    metrics['volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
    
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = metrics['annualized_return'] / metrics['downside_deviation']
    else:
        metrics['downside_deviation'] = 0
        metrics['sortino_ratio'] = float('inf')
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    if abs(metrics['max_drawdown']) > 0:
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = float('inf')
    
    metrics['win_rate'] = (returns > 0).mean()
    
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    
    metrics['avg_win'] = winning_returns.mean() if len(winning_returns) > 0 else 0
    metrics['avg_loss'] = losing_returns.mean() if len(losing_returns) > 0 else 0
    
    if metrics['avg_loss'] != 0:
        metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
    else:
        metrics['win_loss_ratio'] = float('inf')
    
    if benchmark_returns is not None:
        benchmark_total = (1 + benchmark_returns).prod() - 1
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        metrics['excess_return'] = metrics['total_return'] - benchmark_total
        metrics['information_ratio'] = (returns - benchmark_returns).mean() / (returns - benchmark_returns).std()
        metrics['tracking_error'] = (returns - benchmark_returns).std() * np.sqrt(252)
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        risk_free_rate = 0.02
        benchmark_annual = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
        metrics['alpha'] = metrics['annualized_return'] - (risk_free_rate + metrics['beta'] * (benchmark_annual - risk_free_rate))
    
    return metrics

def print_performance_summary(metrics: Dict[str, float]):
    """Print formatted performance summary."""
    print("PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"Total Return:        {metrics['total_return']:.2%}")
    print(f"Annualized Return:   {metrics['annualized_return']:.2%}")
    print(f"Volatility:          {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:.3f}")
    print(f"Win Rate:            {metrics['win_rate']:.2%}")
    print(f"Win/Loss Ratio:      {metrics['win_loss_ratio']:.3f}")
    
    if 'alpha' in metrics:
        print(f"Alpha:               {metrics['alpha']:.2%}")
        print(f"Beta:                {metrics['beta']:.3f}")
        print(f"Information Ratio:   {metrics['information_ratio']:.3f}")

# ================================
# MODEL UTILITIES
# ================================

def save_model_artifacts(model, scaler, config: Dict[str, Any], base_path: str):
    """
    Save all model artifacts for reproducibility.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        config: Model configuration
        base_path: Base path for saving artifacts
    """
    import pickle
    import json
    from pathlib import Path
    
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)

    model_path = base_path / "model.h5"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    

    scaler_path = base_path / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    

    config_path = base_path / "config.json"
    config['timestamp'] = datetime.now().isoformat()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Configuration saved to {config_path}")

def load_model_artifacts(base_path: str):
    """
    Load all model artifacts.
    
    Args:
        base_path: Base path containing artifacts
        
    Returns:
        Tuple of (model, scaler, config)
    """
    import pickle
    import json
    from pathlib import Path
    import tensorflow as tf
    
    base_path = Path(base_path)

    model_path = base_path / "model.h5"
    model = tf.keras.models.load_model(str(model_path))
    
    scaler_path = base_path / "scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    config_path = base_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Model artifacts loaded from {base_path}")
    
    return model, scaler, config

def compare_models(results_list: List[Dict[str, Any]], 
                   model_names: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results_list: List of evaluation results dictionaries
        model_names: Names for the models
        
    Returns:
        DataFrame with comparison metrics
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(results_list))]
    
    comparison_data = []
    
    for i, results in enumerate(results_list):
        metrics = results.get('metrics', {})
        row = {
            'Model': model_names[i],
            'R²': metrics.get('r2', np.nan),
            'RMSE': metrics.get('rmse', np.nan),
            'MAE': metrics.get('mae', np.nan),
            'MAPE': metrics.get('mape', np.nan),
            'Dir_Acc': metrics.get('directional_accuracy', np.nan),
            'Correlation': metrics.get('correlation', np.nan),
            'Theil_U': metrics.get('theil_u', np.nan)
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Model')
    
    return comparison_df

def print_model_comparison(comparison_df: pd.DataFrame):
    """Print formatted model comparison table."""
    print("🏆 MODEL COMPARISON")
    print("=" * 80)
    formatted_df = comparison_df.copy()
    
    for col in formatted_df.columns:
        if formatted_df[col].dtype in ['float64', 'float32']:
            if col in ['R²', 'Correlation']:
                formatted_df[col] = formatted_df[col].round(4)
            elif col in ['RMSE', 'MAE']:
                formatted_df[col] = formatted_df[col].round(6)
            elif col in ['MAPE', 'Dir_Acc']:
                formatted_df[col] = formatted_df[col].round(2)
            else:
                formatted_df[col] = formatted_df[col].round(4)
    
    print(formatted_df.to_string())
    
    print("\n🥇 BEST PERFORMERS:")
    for metric in ['R²', 'Dir_Acc', 'Correlation']:
        if metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmax()
            best_value = comparison_df.loc[best_model, metric]
            print(f"  {metric}: {best_model} ({best_value:.4f})")
    
    for metric in ['RMSE', 'MAE', 'MAPE', 'Theil_U']:
        if metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmin()
            best_value = comparison_df.loc[best_model, metric]
            print(f"  {metric}: {best_model} ({best_value:.6f})")

# ================================
# CRYPTO-SPECIFIC UTILITIES
# ================================

def calculate_crypto_metrics(prices: pd.Series, returns: pd.Series = None) -> Dict[str, float]:
    """
    Calculate cryptocurrency-specific metrics.
    
    Args:
        prices: Price series
        returns: Returns series (computed if None)
        
    Returns:
        Dictionary of crypto metrics
    """
    if returns is None:
        returns = prices.pct_change().dropna()
    
    metrics = {}
    
    metrics['realized_volatility'] = returns.std() * np.sqrt(252)
    metrics['garch_volatility'] = returns.rolling(30).std().mean() * np.sqrt(252)
    
    metrics['max_single_day_gain'] = returns.max()
    metrics['max_single_day_loss'] = returns.min()
    metrics['days_with_large_moves'] = ((abs(returns) > 0.1).sum() / len(returns)) * 100
    
    sma_20 = prices.rolling(20).mean()
    sma_50 = prices.rolling(50).mean()
    
    metrics['trend_strength'] = ((prices > sma_20).sum() / len(prices)) * 100
    metrics['bull_market_days'] = ((sma_20 > sma_50).sum() / len(sma_20.dropna())) * 100
    
    metrics['hurst_exponent'] = calculate_hurst_exponent(prices.values)
    
    return metrics

def calculate_hurst_exponent(price_series: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent for market efficiency analysis.
    
    Args:
        price_series: Array of prices
        max_lag: Maximum lag to consider
        
    Returns:
        Hurst exponent value
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(price_series[lag:], price_series[:-lag])) for lag in lags]
    
    tau = [max(t, 1e-10) for t in tau]
    lags = [max(l, 1e-10) for l in lags]
    
    poly_coef = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst_exponent = poly_coef[0]
    
    return hurst_exponent

def analyze_volatility_regimes(volatility: pd.Series, 
                              threshold_percentiles: Tuple[float, float] = (25, 75)) -> Dict[str, Any]:
    """
    Analyze volatility regimes in the data.
    
    Args:
        volatility: Volatility series
        threshold_percentiles: Percentiles for low/high volatility thresholds
        
    Returns:
        Dictionary with regime analysis
    """
    low_threshold = np.percentile(volatility, threshold_percentiles[0])
    high_threshold = np.percentile(volatility, threshold_percentiles[1])
    
    regimes = pd.Series(index=volatility.index, dtype='category')
    regimes[volatility <= low_threshold] = 'Low'
    regimes[(volatility > low_threshold) & (volatility < high_threshold)] = 'Medium'
    regimes[volatility >= high_threshold] = 'High'
    
    analysis = {
        'thresholds': {
            'low': low_threshold,
            'high': high_threshold
        },
        'regime_distribution': regimes.value_counts(normalize=True).to_dict(),
        'regime_durations': {},
        'transitions': {}
    }
    
    for regime in ['Low', 'Medium', 'High']:
        regime_periods = (regimes == regime)
        if regime_periods.any():
            regime_groups = (regime_periods != regime_periods.shift()).cumsum()[regime_periods]
            durations = regime_groups.value_counts()
            analysis['regime_durations'][regime] = durations.mean() if len(durations) > 0 else 0
    
    for from_regime in ['Low', 'Medium', 'High']:
        for to_regime in ['Low', 'Medium', 'High']:
            transitions = ((regimes.shift(1) == from_regime) & (regimes == to_regime)).sum()
            total_from = (regimes.shift(1) == from_regime).sum()
            
            transition_key = f"{from_regime}_to_{to_regime}"
            analysis['transitions'][transition_key] = transitions / total_from if total_from > 0 else 0
    
    return analysis

# ================================
# UTILITY FUNCTIONS
# ================================

def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Log file path (optional)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def create_project_structure(base_path: str = "lstm_crypto_project"):
    """
    Create the complete project directory structure.
    
    Args:
        base_path: Base project directory
    """
    from pathlib import Path
    
    base_path = Path(base_path)
    
    directories = [
        'data',
        'notebooks', 
        'lstm_project',
        'models',
        'logs',
        'plots',
        'results'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    (base_path / 'lstm_project' / '__init__.py').touch()
    
    requirements = [
        'tensorflow>=2.8.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
        'yfinance>=0.1.70',
        'jupyter>=1.0.0',
        'plotly>=5.0.0'
    ]
    
    with open(base_path / 'requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    logger.info(f"Project structure created at {base_path}")

def get_system_info() -> Dict[str, Any]:
    """Get system information for reproducibility."""
    import platform
    import tensorflow as tf
    
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.experimental.list_physical_devices('GPU')) > 0,
        'timestamp': datetime.now().isoformat()
    }
    
    return info

if __name__ == "__main__":
    print("Testing utility functions...")
    
    test_data = pd.DataFrame({
        'price': [100, 101, 102, np.nan, 104],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'returns': [0, 0.01, 0.0099, -0.02, 0.02]
    })
    
    quality_report = validate_data_quality(test_data)
    print_data_quality_report(quality_report)
    
    print("\nUtility functions test completed!")