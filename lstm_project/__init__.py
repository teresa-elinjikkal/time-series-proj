"""
LSTM Cryptocurrency Volatility Prediction Package

Author: Teresa Elinjikkal
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Teresa Elinjikkal"
__email__ = "teresa.work25@gmail.com"

# Import main classes for easy access
from .data_preprocessing import DataPreprocessor
from .model import VolatilityLSTM, ModelBuilder, AttentionLayer
from .train import ModelTrainer
from .evaluate import ModelEvaluator
from .utils import (
    validate_data_quality,
    plot_correlation_matrix,
    calculate_portfolio_metrics,
    compare_models
)

# Package metadata
__all__ = [
    'DataPreprocessor',
    'VolatilityLSTM', 
    'ModelBuilder',
    'AttentionLayer',
    'ModelTrainer',
    'ModelEvaluator',
    'validate_data_quality',
    'plot_correlation_matrix',
    'calculate_portfolio_metrics',
    'compare_models'
]

# Package information
PACKAGE_INFO = {
    'name': 'lstm_crypto_volatility',
    'version': __version__,
    'description': 'LSTM-based cryptocurrency volatility prediction with sentiment analysis',
    'author': __author__,
    'email': __email__,
    'url': 'https://github.com/yourusername/lstm-crypto-volatility',
    'license': 'MIT',
    'keywords': ['lstm', 'cryptocurrency', 'volatility', 'prediction', 'deep-learning', 'fintech'],
    'python_requires': '>=3.7',
    'dependencies': [
        'tensorflow>=2.8.0',
        'numpy>=1.21.0', 
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'yfinance>=0.1.70'
    ]
}

def get_version():
    """Return the package version."""
    return __version__

def get_package_info():
    """Return package information dictionary."""
    return PACKAGE_INFO.copy()

def print_package_info():
    """Print formatted package information."""
    print(f"""
{PACKAGE_INFO['name']} v{PACKAGE_INFO['version']}
{'=' * 60}
Description: {PACKAGE_INFO['description']}
Author: {PACKAGE_INFO['author']} ({PACKAGE_INFO['email']})
URL: {PACKAGE_INFO['url']}
License: {PACKAGE_INFO['license']}
Python: {PACKAGE_INFO['python_requires']}

Keywords: {', '.join(PACKAGE_INFO['keywords'])}

Main Components:
  • DataPreprocessor: Data cleaning and feature engineering
  • VolatilityLSTM: Advanced LSTM model architecture  
  • ModelTrainer: Training pipeline with callbacks
  • ModelEvaluator: Comprehensive evaluation and analysis
  • Utilities: Helper functions and visualization tools


{'=' * 60}
    """)

# Configuration validation
def _validate_imports():
    """Validate that all required dependencies are available."""
    missing_deps = []
    
    try:
        import tensorflow
        if tuple(map(int, tensorflow.__version__.split('.')[:2])) < (2, 8):
            missing_deps.append(f"tensorflow>={PACKAGE_INFO['dependencies'][0].split('>=')[1]} (found {tensorflow.__version__})")
    except ImportError:
        missing_deps.append("tensorflow>=2.8.0")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy>=1.21.0")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas>=1.3.0")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn>=1.0.0")
    
    if missing_deps:
        import warnings
        warnings.warn(
            f"Missing dependencies: {', '.join(missing_deps)}. "
            f"Install with: pip install {' '.join(missing_deps)}",
            ImportWarning
        )
    
    return len(missing_deps) == 0

# Run validation on import
_validate_imports()

# Package initialization complete
import logging
logger = logging.getLogger(__name__)
logger.info(f"LSTM Crypto Volatility package v{__version__} imported successfully")