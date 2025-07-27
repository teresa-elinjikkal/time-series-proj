import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from model import ModelBuilder
from config import EVALUATION_CONFIG, PLOTS_DIR, MODEL_SAVE_CONFIG, DATA_CONFIG


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(MODEL_SAVE_CONFIG['model_path'])
        self.model = None
        self.preprocessor = None
        self.evaluation_results = {}
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_model_and_preprocessor(self):
        """Load trained model and preprocessor."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = ModelBuilder.load_pretrained_model(self.model_path)
            logger.info("Model loaded successfully")
            
            self.preprocessor = DataPreprocessor()
            scaler_path = str(MODEL_SAVE_CONFIG['scaler_path'])
            self.preprocessor.load_scalers(scaler_path)
            logger.info("Preprocessor loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model/preprocessor: {e}")
            raise
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure arrays are 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE
        mape_mask = y_true != 0
        if np.sum(mape_mask) > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
        else:
            metrics['mape'] = float('inf')
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction) * 100
        else:
            metrics['directional_accuracy'] = 0.0
        
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        
        # Correlation coefficient
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            metrics['correlation'] = 0.0
        
        # Theil's U statistic
        naive_forecast = np.roll(y_true, 1)[1:]  # Previous value as forecast
        actual_series = y_true[1:]
        predicted_series = y_pred[1:]
        
        if len(actual_series) > 0:
            mse_model = np.mean((actual_series - predicted_series) ** 2)
            mse_naive = np.mean((actual_series - naive_forecast) ** 2)
            
            if mse_naive > 0:
                metrics['theil_u'] = np.sqrt(mse_model) / np.sqrt(mse_naive)
            else:
                metrics['theil_u'] = float('inf')
        else:
            metrics['theil_u'] = float('inf')
        
        return metrics
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_preprocessor() first.")
        
        logger.info("Evaluating model performance")
        
        predictions = self.model.predict(X_test, verbose=0)
        predictions = predictions.flatten()
        
        if hasattr(self.preprocessor, 'target_scaler'):
            try:
                y_test_original = self.preprocessor.target_scaler.inverse_transform(
                    y_test.reshape(-1, 1)
                ).flatten()
                predictions_original = self.preprocessor.target_scaler.inverse_transform(
                    predictions.reshape(-1, 1)
                ).flatten()
            except:
                logger.warning("Could not inverse transform. Using scaled values.")
                y_test_original = y_test
                predictions_original = predictions
        else:
            y_test_original = y_test
            predictions_original = predictions
        
        metrics = self.calculate_metrics(y_test_original, predictions_original)
        
        self.evaluation_results = {
            'predictions': predictions_original,
            'actual': y_test_original,
            'predictions_scaled': predictions,
            'actual_scaled': y_test,
            'metrics': metrics,
            'n_samples': len(y_test),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Model Performance Metrics:")
        logger.info(f"  R² Score: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        return self.evaluation_results
    
    def plot_predictions(self, save_plot: bool = True, figsize: tuple[int, int] = (15, 10)):
        """
        Create comprehensive prediction visualization plots.
        
        Args:
            save_plot: Whether to save plots to disk
            figsize: Figure size for plots
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        logger.info("Creating prediction visualizations")
        
        actual = self.evaluation_results['actual']
        predictions = self.evaluation_results['predictions']
        metrics = self.evaluation_results['metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('LSTM Cryptocurrency Volatility Prediction Results', fontsize=16, fontweight='bold')
        
        time_range = range(len(actual))
        axes[0, 0].plot(time_range, actual, label='Actual', alpha=0.8, linewidth=1.5)
        axes[0, 0].plot(time_range, predictions, label='Predicted', alpha=0.8, linewidth=1.5)
        axes[0, 0].set_title(f'Volatility Prediction Over Time (R² = {metrics["r2"]:.3f})')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Volatility')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot: Predictions vs Actual
        axes[0, 1].scatter(actual, predictions, alpha=0.6, s=30)
        
        min_val, max_val = min(actual.min(), predictions.min()), max(actual.max(), predictions.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[0, 1].set_title('Predictions vs Actual Values')
        axes[0, 1].set_xlabel('Actual Volatility')
        axes[0, 1].set_ylabel('Predicted Volatility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 1].text(0.05, 0.95, f'Correlation: {metrics["correlation"]:.3f}', 
                       transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 3. Residuals analysis
        residuals = actual - predictions
        axes[1, 0].scatter(predictions, residuals, alpha=0.6, s=30)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('Residuals vs Predicted Values')
        axes[1, 0].set_xlabel('Predicted Volatility')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black', density=True)
        axes[1, 1].axvline(residuals.mean(), color='red', linestyle='--', 
                          label=f'Mean: {residuals.mean():.4f}')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residual Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = PLOTS_DIR / f'prediction_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
            logger.info(f"Prediction plot saved to {plot_path}")
        
        plt.show()
    
    def plot_training_history(self, history_path: str = None, save_plot: bool = True):
        """
        Plot training history curves.
        
        Args:
            history_path: Path to training history file
            save_plot: Whether to save plot to disk
        """
        history_path = history_path or str(MODEL_SAVE_CONFIG['history_path'])
        
        try:
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            
            logger.info("Plotting training history")
            
            if isinstance(history, list):
                history = self._average_ensemble_history(history)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('Training History', fontsize=16, fontweight='bold')
            
            axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0].set_title('Model Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            if 'mae' in history:
                axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
                if 'val_mae' in history:
                    axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
                axes[1].set_title('Mean Absolute Error')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = PLOTS_DIR / f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(plot_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
                logger.info(f"Training history plot saved to {plot_path}")
            
            plt.show()
            
        except FileNotFoundError:
            logger.warning(f"Training history file not found: {history_path}")
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
    
    def _average_ensemble_history(self, histories: list[Dict]) -> Dict:
        """Average training histories from ensemble models."""
        if not histories:
            return {}
        
        keys = histories[0].keys()
        averaged_history = {}
        
        for key in keys:
            values = [hist[key] for hist in histories if key in hist]
            if values:
                averaged_history[key] = np.mean(values, axis=0).tolist()
        
        return averaged_history
    
    def plot_feature_importance(self, X_test: np.ndarray, n_samples: int = 100):
        """
        Analyze feature importance using permutation importance.
        
        Args:
            X_test: Test sequences
            n_samples: Number of samples to use for analysis
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        logger.info("Analyzing feature importance")
        
        if len(X_test) > n_samples:
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test[indices]
        else:
            X_sample = X_test
        
        baseline_pred = self.model.predict(X_sample, verbose=0)
        baseline_mse = np.mean((baseline_pred - baseline_pred.mean()) ** 2)
        
        price_features = ['Close', 'Volume', 'Returns', 'SMA_20', 'SMA_50', 
                         'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Volume_ratio']
        sentiment_features = ['Sentiment', 'Sent_MA7', 'Sent_MA30', 'Sent_Vol', 'News_Vol']
        feature_names = price_features + sentiment_features
        
        importance_scores = []
        
        for i in range(X_sample.shape[2]):
            X_permuted = X_sample.copy()
            
            for t in range(X_sample.shape[1]):
                np.random.shuffle(X_permuted[:, t, i])
            
            permuted_pred = self.model.predict(X_permuted, verbose=0)
            permuted_mse = np.mean((permuted_pred - baseline_pred) ** 2)
            
            importance = permuted_mse - baseline_mse
            importance_scores.append(importance)
        
        plt.figure(figsize=(12, 8))
        feature_names_plot = feature_names[:len(importance_scores)]
        
        bars = plt.barh(range(len(importance_scores)), importance_scores)
        plt.yticks(range(len(importance_scores)), feature_names_plot)
        plt.xlabel('Importance Score (MSE Increase)')
        plt.title('Feature Importance Analysis')
        plt.grid(True, alpha=0.3)
        
        colors = ['red' if score > np.mean(importance_scores) else 'blue' for score in importance_scores]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        plot_path = PLOTS_DIR / f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {plot_path}")
        
        plt.show()
        
        return dict(zip(feature_names_plot, importance_scores))
    
    def generate_evaluation_report(self) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Formatted evaluation report string
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        metrics = self.evaluation_results['metrics']
        
        report = f"""
CRYPTOCURRENCY VOLATILITY PREDICTION - EVALUATION REPORT
{'=' * 70}

PERFORMANCE METRICS:
  • R² Score:              {metrics['r2']:.4f}
  • Root Mean Square Error: {metrics['rmse']:.6f}
  • Mean Absolute Error:    {metrics['mae']:.6f}
  • Mean Abs. Percentage:   {metrics['mape']:.2f}%
  • Max Error:              {metrics['max_error']:.6f}
  • Directional Accuracy:   {metrics['directional_accuracy']:.2f}%

STATISTICAL ANALYSIS:
  • Correlation:            {metrics['correlation']:.4f}
  • Mean Residual:          {metrics['mean_residual']:.6f}
  • Residual Std Dev:       {metrics['std_residual']:.6f}
  • Theil's U Statistic:    {metrics['theil_u']:.4f}

MODEL INSIGHTS:
  • Total Test Samples:     {self.evaluation_results['n_samples']:,}
  • Evaluation Date:        {self.evaluation_results['evaluation_timestamp']}
  • Model Path:            {self.model_path}

PERFORMANCE INTERPRETATION:
"""
        if metrics['r2'] > 0.7:
            report += "  EXCELLENT: High predictive accuracy (R² > 0.7)\n"
        elif metrics['r2'] > 0.5:
            report += "  GOOD: Moderate predictive accuracy (R² > 0.5)\n"
        elif metrics['r2'] > 0.3:
            report += "  FAIR: Limited predictive accuracy (R² > 0.3)\n"
        else:
            report += "  POOR: Low predictive accuracy (R² ≤ 0.3)\n"
        
        if metrics['directional_accuracy'] > 60:
            report += "  Strong directional prediction capability\n"
        elif metrics['directional_accuracy'] > 50:
            report += "  Moderate directional prediction capability\n"
        else:
            report += "  Weak directional prediction capability\n"
        
        if metrics['theil_u'] < 1.0:
            report += "  Outperforms naive forecast\n"
        else:
            report += "  Underperforms naive forecast\n"
        
        report += f"\n{'=' * 70}"
        
        return report
    
    def run_complete_evaluation(self, X_test: np.ndarray = None, y_test: np.ndarray = None) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            X_test: Test sequences (if None, will load from preprocessor)
            y_test: Test targets (if None, will load from preprocessor)
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting complete evaluation pipeline")
        
        self.load_model_and_preprocessor()
        
        if X_test is None or y_test is None:
            logger.info("Loading test data from preprocessor")
            X_train, X_test, y_train, y_test = self.preprocessor.preprocess_pipeline()
        
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        self.plot_predictions()
        self.plot_training_history()
        
        try:
            feature_importance = self.plot_feature_importance(X_test)
            evaluation_results['feature_importance'] = feature_importance
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {e}")
        
        report = self.generate_evaluation_report()
        print(report)
        
        logger.info("Complete evaluation pipeline finished")
        
        return evaluation_results

def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate LSTM volatility prediction model')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--no_plots', action='store_true', help='Skip plotting')
    parser.add_argument('--report_only', action='store_true', help='Generate report only')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(model_path=args.model_path)
    
    if args.report_only:
        evaluator.load_model_and_preprocessor()
        print("Report-only mode not fully implemented")
    else:
        results = evaluator.run_complete_evaluation()

        print("\nEvaluation Summary:")
        print("=" * 50)
        print(f"R² Score: {results['metrics']['r2']:.4f}")
        print(f"RMSE: {results['metrics']['rmse']:.6f}")
        print(f"MAE: {results['metrics']['mae']:.6f}")
        print(f"MAPE: {results['metrics']['mape']:.2f}%")
        print(f"Directional Accuracy: {results['metrics']['directional_accuracy']:.2f}%")

if __name__ == "__main__":
    main()