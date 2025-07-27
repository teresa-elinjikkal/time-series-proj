#!/usr/bin/env python3
"""
Main entry point for the LSTM Cryptocurrency Volatility Prediction project.
Provides command-line interface for training, evaluation, and prediction.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'lstm_project'))

from data_preprocessing import DataPreprocessor
from train import ModelTrainer
from evaluate import ModelEvaluator
from utils import setup_logging, get_system_info, create_project_structure
from config import (
    TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG, 
    EVALUATION_CONFIG, LOGGING_CONFIG
)

def setup_project():
    setup_logging(
        log_level=LOGGING_CONFIG['level'],
        log_file=str(LOGGING_CONFIG['log_file'])
    )
    
    logger = logging.getLogger(__name__)
    
    system_info = get_system_info()
    logger.info("Starting LSTM Cryptocurrency Volatility Prediction")
    logger.info("=" * 60)
    logger.info(f"Python Version: {system_info['python_version']}")
    logger.info(f"TensorFlow Version: {system_info['tensorflow_version']}")
    logger.info(f"GPU Available: {system_info['gpu_available']}")
    logger.info(f"Platform: {system_info['platform']}")
    logger.info("=" * 60)
    
    return logger

def run_preprocessing(args):
    """Run data preprocessing pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing pipeline")
    
    preprocessor = DataPreprocessor()
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
        
        print("\nDATA PREPROCESSING COMPLETED")
        print("=" * 50)
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Sequence length: {X_train.shape[1]}")
        print(f"Features: {X_train.shape[2]}")
        print(f"Data saved to: {DATA_CONFIG['processed_data_path']}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

def run_training(args):
    """Run model training pipeline."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting model training with type: {args.model_type}")
    
    # Override config with command line arguments
    if args.epochs:
        TRAINING_CONFIG['epochs'] = args.epochs
    if args.batch_size:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    if args.learning_rate:
        TRAINING_CONFIG['learning_rate'] = args.learning_rate
    
    trainer = ModelTrainer(model_type=args.model_type)
    
    try:
        if args.resume:
            # Resume training from checkpoint
            trainer.resume_training(args.resume, args.additional_epochs or 50)
        else:
            # Start fresh training
            results = trainer.run_training_pipeline()
            
            print("\nMODEL TRAINING COMPLETED")
            print("=" * 50)
            print(f"Model Type: {results['model_type']}")
            print(f"Training Samples: {results['training_samples']:,}")
            print(f"Test Samples: {results['test_samples']:,}")
            print(f"Input Shape: {results['input_shape']}")
            print(f"Model Saved: {results['model_path']}")
            print(f"Completed: {results['training_completed']}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

def run_evaluation(args):
    """Run model evaluation pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation")
    
    evaluator = ModelEvaluator(model_path=args.model_path)
    
    try:
        results = evaluator.run_complete_evaluation()
        
        print("\nMODEL EVALUATION COMPLETED")
        print("=" * 50)
        metrics = results['metrics']
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        print(f"Correlation: {metrics['correlation']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

def run_prediction(args):
    """Run prediction on new data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting prediction pipeline")
    
    try:
        from model import ModelBuilder
        
        model = ModelBuilder.load_pretrained_model(args.model_path)
        preprocessor = DataPreprocessor()
        preprocessor.load_scalers()
        
        if args.crypto_symbol:
            DATA_CONFIG['crypto_symbol'] = args.crypto_symbol
        
        raw_data = preprocessor.fetch_crypto_data(period='3mo')
        price_data = preprocessor.calculate_technical_indicators(raw_data)
        sentiment_data = preprocessor.generate_sentiment_data(price_data)
        
        combined_data = pd.concat([price_data, sentiment_data], axis=1)
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        X, _ = preprocessor.create_sequences(combined_data)

        predictions = model.predict(X[-args.forecast_days:], verbose=0)
        

        if hasattr(preprocessor, 'target_scaler'):
            predictions = preprocessor.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
        
        print(f"\n🔮 VOLATILITY PREDICTIONS for {DATA_CONFIG['crypto_symbol']}")
        print("=" * 60)
        
        for i, pred in enumerate(predictions):
            print(f"Day {i+1}: {pred:.6f}")
        
        print(f"\nAverage predicted volatility: {predictions.mean():.6f}")
        print(f"Volatility range: {predictions.min():.6f} - {predictions.max():.6f}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

def run_full_pipeline(args):
    """Run the complete pipeline: preprocessing -> training -> evaluation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting complete pipeline")
    
    try:
        print("Step 1/3: Data Preprocessing")
        run_preprocessing(args)
        
        print("Step 2/3: Model Training")
        run_training(args)
        
        print("Step 3/3: Model Evaluation")
        run_evaluation(args)

        print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the following directories for results:")
        print(f"  • Models: models/")
        print(f"  • Plots: plots/")
        print(f"  • Logs: logs/")
        print(f"  • Data: data/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='LSTM Cryptocurrency Volatility Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run.py pipeline --model_type advanced
  
  # Train model only
  python run.py train --model_type ensemble --epochs 150
  
  # Evaluate existing model
  python run.py evaluate --model_path models/lstm_volatility_model.h5
  
  # Make predictions
  python run.py predict --model_path models/lstm_volatility_model.h5 --crypto_symbol ETH-USD
  
  # Just preprocess data
  python run.py preprocess
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--model_type', choices=['advanced', 'simple', 'ensemble'],
                                default='advanced', help='Type of model to train')
    pipeline_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    pipeline_parser.add_argument('--batch_size', type=int, help='Training batch size')
    
    preprocess_parser = subparsers.add_parser('preprocess', help='Run data preprocessing only')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model_type', choices=['advanced', 'simple', 'ensemble'],
                             default='advanced', help='Type of model to train')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, help='Training batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    train_parser.add_argument('--resume', type=str, help='Path to model to resume training')
    train_parser.add_argument('--additional_epochs', type=int, help='Additional epochs for resume')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model_path', type=str, help='Path to trained model')
    eval_parser.add_argument('--no_plots', action='store_true', help='Skip plotting')
    
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    predict_parser.add_argument('--crypto_symbol', type=str, default='BTC-USD', 
                               help='Cryptocurrency symbol')
    predict_parser.add_argument('--forecast_days', type=int, default=7, 
                               help='Number of days to forecast')

    setup_parser = subparsers.add_parser('setup', help='Setup project structure')
    setup_parser.add_argument('--project_name', type=str, default='lstm_crypto_project',
                             help='Project directory name')
    
    args = parser.parse_args()
    logger = setup_project()
    
    if args.command == 'pipeline':
        run_full_pipeline(args)
    elif args.command == 'preprocess':
        run_preprocessing(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'predict':
        run_prediction(args)
    elif args.command == 'setup':
        create_project_structure(args.project_name)
        print(f"Project structure created at: {args.project_name}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()