import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import tensorflow as tf

from data_preprocessing import DataPreprocessor
from model import ModelBuilder, VolatilityLSTM
from config import (
    TRAINING_CONFIG, MODEL_CONFIG, MODEL_SAVE_CONFIG, 
    LOGGING_CONFIG, DATA_CONFIG, PLOTS_DIR
)

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:

    
    def __init__(self, model_type: str = 'advanced'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.training_history = None
        self.best_model_path = str(MODEL_SAVE_CONFIG['model_path'])
        
    def setup_training_environment(self):
        logger.info("Setting up training environment")
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}")
        else:
            logger.info("No GPU found, using CPU")
        
        tf.random.set_seed(DATA_CONFIG['random_state'])
        np.random.seed(DATA_CONFIG['random_state'])
        
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
    
    def prepare_data(self) -> Tuple[np.ndarray, ...]:
        """
        Prepare training data using the preprocessing pipeline.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing training data")
        
        self.preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_pipeline()
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape: Tuple[int, int]) -> VolatilityLSTM:
        """
        Build the LSTM model for training.
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Built VolatilityLSTM model
        """
        logger.info(f"Building {self.model_type} model")
        
        model_builder = ModelBuilder()
        self.model = model_builder.create_volatility_lstm(input_shape, self.model_type)
        
        logger.info("Model built successfully")
        logger.info(f"Model parameters: {self.model.model.count_params():,}")
        
        return self.model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info("Starting model training")
        
        if X_val is None or y_val is None:
            validation_data = None
            validation_split = TRAINING_CONFIG.get('validation_split', 0.2)
            logger.info(f"Using validation split: {validation_split}")
        else:
            validation_data = (X_val, y_val)
            validation_split = None
            logger.info(f"Using provided validation data: {X_val.shape}")
        
        callbacks = self.model.get_callbacks(self.best_model_path)
        
        start_time = datetime.now()
        logger.info(f"Training started at: {start_time}")
        
        history = self.model.model.fit(
            X_train, y_train,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"Training completed in: {training_duration}")
        
        self.training_history = history.history
        
        self._save_training_history()
        
        return self.training_history
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None) -> list[Dict[str, Any]]:
        """
        Train ensemble of models.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            
        Returns:
            List of training histories for each model
        """
        if not hasattr(self.model, 'ensemble_models'):
            raise ValueError("Ensemble models not available. Use model_type='ensemble'")
        
        logger.info(f"Training ensemble of {len(self.model.ensemble_models)} models")
        
        histories = []
        
        for i, model in enumerate(self.model.ensemble_models):
            logger.info(f"Training ensemble model {i+1}/{len(self.model.ensemble_models)}")
            
            model_path = str(MODEL_SAVE_CONFIG['model_path']).replace('.h5', f'_ensemble_{i+1}.h5')
            callbacks = self._get_callbacks_for_model(model_path)
            
            if X_val is None or y_val is None:
                validation_data = None
                validation_split = 0.2
            else:
                validation_data = (X_val, y_val)
                validation_split = None
            
            history = model.fit(
                X_train, y_train,
                epochs=TRAINING_CONFIG['epochs'],
                batch_size=TRAINING_CONFIG['batch_size'],
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            histories.append(history.history)
            logger.info(f"Ensemble model {i+1} training completed")
        
        self.training_history = histories
        return histories
    
    def _get_callbacks_for_model(self, model_path: str):
        """Get callbacks for individual model training."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        return callbacks
    
    def _save_training_history(self):
        """Save training history to disk."""
        history_path = MODEL_SAVE_CONFIG['history_path']
        
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        logger.info(f"Training history saved to {history_path}")
    
    def save_model_config(self):
        """Save model configuration for reproducibility."""
        config_data = {
            'model_config': MODEL_CONFIG,
            'training_config': TRAINING_CONFIG,
            'data_config': DATA_CONFIG,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'input_shape': self.model.input_shape if self.model else None
        }
        
        config_path = MODEL_SAVE_CONFIG['config_path']
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Model configuration saved to {config_path}")
    
    def validate_training_setup(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Validate training setup before starting training.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
        """
        logger.info("Validating training setup")
        
        if len(X_train.shape) != 3:
            raise ValueError(f"Expected 3D training data, got shape: {X_train.shape}")
        
        if len(y_train.shape) != 1:
            raise ValueError(f"Expected 1D target data, got shape: {y_train.shape}")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Sample count mismatch: X={X_train.shape[0]}, y={y_train.shape[0]}")
        
        if np.isnan(X_train).any():
            raise ValueError("Training data contains NaN values")
        
        if np.isnan(y_train).any():
            raise ValueError("Training targets contain NaN values")
        
        logger.info(f"Training data range: [{X_train.min():.6f}, {X_train.max():.6f}]")
        logger.info(f"Training targets range: [{y_train.min():.6f}, {y_train.max():.6f}]")
        
        logger.info("Training setup validation passed")
    
    def run_training_pipeline(self) -> Dict[str, Any]:

        logger.info("Starting complete training pipeline")
        
        self.setup_training_environment()
        
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        self.validate_training_setup(X_train, y_train)
        
        self.save_model_config()
        
        if self.model_type == 'ensemble':
            training_history = self.train_ensemble(X_train, y_train)
        else:
            training_history = self.train_model(X_train, y_train)
        
        results = {
            'training_history': training_history,
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'input_shape': (X_train.shape[1], X_train.shape[2]),
            'model_path': self.best_model_path,
            'training_completed': datetime.now().isoformat()
        }
        
        logger.info("Training pipeline completed successfully")
        
        return results
    
    def resume_training(self, model_path: str, additional_epochs: int = 50):
        """
        Resume training from a saved checkpoint.
        
        Args:
            model_path: Path to saved model
            additional_epochs: Number of additional epochs to train
        """
        logger.info(f"Resuming training from {model_path}")
        
        from model import ModelBuilder
        self.model = ModelBuilder.load_pretrained_model(model_path)
        
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        callbacks = self._get_callbacks_for_model(model_path)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=additional_epochs,
            batch_size=TRAINING_CONFIG['batch_size'],
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training resumed successfully")
        return history.history

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM volatility prediction model')
    parser.add_argument('--model_type', choices=['advanced', 'simple', 'ensemble'], 
                       default='advanced', help='Type of model to train')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--resume', type=str, help='Path to model to resume training')
    
    args = parser.parse_args()
    
    if args.epochs:
        TRAINING_CONFIG['epochs'] = args.epochs
    if args.batch_size:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    
    trainer = ModelTrainer(model_type=args.model_type)
    
    if args.resume:
        trainer.resume_training(args.resume)
    else:
        results = trainer.run_training_pipeline()
        
        print("\nTraining Results Summary:")
        print("=" * 50)
        print(f"Model Type: {results['model_type']}")
        print(f"Training Samples: {results['training_samples']:,}")
        print(f"Test Samples: {results['test_samples']:,}")
        print(f"Input Shape: {results['input_shape']}")
        print(f"Model Saved: {results['model_path']}")
        print(f"Completed: {results['training_completed']}")

if __name__ == "__main__":
    main()