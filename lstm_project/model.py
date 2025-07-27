import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, 
    BatchNormalization, LayerNormalization, Attention
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
import logging
from typing import Tuple, List, Dict, Any
from config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_SAVE_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom attention layer for sequence modeling.
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Calculate attention scores
        attention_scores = tf.nn.tanh(tf.tensordot(inputs, self.attention_weights, axes=1))
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        weighted_input = inputs * attention_weights
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class VolatilityLSTM:
    """
    Advanced LSTM model for cryptocurrency volatility prediction.
    """
    
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        self.model = None
        
    def build_lstm_model(self) -> Model:
        """
        Build LSTM model with attention mechanism.
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LSTM model with input shape: {self.input_shape}")
        
        # Input layer
        inputs = Input(shape=self.input_shape, name='sequence_input')
        x = inputs
        
        # LSTM layers
        for i, layer_config in enumerate(MODEL_CONFIG['lstm_layers']):
            x = LSTM(
                units=layer_config['units'],
                return_sequences=layer_config['return_sequences'],
                dropout=layer_config['dropout'],
                recurrent_dropout=layer_config['recurrent_dropout'],
                kernel_regularizer=l2(0.001),
                recurrent_regularizer=l2(0.001),
                name=f'lstm_{i+1}'
            )(x)
            
            if MODEL_CONFIG['use_batch_norm']:
                x = BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
        
        # Attention mechanism (if enabled and we have sequences)
        if MODEL_CONFIG['use_attention'] and MODEL_CONFIG['lstm_layers'][-2]['return_sequences']:
            lstm_output_with_sequences = x
            
            attention_output = AttentionLayer(name='attention')(lstm_output_with_sequences)
            
            if not MODEL_CONFIG['lstm_layers'][-1]['return_sequences']:
                final_lstm = x
                x = Concatenate(name='concat_attention')([attention_output, final_lstm])
            else:
                x = attention_output
        
        # Dense layers
        for i, layer_config in enumerate(MODEL_CONFIG['dense_layers']):
            x = Dense(
                units=layer_config['units'],
                activation=layer_config['activation'],
                kernel_regularizer=l2(0.001),
                name=f'dense_{i+1}'
            )(x)
            
            x = Dropout(layer_config['dropout'], name=f'dropout_dense_{i+1}')(x)
            
            if MODEL_CONFIG['use_batch_norm']:
                x = BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
        
        outputs = Dense(
            1, 
            activation=MODEL_CONFIG['output_activation'],
            name='volatility_output'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='VolatilityLSTM')
        
        optimizer = self._get_optimizer()
        
        model.compile(
            optimizer=optimizer,
            loss=TRAINING_CONFIG['loss_function'],
            metrics=TRAINING_CONFIG['metrics']
        )
        
        logger.info("Model compiled successfully")
        logger.info(f"Model parameters: {model.count_params():,}")
        
        self.model = model
        return model
    
    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Get configured optimizer."""
        if TRAINING_CONFIG['optimizer'].lower() == 'adam':
            return Adam(
                learning_rate=TRAINING_CONFIG['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
        elif TRAINING_CONFIG['optimizer'].lower() == 'rmsprop':
            return RMSprop(learning_rate=TRAINING_CONFIG['learning_rate'])
        else:
            raise ValueError(f"Unsupported optimizer: {TRAINING_CONFIG['optimizer']}")
    
    def build_simple_lstm(self) -> Model:
        """
        Build a simpler LSTM model for comparison.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building simple LSTM model")
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def build_ensemble_model(self, n_models: int = 3) -> List[Model]:
        """
        Build ensemble of LSTM models with different architectures.
        
        Args:
            n_models: Number of models in ensemble
            
        Returns:
            List of compiled models
        """
        logger.info(f"Building ensemble of {n_models} models")
        
        models = []
        
        for i in range(n_models):
            # Vary the architecture slightly for each model
            inputs = Input(shape=self.input_shape)
            
            # Different LSTM configurations for diversity
            if i == 0:
                x = LSTM(128, return_sequences=True, dropout=0.2)(inputs)
                x = LSTM(64, return_sequences=False, dropout=0.2)(x)
            elif i == 1:
                x = LSTM(100, return_sequences=True, dropout=0.3)(inputs)
                x = LSTM(50, return_sequences=True, dropout=0.3)(x)
                x = LSTM(25, return_sequences=False, dropout=0.3)(x)
            else:
                x = LSTM(150, return_sequences=True, dropout=0.1)(inputs)
                x = LSTM(75, return_sequences=False, dropout=0.2)(x)
            
            x = Dense(50, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(25, activation='relu')(x)
            outputs = Dense(1, activation='linear')(x)
            
            model = Model(inputs=inputs, outputs=outputs, name=f'LSTM_Ensemble_{i+1}')
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='huber_loss',
                metrics=['mae', 'mse']
            )
            
            models.append(model)
        
        return models
    
    def get_callbacks(self, model_path: str = None) -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks.
        
        Args:
            model_path: Path to save the best model
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        early_stopping = EarlyStopping(
            monitor=TRAINING_CONFIG['early_stopping']['monitor'],
            patience=TRAINING_CONFIG['early_stopping']['patience'],
            restore_best_weights=TRAINING_CONFIG['early_stopping']['restore_best_weights'],
            verbose=1
        )
        callbacks.append(early_stopping)
        
        lr_scheduler = ReduceLROnPlateau(
            monitor=TRAINING_CONFIG['lr_scheduler']['monitor'],
            factor=TRAINING_CONFIG['lr_scheduler']['factor'],
            patience=TRAINING_CONFIG['lr_scheduler']['patience'],
            min_lr=TRAINING_CONFIG['lr_scheduler']['min_lr'],
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        if model_path:
            checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=MODEL_SAVE_CONFIG['save_best_only'],
                save_weights_only=MODEL_SAVE_CONFIG['save_weights_only'],
                verbose=1
            )
            callbacks.append(checkpoint)
        
        tensorboard = TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def get_model_summary(self) -> str:
        """
        Get detailed model summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"
        
        import io
        import contextlib
        
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.model.summary()
        
        return stream.getvalue()
    
    def plot_model_architecture(self, filepath: str = None):
        """
        Plot model architecture diagram.
        
        Args:
            filepath: Path to save the plot
        """
        if self.model is None:
            logger.error("Model not built yet")
            return
        
        try:
            from tensorflow.keras.utils import plot_model
            
            filepath = filepath or 'model_architecture.png'
            plot_model(
                self.model,
                to_file=filepath,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=300
            )
            logger.info(f"Model architecture saved to {filepath}")
        except ImportError:
            logger.warning("Could not plot model. Install graphviz and pydot for visualization.")

class ModelBuilder:
    """
    Factory class for building different types of models.
    """
    
    @staticmethod
    def create_volatility_lstm(input_shape: Tuple[int, int], 
                              model_type: str = 'advanced') -> VolatilityLSTM:
        """
        Create LSTM model for volatility prediction.
        
        Args:
            input_shape: Shape of input sequences
            model_type: Type of model ('advanced', 'simple', 'ensemble')
            
        Returns:
            VolatilityLSTM instance
        """
        lstm_model = VolatilityLSTM(input_shape)
        
        if model_type == 'advanced':
            lstm_model.build_lstm_model()
        elif model_type == 'simple':
            lstm_model.build_simple_lstm()
        elif model_type == 'ensemble':
            models = lstm_model.build_ensemble_model()
            lstm_model.ensemble_models = models
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return lstm_model
    
    @staticmethod
    def load_pretrained_model(model_path: str) -> Model:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded Keras model
        """
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

# Custom loss functions
def huber_loss_custom(delta: float = 1.0):
    """
    Custom Huber loss function.
    
    Args:
        delta: Threshold for switching between MSE and MAE
        
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        squared_loss = tf.square(error) / 2
        linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    return loss

def quantile_loss(quantile: float = 0.5):
    """
    Quantile loss for probabilistic predictions.
    
    Args:
        quantile: Quantile to predict (0.5 for median)
        
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.maximum(quantile * error, (quantile - 1) * error)
    
    return loss

# Custom metrics
def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (percentage of correct trend predictions).
    """
    true_direction = tf.sign(y_true[1:] - y_true[:-1])
    pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])
    
    correct_predictions = tf.equal(true_direction, pred_direction)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return accuracy

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE metric.
    """
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, float('inf')))
    return 100.0 * tf.reduce_mean(diff)

if __name__ == "__main__":
    # Test model building
    input_shape = (60, 16)  # 60 time steps, 16 features
    
    # Build advanced model
    model_builder = ModelBuilder()
    lstm_model = model_builder.create_volatility_lstm(input_shape, 'advanced')
    
    print("Model Summary:")
    print(lstm_model.get_model_summary())
    
    # Test simple model
    simple_lstm = model_builder.create_volatility_lstm(input_shape, 'simple')
    print(f"\nSimple model parameters: {simple_lstm.model.count_params():,}")
    
    # Test callbacks
    callbacks = lstm_model.get_callbacks('test_model.h5')
    print(f"\nConfigured {len(callbacks)} callbacks")