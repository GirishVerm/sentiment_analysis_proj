"""
Model implementations for sentiment analysis.

Includes:
- Logistic Regression
- Support Vector Machine (SVM)
- Basic Neural Network
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Tuple, Optional
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseSentimentModel:
    """Base class for sentiment analysis models."""
    
    def __init__(self, vectorizer_type: str = "tfidf", max_features: int = 10000):
        """
        Initialize base model.
        
        Args:
            vectorizer_type: Type of vectorizer ('tfidf' or 'count')
            max_features: Maximum number of features for vectorizer
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        
        if vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
        
        self.model = None
        self.is_trained = False
    
    def fit(self, X_train: list, y_train: np.ndarray):
        """
        Train the model.
        
        Args:
            X_train: List of preprocessed text strings
            y_train: Array of labels
        """
        logger.info("Vectorizing training data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        logger.info(f"Training model on {X_train_vec.shape[0]} samples...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        logger.info("Training complete!")
    
    def predict(self, X: list) -> np.ndarray:
        """
        Predict labels for given texts.
        
        Args:
            X: List of preprocessed text strings
            
        Returns:
            Array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def predict_proba(self, X: list) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: List of preprocessed text strings
            
        Returns:
            Array of probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_vec = self.vectorizer.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_vec)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def save(self, filepath: str):
        """Save model and vectorizer to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and vectorizer from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.vectorizer_type = model_data['vectorizer_type']
        self.max_features = model_data['max_features']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class LogisticRegressionModel(BaseSentimentModel):
    """Logistic Regression for sentiment classification."""
    
    def __init__(self, vectorizer_type: str = "tfidf", max_features: int = 10000,
                 C: float = 1.0, max_iter: int = 1000):
        """
        Initialize Logistic Regression model.
        
        Args:
            vectorizer_type: Type of vectorizer
            max_features: Maximum features
            C: Regularization parameter
            max_iter: Maximum iterations
        """
        super().__init__(vectorizer_type, max_features)
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)


class SVMModel(BaseSentimentModel):
    """Support Vector Machine for sentiment classification."""
    
    def __init__(self, vectorizer_type: str = "tfidf", max_features: int = 10000,
                 C: float = 1.0, kernel: str = "linear"):
        """
        Initialize SVM model.
        
        Args:
            vectorizer_type: Type of vectorizer
            max_features: Maximum features
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly')
        """
        super().__init__(vectorizer_type, max_features)
        self.model = SVC(C=C, kernel=kernel, probability=True, random_state=42)


class NeuralNetworkModel(BaseSentimentModel):
    """Basic Neural Network for sentiment classification."""
    
    def __init__(self, vectorizer_type: str = "tfidf", max_features: int = 10000,
                 hidden_layers: Tuple[int, ...] = (128, 64), activation: str = "relu",
                 epochs: int = 10, batch_size: int = 32):
        """
        Initialize Neural Network model.
        
        Args:
            vectorizer_type: Type of vectorizer
            max_features: Maximum features
            hidden_layers: Tuple of hidden layer sizes
            activation: Activation function
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        super().__init__(vectorizer_type, max_features)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self, X_train: list, y_train: np.ndarray, X_val: Optional[list] = None, 
            y_val: Optional[np.ndarray] = None):
        """
        Train the neural network.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
        
        logger.info("Vectorizing training data...")
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
        
        if X_val is not None:
            X_val_vec = self.vectorizer.transform(X_val).toarray()
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Dense(self.hidden_layers[0], activation=self.activation, 
                              input_dim=X_train_vec.shape[1]))
        
        for layer_size in self.hidden_layers[1:]:
            model.add(layers.Dense(layer_size, activation=self.activation))
            model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        logger.info(f"Training neural network for {self.epochs} epochs...")
        validation_data = (X_val_vec, y_val) if X_val is not None else None
        
        model.fit(
            X_train_vec, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        self.model = model
        self.is_trained = True
        logger.info("Training complete!")
    
    def predict(self, X: list) -> np.ndarray:
        """Predict labels using neural network."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_vec = self.vectorizer.transform(X).toarray()
        predictions = self.model.predict(X_vec, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: list) -> np.ndarray:
        """Predict probabilities using neural network."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_vec = self.vectorizer.transform(X).toarray()
        predictions = self.model.predict(X_vec, verbose=0)
        # Return probabilities for both classes
        return np.hstack([1 - predictions, predictions])

