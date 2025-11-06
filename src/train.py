"""
Training script for sentiment analysis models.
"""

import argparse
import pandas as pd
import numpy as np
from data_loader import MovieReviewLoader
from preprocessor import TextPreprocessor
from models import LogisticRegressionModel, SVMModel, NeuralNetworkModel
from evaluate import ModelEvaluator
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model_type: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                preprocessor: TextPreprocessor, model_dir: str = "models") -> object:
    """
    Train a sentiment analysis model.
    
    Args:
        model_type: Type of model ('lr', 'svm', 'nn')
        train_df: Training DataFrame with 'text' and 'label' columns
        val_df: Validation DataFrame
        preprocessor: Text preprocessor
        model_dir: Directory to save models
        
    Returns:
        Trained model
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Preprocess texts
    logger.info("Preprocessing training data...")
    X_train = preprocessor.preprocess_batch(train_df['text'].tolist())
    y_train = train_df['label'].values
    
    X_val = preprocessor.preprocess_batch(val_df['text'].tolist())
    y_val = val_df['label'].values
    
    # Initialize model
    if model_type == "lr":
        model = LogisticRegressionModel()
        model_name = "Logistic Regression"
    elif model_type == "svm":
        model = SVMModel()
        model_name = "SVM"
    elif model_type == "nn":
        model = NeuralNetworkModel()
        model_name = "Neural Network"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    logger.info(f"Training {model_name}...")
    if model_type == "nn":
        model.fit(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
    model.save(model_path)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file with reviews")
    parser.add_argument("--model", type=str, choices=["lr", "svm", "nn"], 
                       default="lr", help="Model type to train")
    parser.add_argument("--text-col", type=str, default="review", 
                       help="Name of text column in CSV")
    parser.add_argument("--rating-col", type=str, default="rating", 
                       help="Name of rating column in CSV")
    parser.add_argument("--threshold", type=float, default=3.0,
                       help="Rating threshold for binary classification")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    # Load data
    loader = MovieReviewLoader()
    df = loader.load_from_csv(args.data, args.text_col, args.rating_col)
    df = loader.prepare_binary_labels(df, args.threshold)
    
    # Split data
    train_df, val_df, test_df = loader.split_data(df)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    # Train model
    model = train_model(args.model, train_df, val_df, preprocessor, args.model_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

