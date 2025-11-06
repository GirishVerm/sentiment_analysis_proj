"""
Main entry point for sentiment analysis project.

This script provides a complete pipeline:
1. Load data
2. Preprocess
3. Train models
4. Evaluate models
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis of Movie Reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and evaluate all models
  python main.py --data data/reviews.csv --train-all
  
  # Train only logistic regression
  python main.py --data data/reviews.csv --model lr
  
  # Evaluate existing model
  python main.py --data data/reviews.csv --evaluate --model-path models/lr_model.pkl
        """
    )
    
    parser.add_argument("--data", type=str, required=True,
                       help="Path to CSV file with reviews")
    parser.add_argument("--text-col", type=str, default="review",
                       help="Name of text column in CSV")
    parser.add_argument("--rating-col", type=str, default=None,
                       help="Name of rating column in CSV (optional, use with --sentiment-col)")
    parser.add_argument("--sentiment-col", type=str, default=None,
                       help="Name of sentiment column in CSV (e.g., 'sentiment' for positive/negative labels)")
    parser.add_argument("--threshold", type=float, default=3.0,
                       help="Rating threshold for binary classification (default: 3.0, only used with ratings)")
    parser.add_argument("--model", type=str, choices=["lr", "svm", "nn"],
                       help="Model type to train (lr, svm, or nn)")
    parser.add_argument("--train-all", action="store_true",
                       help="Train all models (lr, svm, nn)")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory to save/load models")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to use (for testing)")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    loader = MovieReviewLoader()
    df = loader.load_from_csv(args.data, args.text_col, args.rating_col, args.sentiment_col)
    
    # Limit samples if specified
    if args.max_samples and len(df) > args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples")
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
    
    # Prepare binary labels
    if "sentiment" in df.columns:
        df = loader.prepare_binary_labels(df, sentiment_col="sentiment")
    else:
        df = loader.prepare_binary_labels(df, args.threshold)
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Split data
    train_df, val_df, test_df = loader.split_data(df)
    
    # Initialize preprocessor
    logger.info("Initializing preprocessor...")
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    
    # Preprocess texts
    logger.info("Preprocessing texts...")
    X_train = preprocessor.preprocess_batch(train_df['text'].tolist())
    X_val = preprocessor.preprocess_batch(val_df['text'].tolist())
    X_test = preprocessor.preprocess_batch(test_df['text'].tolist())
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Determine which models to train
    models_to_train = []
    if args.train_all:
        models_to_train = ["lr", "svm", "nn"]
    elif args.model:
        models_to_train = [args.model]
    else:
        logger.warning("No model specified. Use --model or --train-all")
        return
    
    # Train and evaluate models
    evaluator = ModelEvaluator()
    all_results = {}
    
    for model_type in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()} model")
        logger.info(f"{'='*60}")
        
        # Initialize model
        try:
            if model_type == "lr":
                model = LogisticRegressionModel()
                model_name = "Logistic Regression"
            elif model_type == "svm":
                model = SVMModel()
                model_name = "SVM"
            elif model_type == "nn":
                try:
                    model = NeuralNetworkModel()
                    model_name = "Neural Network"
                except ImportError as e:
                    logger.warning(f"Neural Network model requires TensorFlow. Skipping. Error: {e}")
                    logger.info("Install TensorFlow with: pip install tensorflow")
                    continue
            
            # Train model
            if model_type == "nn":
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)
            
            # Save model
            model_path = os.path.join(args.model_dir, f"{model_type}_model.pkl")
            model.save(model_path)
            
            # Evaluate on test set
            logger.info(f"\nEvaluating {model_name} on test set...")
            metrics = evaluator.evaluate_model(
                model, X_test, y_test, model_name, args.results_dir
            )
            
            all_results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY OF ALL MODELS")
    logger.info(f"{'='*60}")
    for model_name, metrics in all_results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()

