"""
Evaluation module for sentiment analysis models.

Provides metrics and visualization tools:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Classification Report
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates sentiment analysis models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def print_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
        """
        Print classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        metrics = self.compute_metrics(y_true, y_pred)
        
        print(f"\n{'='*50}")
        print(f"Evaluation Metrics for {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"{'='*50}\n")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   target_names: list = ['Negative', 'Positive']):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of the classes
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\nClassification Report:")
        print(report)
    
    def evaluate_model(self, model, X_test: list, y_test: np.ndarray, 
                      model_name: str = "Model", save_dir: str = "results") -> dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model with predict() method
            X_test: Test texts
            y_test: Test labels
            model_name: Name of the model
            save_dir: Directory to save results
            
        Returns:
            Dictionary of metrics
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute and print metrics
        metrics = self.print_metrics(y_test, y_pred, model_name)
        
        # Print classification report
        self.print_classification_report(y_test, y_pred)
        
        # Plot confusion matrix
        save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        self.plot_confusion_matrix(y_test, y_pred, model_name, save_path)
        
        return metrics

