"""
Data loading module for movie review sentiment analysis.

This module handles loading movie reviews and star ratings from various sources:
- CSV files (e.g., from Kaggle datasets)
- APIs (IMDb, Letterboxd, etc.) - to be implemented
"""

import pandas as pd
import os
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieReviewLoader:
    """Loads movie review data from various sources."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_from_csv(self, filepath: str, text_col: str = "review", 
                     rating_col: Optional[str] = None, 
                     sentiment_col: Optional[str] = None) -> pd.DataFrame:
        """
        Load reviews from a CSV file.
        
        Args:
            filepath: Path to CSV file
            text_col: Name of column containing review text
            rating_col: Name of column containing star ratings (optional)
            sentiment_col: Name of column containing sentiment labels like "positive"/"negative" (optional)
            
        Returns:
            DataFrame with 'text' and either 'rating' or 'sentiment' column
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Standardize text column name
        if text_col in df.columns:
            df = df.rename(columns={text_col: "text"})
        else:
            raise ValueError(f"CSV must contain '{text_col}' column for review text")
        
        # Handle rating or sentiment column
        if sentiment_col and sentiment_col in df.columns:
            df = df.rename(columns={sentiment_col: "sentiment"})
            required_cols = ["text", "sentiment"]
        elif rating_col and rating_col in df.columns:
            df = df.rename(columns={rating_col: "rating"})
            required_cols = ["text", "rating"]
        elif "sentiment" in df.columns:
            # Auto-detect sentiment column
            required_cols = ["text", "sentiment"]
        elif "rating" in df.columns:
            # Auto-detect rating column
            required_cols = ["text", "rating"]
        else:
            raise ValueError(f"CSV must contain either '{rating_col or 'rating'}' or '{sentiment_col or 'sentiment'}' column")
        
        # Ensure we have required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Remove rows with missing data
        df = df.dropna(subset=required_cols)
        
        logger.info(f"Loaded {len(df)} reviews")
        return df[required_cols]
    
    def load_from_kaggle(self, dataset_name: str, filename: str) -> pd.DataFrame:
        """
        Load dataset from Kaggle (requires kaggle API setup).
        
        Args:
            dataset_name: Kaggle dataset name (e.g., "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
            filename: Name of the CSV file in the dataset
            
        Returns:
            DataFrame with reviews
        """
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            logger.info(f"Downloading dataset {dataset_name}")
            api.dataset_download_files(dataset_name, path=self.data_dir, unzip=True)
            
            filepath = os.path.join(self.data_dir, filename)
            return self.load_from_csv(filepath)
            
        except ImportError:
            logger.error("Kaggle API not installed. Install with: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Error loading from Kaggle: {e}")
            raise
    
    def load_from_imdb_api(self, movie_ids: list) -> pd.DataFrame:
        """
        Load reviews from IMDb API (to be implemented).
        
        Note: IMDb doesn't have an official public API. This would require
        using third-party libraries or web scraping (with proper ToS compliance).
        
        Args:
            movie_ids: List of IMDb movie IDs
            
        Returns:
            DataFrame with reviews
        """
        # TODO: Implement IMDb data collection
        # Options:
        # 1. Use imdb-sentiment-analysis library
        # 2. Use IMDbPY library
        # 3. Web scraping (ensure ToS compliance)
        raise NotImplementedError("IMDb API integration not yet implemented")
    
    def prepare_binary_labels(self, df: pd.DataFrame, threshold: float = 3.0, 
                             sentiment_col: Optional[str] = None) -> pd.DataFrame:
        """
        Convert star ratings or sentiment labels to binary labels (positive/negative).
        
        Args:
            df: DataFrame with 'rating' column or sentiment column
            threshold: Ratings >= threshold are positive (1), else negative (0) (for numeric ratings)
            sentiment_col: Name of sentiment column if using string labels (e.g., "positive"/"negative")
            
        Returns:
            DataFrame with added 'label' column
        """
        df = df.copy()
        
        if sentiment_col and sentiment_col in df.columns:
            # Handle string sentiment labels (positive/negative)
            logger.info(f"Converting sentiment labels from '{sentiment_col}' column")
            df["label"] = (df[sentiment_col].str.lower().str.strip() == "positive").astype(int)
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        elif "rating" in df.columns:
            # Handle numeric ratings
            df["label"] = (df["rating"] >= threshold).astype(int)
        else:
            raise ValueError("Must provide either 'rating' column or 'sentiment_col' parameter")
        
        return df
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, test_size=(1 - train_ratio), random_state=random_state, stratify=df.get("label", None)
        )
        
        # Second split: val vs test
        val_size = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size), random_state=random_state, stratify=temp_df.get("label", None)
        )
        
        logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

