"""
Helper script to download sample movie review datasets.

This script provides utilities to download common sentiment analysis datasets
from sources like Kaggle or other public repositories.
"""

import os
import requests
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dataset_name: str, filename: str, data_dir: str = "data"):
    """
    Download a dataset from Kaggle.
    
    Requires Kaggle API credentials:
    1. Install: pip install kaggle
    2. Get API token from https://www.kaggle.com/account
    3. Place kaggle.json in ~/.kaggle/
    
    Args:
        dataset_name: Kaggle dataset name (e.g., "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        filename: Name of the CSV file to extract
        data_dir: Directory to save the data
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        
        logger.info(f"Dataset downloaded to {data_dir}")
        logger.info(f"Look for {filename} in {data_dir}")
        
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        logger.info("Alternatively, manually download datasets from https://www.kaggle.com/datasets")
        raise
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def get_sample_datasets_info():
    """
    Print information about popular movie review datasets.
    """
    datasets = {
        "IMDb Movie Reviews": {
            "source": "Kaggle",
            "dataset": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
            "filename": "IMDB Dataset.csv",
            "description": "50,000 movie reviews with binary sentiment labels",
            "columns": "review, sentiment"
        },
        "Rotten Tomatoes Reviews": {
            "source": "Kaggle",
            "dataset": "utathya/imdb-movie-reviews-dataset",
            "filename": "reviews.csv",
            "description": "Movie reviews with ratings",
            "columns": "review, rating"
        },
        "Movie Review Dataset": {
            "source": "Kaggle",
            "dataset": "yasserh/movie-review-dataset",
            "filename": "Movie_Reviews.csv",
            "description": "Movie reviews with sentiment labels",
            "columns": "Review, Sentiment"
        }
    }
    
    print("\n" + "="*70)
    print("POPULAR MOVIE REVIEW DATASETS")
    print("="*70)
    
    for name, info in datasets.items():
        print(f"\n{name}:")
        print(f"  Source: {info['source']}")
        print(f"  Dataset: {info['dataset']}")
        print(f"  Filename: {info['filename']}")
        print(f"  Description: {info['description']}")
        print(f"  Columns: {info['columns']}")
        print(f"  Download: kaggle datasets download -d {info['dataset']}")
    
    print("\n" + "="*70)
    print("To download a dataset:")
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Get API token from https://www.kaggle.com/account")
    print("3. Place kaggle.json in ~/.kaggle/")
    print("4. Run: python src/download_data.py --dataset <dataset-name>")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download movie review datasets")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--dataset", type=str,
                       help="Kaggle dataset name (e.g., lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)")
    parser.add_argument("--filename", type=str,
                       help="Name of CSV file in the dataset")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory to save data")
    
    args = parser.parse_args()
    
    if args.list:
        get_sample_datasets_info()
    elif args.dataset:
        if not args.filename:
            logger.error("--filename is required when downloading a dataset")
        else:
            download_kaggle_dataset(args.dataset, args.filename, args.data_dir)
    else:
        get_sample_datasets_info()
        parser.print_help()

