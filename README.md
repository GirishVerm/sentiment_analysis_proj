# Sentiment Analysis of Movie Reviews

This README file is written with AI, cause I like detailed READMEs but I am not gonna write them myself <3
Datasets from


Session 1 Dataset

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download


Session 2 Dataset

https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset/


A supervised machine learning project for classifying movie reviews as positive or negative sentiment using traditional ML algorithms and neural networks.

**Course**: COMP 4750 - Introduction to Natural Language Processing (Fall 2025)  
**Student**: Girish Verma (gverma@mun.ca, Student ID: 202157608)

---

## Project Overview

This project develops a sentiment analysis system that classifies movie reviews based on their emotional valence (positive/negative). The system uses star ratings or sentiment labels as ground-truth data and implements multiple classification approaches including Logistic Regression, Support Vector Machines (SVM), and Neural Networks.

### Objectives

- Build a sentiment classifier for movie reviews using supervised learning
- Compare traditional ML approaches (Logistic Regression, SVM) with neural networks
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices
- Implement standard NLP preprocessing techniques (tokenization, stemming, lemmatization)

---

## Session 1: Project Setup and Initial Implementation

### What We Built

#### 1. Project Structure
```
sentiment_analysis_proj/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading from CSV, Kaggle, or APIs
│   ├── preprocessor.py     # NLP text preprocessing
│   ├── models.py           # Model implementations (LR, SVM, NN)
│   ├── evaluate.py         # Evaluation metrics and visualization
│   ├── train.py            # Training script
│   ├── main.py             # Main entry point
│   └── download_data.py    # Helper for downloading datasets
├── data/                   # Directory for datasets
├── models/                 # Directory for saved models
├── results/                # Directory for evaluation results
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

#### 2. Core Modules

**Data Loader (`data_loader.py`)**
- Loads movie reviews from CSV files
- Supports both star ratings and sentiment labels (positive/negative)
- Auto-detects column names (review, rating, sentiment)
- Splits data into train/validation/test sets with stratification
- Includes placeholders for Kaggle and IMDb API integration

**Text Preprocessor (`preprocessor.py`)**
- Text cleaning (URL removal, email removal, special characters)
- Tokenization using NLTK (with fallback to simple split)
- Stop word removal
- Stemming and lemmatization support
- Handles both older (`punkt`) and newer (`punkt_tab`) NLTK tokenizers

**Models (`models.py`)**
- **Logistic Regression**: Binary classification with TF-IDF/Count vectorization
- **Support Vector Machine**: Linear, RBF, or polynomial kernels
- **Neural Network**: Multi-layer perceptron using TensorFlow/Keras
- All models support saving/loading for persistence

**Evaluator (`evaluate.py`)**
- Computes accuracy, precision, recall, F1-score
- Generates confusion matrices with visualization
- Prints detailed classification reports
- Saves results to files

**Main Pipeline (`main.py`)**
- Complete end-to-end pipeline: load → preprocess → train → evaluate
- Supports training individual models or all models
- Handles missing dependencies gracefully (e.g., TensorFlow)
- Comprehensive logging and progress tracking

#### 3. Dataset

**IMDB Dataset**
- 50,000 movie reviews with binary sentiment labels
- Columns: `review` (text) and `sentiment` (positive/negative)
- Balanced dataset (25,000 positive, 25,000 negative reviews)
- All reviews in English

#### 4. Initial Results

Tested on 1,000 sample reviews (800 train, 100 validation, 100 test):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 81.00% | 82.36% | 81.00% | 80.63% |
| SVM | 84.00% | 84.39% | 84.00% | 83.88% |
| Neural Network | *Requires TensorFlow* | | | |

*Note: Neural Network training requires TensorFlow installation*

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not automatically downloaded):
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Optional: Install TensorFlow for Neural Network model**:
   ```bash
   pip install tensorflow
   ```

---

## Usage

### Basic Usage

Train all models on the full dataset:
```bash
python3 src/main.py --data "IMDB Dataset.csv" --sentiment-col sentiment --train-all
```

Train a specific model:
```bash
# Logistic Regression only
python3 src/main.py --data "IMDB Dataset.csv" --sentiment-col sentiment --model lr

# SVM only
python3 src/main.py --data "IMDB Dataset.csv" --sentiment-col sentiment --model svm

# Neural Network only (requires TensorFlow)
python3 src/main.py --data "IMDB Dataset.csv" --sentiment-col sentiment --model nn
```

### Command Line Arguments

- `--data`: Path to CSV file with reviews (required)
- `--text-col`: Name of text column (default: "review")
- `--sentiment-col`: Name of sentiment column (e.g., "sentiment" for positive/negative labels)
- `--rating-col`: Name of rating column (for numeric star ratings)
- `--threshold`: Rating threshold for binary classification (default: 3.0, only for numeric ratings)
- `--model`: Model type to train ("lr", "svm", or "nn")
- `--train-all`: Train all available models
- `--max-samples`: Limit number of samples (useful for testing)
- `--model-dir`: Directory to save models (default: "models")
- `--results-dir`: Directory to save results (default: "results")

### Example: Quick Test Run

Test with a small sample:
```bash
python3 src/main.py --data "IMDB Dataset.csv" --sentiment-col sentiment --train-all --max-samples 1000
```

---

## Features Implemented

### Data Handling
- ✅ CSV file loading with auto-detection
- ✅ Support for sentiment labels (positive/negative)
- ✅ Support for numeric star ratings
- ✅ Train/validation/test splitting with stratification
- ✅ Data balancing and missing value handling

### Text Preprocessing
- ✅ Text cleaning (URLs, emails, special characters)
- ✅ Tokenization (NLTK with fallback)
- ✅ Stop word removal
- ✅ Lemmatization
- ✅ Stemming (optional)
- ✅ Lowercasing

### Models
- ✅ Logistic Regression with TF-IDF vectorization
- ✅ Support Vector Machine (linear kernel)
- ✅ Neural Network (multi-layer perceptron)
- ✅ Model persistence (save/load)

### Evaluation
- ✅ Accuracy, Precision, Recall, F1-score
- ✅ Confusion matrix visualization
- ✅ Classification reports
- ✅ Results saving

### Infrastructure
- ✅ Modular code structure
- ✅ Comprehensive logging
- ✅ Error handling
- ✅ Graceful handling of missing dependencies

---

## Next Steps (Future Sessions)

- [ ] Train on full 50K dataset
- [ ] Hyperparameter tuning for all models
- [ ] Experiment with different preprocessing techniques
- [ ] Compare different vectorization methods (TF-IDF vs. Count vs. Word2Vec)
- [ ] Implement morphological analysis features
- [ ] Fine-tune neural network architecture
- [ ] Cross-validation for robust evaluation
- [ ] Error analysis and misclassification review
- [ ] Model interpretation and feature importance
- [ ] Final report and documentation

---

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `nltk` - Natural language processing
- `matplotlib` / `seaborn` - Visualization
- `tensorflow` (optional) - Neural network model

---

## Notes

- The IMDB Dataset CSV file is excluded from git (see `.gitignore`)
- Models are saved in the `models/` directory
- Evaluation results (confusion matrices, metrics) are saved in `results/`
- The project uses Python 3.8+ features

---

## References

1. Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. *Proceedings of the ACL-02 conference on Empirical methods in natural language processing*, 79–86.

2. Pang, B., & Lee, L. (2008). Opinion Mining and Sentiment Analysis. *Foundations and Trends® in Information Retrieval*, 2(1–2), 1–135.

3. Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142–150.

4. Socher, R., et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, 1631–1642.

5. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1746–1751.

---

## License

This project is for academic purposes as part of COMP 4750 coursework.

