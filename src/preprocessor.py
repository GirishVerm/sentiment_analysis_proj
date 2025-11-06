"""
Text preprocessing module for movie reviews.

Implements standard NLP preprocessing techniques:
- Tokenization
- Stemming/Lemmatization
- Stop word removal
- Feature extraction
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        # Try punkt_tab for newer NLTK versions
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Preprocesses text for sentiment analysis."""
    
    def __init__(self, use_stemming: bool = False, use_lemmatization: bool = True,
                 remove_stopwords: bool = True, lowercase: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_stemming: Whether to use stemming
            use_lemmatization: Whether to use lemmatization
            remove_stopwords: Whether to remove stop words
            lowercase: Whether to convert to lowercase
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text string
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except LookupError:
            # Try downloading punkt_tab if missing
            try:
                nltk.download('punkt_tab', quiet=True)
                tokens = word_tokenize(text)
                return tokens
            except:
                logger.warning("NLTK tokenizer not available, using simple split")
                return text.split()
        except Exception as e:
            logger.warning(f"Tokenization error: {e}, using simple split")
            return text.split()
    
    def process_tokens(self, tokens: List[str]) -> List[str]:
        """
        Process tokens (stemming, lemmatization, stopword removal).
        
        Args:
            tokens: List of token strings
            
        Returns:
            Processed tokens
        """
        processed = []
        
        for token in tokens:
            # Remove stop words
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Apply stemming
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            
            # Apply lemmatization (takes precedence over stemming)
            if self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            processed.append(token)
        
        return processed
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Process tokens
        tokens = self.process_tokens(tokens)
        
        # Join back into string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

