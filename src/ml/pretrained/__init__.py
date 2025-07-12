# src/ml/pretrained/__init__.py
from .time_series import TimeSeriesForecaster
from .anomaly_detection import AnomalyDetector

# todo
# try:
#     from .text_analysis import TextAnalyzer
#     TEXT_ANALYSIS_AVAILABLE = True
# except ImportError:
#     TEXT_ANALYSIS_AVAILABLE = False

# try:
#     from .pattern_mining import PatternMiner
#     PATTERN_MINING_AVAILABLE = True
# except ImportError:
#     PATTERN_MINING_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "George Fotopoulos"

__all__ = [
    'TimeSeriesForecaster',
    'AnomalyDetector'
]

# todo
# if TEXT_ANALYSIS_AVAILABLE:
#     __all__.append('TextAnalyzer')

# if PATTERN_MINING_AVAILABLE:
#     __all__.append('PatternMiner')

# Available pretrained models
AVAILABLE_MODELS = {
    'time_series': {
        'Prophet': 'Facebook Prophet for business time series',
        'ARIMA': 'Classical statistical forecasting',
        'Seasonal': 'Seasonal decomposition and forecasting'
    },
    'anomaly_detection': {
        'Isolation Forest': 'Tree-based anomaly detection',
        'Local Outlier Factor': 'Density-based outlier detection',
        'One-Class SVM': 'Support vector anomaly detection'
    },
    'text_analysis': {
        'Sentiment Analysis': 'Basic sentiment classification',
        'Text Statistics': 'Text feature extraction',
        'Topic Modeling': 'Basic topic discovery'
    },
    'pattern_mining': {
        'Association Rules': 'Market basket analysis',
        'Frequent Patterns': 'Pattern discovery in transactions'
    }
}

def get_available_models():
    """Return dictionary of all available pretrained models"""
    return AVAILABLE_MODELS

def check_dependencies():
    """Check which optional dependencies are available"""
    deps = {
        'prophet': False,
        'textblob': False,
        'mlxtend': False,
        'nltk': False
    }
    
    try:
        import prophet
        deps['prophet'] = True
    except ImportError:
        pass
    
    try:
        import textblob
        deps['textblob'] = True
    except ImportError:
        pass
    
    try:
        import mlxtend
        deps['mlxtend'] = True
    except ImportError:
        pass
    
    try:
        import nltk
        deps['nltk'] = True
    except ImportError:
        pass
    
    return deps