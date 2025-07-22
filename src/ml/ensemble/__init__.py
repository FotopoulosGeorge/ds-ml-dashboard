# src/ml/ensemble/__init__.py
from .model_chaining import ModelChainer
from .ensemble_methods import EnsembleMethods
from .stacking import StackingEnsemble
from .pipeline_builder import PipelineBuilder

__version__ = "1.0.0"
__author__ = "George Fotopoulos"

__all__ = [
    'ModelChainer',
    'EnsembleMethods', 
    'StackingEnsemble',
    'PipelineBuilder'
]

# Available ensemble techniques
ENSEMBLE_TECHNIQUES = {
    'sequential_chaining': {
        'name': 'Sequential Model Chaining',
        'description': 'Chain models where output of one feeds into next',
        'use_cases': ['Multi-stage prediction', 'Hierarchical classification', 'Refinement pipelines'],
        'complexity': 'Medium'
    },
    'voting_ensemble': {
        'name': 'Voting Ensemble',
        'description': 'Combine predictions from multiple models through voting',
        'use_cases': ['Reduce overfitting', 'Improve robustness', 'Consensus predictions'],
        'complexity': 'Low'
    },
    'bagging_ensemble': {
        'name': 'Bagging Ensemble',
        'description': 'Bootstrap aggregating with multiple model instances',
        'use_cases': ['Variance reduction', 'Parallel training', 'Stable predictions'],
        'complexity': 'Medium'
    },
    'stacking_ensemble': {
        'name': 'Stacking/Stacked Generalization',
        'description': 'Meta-learner combines predictions from base models',
        'use_cases': ['Complex pattern capture', 'Model strength combination', 'Advanced ensembles'],
        'complexity': 'High'
    },
    'adaptive_boosting': {
        'name': 'Adaptive Boosting',
        'description': 'Sequential model training focusing on errors',
        'use_cases': ['Weak learner improvement', 'Sequential refinement', 'Error correction'],
        'complexity': 'High'
    },
    'pipeline_chaining': {
        'name': 'Processing Pipeline',
        'description': 'Chain preprocessing, feature selection, and models',
        'use_cases': ['Complete ML pipeline', 'Automated preprocessing', 'Feature engineering'],
        'complexity': 'Medium'
    }
}

def get_available_techniques():
    """Return dictionary of all available ensemble techniques"""
    return ENSEMBLE_TECHNIQUES

def get_technique_info(technique_name):
    """Get detailed information about a specific technique"""
    return ENSEMBLE_TECHNIQUES.get(technique_name, {
        'name': 'Unknown',
        'description': 'No information available',
        'use_cases': [],
        'complexity': 'Unknown'
    })

def check_ensemble_dependencies():
    """Check which ensemble dependencies are available"""
    deps = {
        'sklearn_ensemble': False,
        'xgboost': False,
        'lightgbm': False,
        'catboost': False
    }
    
    try:
        from sklearn.ensemble import VotingClassifier, BaggingClassifier
        deps['sklearn_ensemble'] = True
    except ImportError:
        pass
    
    try:
        import xgboost
        deps['xgboost'] = True
    except ImportError:
        pass
    
    try:
        import lightgbm
        deps['lightgbm'] = True
    except ImportError:
        pass
    
    try:
        import catboost
        deps['catboost'] = True
    except ImportError:
        pass
    
    return deps