from .ml_trainer import MLTrainer
from .ml_evaluator import MLEvaluator
from .ml_utils import MLUtils
from .automl import AutoMLEngine
from .ensemble import ModelChainer, EnsembleMethods, StackingEnsemble, PipelineBuilder

__version__ = "0.0.1"
__author__ = "George Fotopoulos"

__all__ = [
    'MLTrainer',
    'MLEvaluator', 
    'MLUtils',
    'AutoMLEngine',
    'ModelChainer',        
    'EnsembleMethods',      
    'StackingEnsemble',    
    'PipelineBuilder'
]

# Module metadata
ML_ALGORITHMS = {
    'classification': [
        'Logistic Regression',
        'Random Forest',
        'Support Vector Machine',
        'Decision Tree',
        'K-Nearest Neighbors',
        'Naive Bayes'
    ],
    'regression': [
        'Linear Regression',
        'Random Forest',
        'Support Vector Regression',
        'Decision Tree',
        'K-Nearest Neighbors'
    ],
    'clustering': [
        'K-Means',
        'DBSCAN',
        'Hierarchical Clustering'
    ],
    'automl': [
        'Automated Model Selection',
        'Hyperparameter Optimization',
        'Auto Feature Selection',
        'Auto Preprocessing'
    ],
    'ensemble': [
        'Sequential Model Chaining',
        'Voting Ensemble',
        'Bagging Ensemble', 
        'Stacking Ensemble',
        'Pipeline Chaining'
    ]
}

SUPPORTED_METRICS = {
    'classification': [
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'ROC-AUC'
    ],
    'regression': [
        'R² Score',
        'Mean Squared Error (MSE)',
        'Root Mean Squared Error (RMSE)',
        'Mean Absolute Error (MAE)'
    ]
}