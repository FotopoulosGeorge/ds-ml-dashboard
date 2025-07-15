# src/ml/ml_utils.py
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.demo.demo_datasets import DemoDatasets

class MLUtils:
    """
    Utility functions for ML operations - data prep, model management, helpers
    """
    
    @staticmethod
    def detect_problem_type(target_series, threshold=10):
        """
        Auto-detect if the problem is classification or regression
        
        Parameters:
        -----------
        target_series : pd.Series
            The target variable
        threshold : int
            If unique values <= threshold, treat as classification
        
        Returns:
        --------
        str : 'classification' or 'regression'
        """
        # Remove missing values for analysis
        clean_target = target_series.dropna()
        
        if len(clean_target) == 0:
            return 'classification'  # Default
        
        # Check data type first
        if clean_target.dtype == 'object' or pd.api.types.is_categorical_dtype(clean_target):
            return 'classification'
        
        # For numeric data, check number of unique values
        unique_values = clean_target.nunique()
        
        if unique_values <= threshold:
            return 'classification'
        else:
            # Additional check: if all values are integers and range is small
            if clean_target.dtype in ['int64', 'int32'] and unique_values <= 20:
                return 'classification'
            return 'regression'
    
    @staticmethod
    def get_available_algorithms(problem_type):
        """
        Get available algorithms for the problem type
        
        Parameters:
        -----------
        problem_type : str
            'classification', 'regression', or 'clustering'
        
        Returns:
        --------
        dict : Algorithm name -> Class mapping
        """
        if problem_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression,
                'Random Forest': RandomForestClassifier,
                'Support Vector Machine': SVC,
                'Decision Tree': DecisionTreeClassifier,
                'K-Nearest Neighbors': KNeighborsClassifier,
                'Naive Bayes': GaussianNB
            }
        
        elif problem_type == 'regression':
            return {
                'Linear Regression': LinearRegression,
                'Random Forest': RandomForestRegressor,
                'Support Vector Regression': SVR,
                'Decision Tree': DecisionTreeRegressor,
                'K-Nearest Neighbors': KNeighborsRegressor
            }
        
        else:
            return {}
    
    @staticmethod
    def prepare_ml_data(df, target_col, feature_cols, test_size=0.2, random_state=42, 
                       handle_missing='drop', scale_features=False):
        """
        Comprehensive data preparation for ML
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        feature_cols : list
            List of feature column names
        test_size : float
            Proportion of data for testing
        random_state : int
            Random state for reproducibility
        handle_missing : str
            How to handle missing values ('drop', 'fill_mean', 'fill_median')
        scale_features : bool
            Whether to scale features
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, scaler)
        """
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values in target
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Handle missing values in features
        if handle_missing == 'drop':
            # Drop rows with any missing values
            complete_indices = ~X.isnull().any(axis=1)
            X = X[complete_indices]
            y = y[complete_indices]
        
        elif handle_missing == 'fill_mean':
            # Fill with mean for numeric columns
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
        
        elif handle_missing == 'fill_median':
            # Fill with median for numeric columns
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Feature scaling if requested
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame to maintain column names
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train, X_test, y_train, y_test, scaler
    
    @staticmethod
    def get_default_params(algorithm_name):
        """
        Get default hyperparameters for algorithms
        
        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm
        
        Returns:
        --------
        dict : Default parameters
        """
        default_params = {
            'Logistic Regression': {
                'max_iter': 1000,
                'random_state': 42
            },
            'Random Forest': {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            },
            'Support Vector Machine': {
                'kernel': 'rbf',
                'random_state': 42
            },
            'Support Vector Regression': {
                'kernel': 'rbf'
            },
            'Decision Tree': {
                'max_depth': None,
                'random_state': 42
            },
            'K-Nearest Neighbors': {
                'n_neighbors': 5
            },
            'Naive Bayes': {},
            'Linear Regression': {}
        }
        
        return default_params.get(algorithm_name, {})
    
    @staticmethod
    def save_model(model, model_name, model_info=None):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        model : sklearn model
            Trained model object
        model_name : str
            Name for the saved model
        model_info : dict
            Additional model information to save
        
        Returns:
        --------
        str : File path where model was saved
        """
        # Create models directory if it doesn't exist
        models_dir = os.path.join('src', 'ml', 'models')
        os.makedirs(models_dir, exist_ok=True)
        from datetime import datetime
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = os.path.join(models_dir, filename)
        # Check if in demo mode
        if DemoDatasets.is_deployed():
        # Demo mode: Show info but don't actually save
            st.warning("🌐 **Demo Mode**: Models cannot be permanently saved on cloud deployment")
            st.info("""
            💡 **In Demo Mode:**
            - Model is saved in session memory only
            - Available for predictions during this session
            - Download model info as backup
            - For permanent saving, run app locally
            """)
        
        # Provide model info as downloadable content instead
            if model_info:
                import json
                from datetime import datetime
                
                # Create model summary for download
                model_summary = {
                    'model_type': model_info.get('algorithm', 'Unknown'),
                    'target': model_info.get('target', 'N/A'),
                    'features': model_info.get('features', []),
                    'problem_type': model_info.get('problem_type', 'Unknown'),
                    'trained_on': 'Demo Dataset',
                    'created_at': datetime.now().isoformat(),
                    'note': 'This model was trained on demo data. To save actual model files, run the app locally.'
                }
                
                model_json = json.dumps(model_summary, indent=2)
                
                st.download_button(
                    label="📥 Download Model Info",
                    data=model_json,
                    file_name=f"{model_name}_info.json",
                    mime="application/json",
                    help="Download model information for reference"
                )
            
            # Return a fake filepath for consistency
            return f"demo_mode_{model_name}.joblib"
        else:
            # Prepare data to save
            save_data = {
                'model': model,
                'model_info': model_info,
                'saved_at': datetime.now().isoformat(),
                'sklearn_version': None  # Could add sklearn version check
            }
            
            try:
                # Save model
                joblib.dump(save_data, filepath)
                return filepath
            except Exception as e:
                raise Exception(f"Failed to save model: {str(e)}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load saved model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file
        
        Returns:
        --------
        tuple : (model, model_info)
        """
        try:
            # Load model data
            save_data = joblib.load(filepath)
            
            model = save_data['model']
            model_info = save_data.get('model_info', {})
            
            return model, model_info
        
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    @staticmethod
    def list_saved_models():
        """
        List all saved models in the models directory
        
        Returns:
        --------
        list : List of model file information
        """
        models_dir = os.path.join('src', 'ml', 'models')
        
        if not os.path.exists(models_dir):
            return []
        
        model_files = []
        
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib'):
                filepath = os.path.join(models_dir, filename)
                try:
                    # Get file info
                    file_stats = os.stat(filepath)
                    created_time = datetime.fromtimestamp(file_stats.st_ctime)
                    file_size = file_stats.st_size
                    
                    model_files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'created': created_time,
                        'size_mb': file_size / (1024 * 1024)
                    })
                except Exception:
                    continue  # Skip files that can't be read
        
        # Sort by creation time (newest first)
        model_files.sort(key=lambda x: x['created'], reverse=True)
        
        return model_files
    
    @staticmethod
    def validate_data_for_ml(df, target_col, feature_cols):
        """
        Validate data before ML training
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        feature_cols : list
            List of feature column names
        
        Returns:
        --------
        dict : Validation results with warnings and recommendations
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'recommendations': [],
            'stats': {}
        }
        
        # Check if columns exist
        missing_cols = []
        if target_col not in df.columns:
            missing_cols.append(target_col)
        
        for col in feature_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            validation_results['valid'] = False
            validation_results['warnings'].append(f"Missing columns: {', '.join(missing_cols)}")
            return validation_results
        
        # Check data size
        total_samples = len(df)
        validation_results['stats']['total_samples'] = total_samples
        
        if total_samples < 10:
            validation_results['valid'] = False
            validation_results['warnings'].append("Too few samples for ML (minimum 10 recommended)")
        elif total_samples < 100:
            validation_results['warnings'].append("Small dataset - consider gathering more data")
        
        # Check for missing values
        target_missing = df[target_col].isnull().sum()
        feature_missing = df[feature_cols].isnull().sum().sum()
        
        validation_results['stats']['target_missing'] = target_missing
        validation_results['stats']['feature_missing'] = feature_missing
        
        if target_missing > 0:
            validation_results['warnings'].append(f"Target variable has {target_missing} missing values")
        
        if feature_missing > 0:
            missing_pct = (feature_missing / (len(df) * len(feature_cols))) * 100
            validation_results['warnings'].append(f"Features have {feature_missing} missing values ({missing_pct:.1f}%)")
        
        # Check for constant features
        constant_features = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            validation_results['warnings'].append(f"Constant features detected: {', '.join(constant_features)}")
            validation_results['recommendations'].append("Remove constant features before training")
        
        # Check target variable distribution
        target_unique = df[target_col].nunique()
        validation_results['stats']['target_unique'] = target_unique
        
        if target_unique == 1:
            validation_results['valid'] = False
            validation_results['warnings'].append("Target variable has only one unique value")
        elif target_unique == total_samples:
            validation_results['warnings'].append("Target variable has all unique values - check if this is correct")
        
        # Check for highly correlated features (if more than 1 feature)
        if len(feature_cols) > 1:
            try:
                corr_matrix = df[feature_cols].corr().abs()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.95:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if high_corr_pairs:
                    validation_results['warnings'].append(f"Highly correlated features detected: {high_corr_pairs}")
                    validation_results['recommendations'].append("Consider removing redundant features")
            
            except Exception:
                pass  # Skip correlation check if it fails
        
        # Data type recommendations
        non_numeric_features = []
        for col in feature_cols:
            if df[col].dtype not in ['int64', 'float64']:
                non_numeric_features.append(col)
        
        if non_numeric_features:
            validation_results['recommendations'].append(
                f"Non-numeric features detected: {', '.join(non_numeric_features)}. "
                "Consider encoding them first using Feature Engineering tab."
            )
        
        return validation_results
    
    @staticmethod
    def generate_model_summary(model, model_info):
        """
        Generate a comprehensive summary of a trained model
        
        Parameters:
        -----------
        model : sklearn model
            Trained model object
        model_info : dict
            Model information dictionary
        
        Returns:
        --------
        dict : Model summary information
        """
        summary = {
            'algorithm': model_info.get('algorithm', 'Unknown'),
            'problem_type': model_info.get('problem_type', 'Unknown'),
            'target_variable': model_info.get('target', 'Unknown'),
            'features': model_info.get('features', []),
            'n_features': len(model_info.get('features', [])),
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
            'training_samples': len(model_info['training_data'][0]) if 'training_data' in model_info else 'Unknown',
            'test_samples': len(model_info['test_data'][0]) if 'test_data' in model_info else 'Unknown'
        }
        
        # Add model-specific information
        if hasattr(model, 'feature_importances_'):
            summary['has_feature_importance'] = True
            summary['feature_importance'] = dict(zip(
                model_info.get('features', []), 
                model.feature_importances_
            ))
        else:
            summary['has_feature_importance'] = False
        
        if hasattr(model, 'coef_'):
            summary['has_coefficients'] = True
        else:
            summary['has_coefficients'] = False
        
        if hasattr(model, 'classes_'):
            summary['classes'] = model.classes_.tolist()
            summary['n_classes'] = len(model.classes_)
        
        return summary
    
    @staticmethod
    def create_sample_prediction_data(feature_cols, n_samples=5):
        """
        Create sample data for prediction demonstration
        
        Parameters:
        -----------
        feature_cols : list
            List of feature column names
        n_samples : int
            Number of sample rows to create
        
        Returns:
        --------
        pd.DataFrame : Sample data for predictions
        """
        sample_data = {}
        
        for col in feature_cols:
            # Generate random sample values
            # This is a simple implementation - in practice, you might want to use
            # the actual data distribution from the training set
            sample_data[col] = np.random.randn(n_samples)
        
        return pd.DataFrame(sample_data)
    
    @staticmethod
    def get_algorithm_info(algorithm_name):
        """
        Get detailed information about an algorithm
        
        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm
        
        Returns:
        --------
        dict : Algorithm information
        """
        algorithm_info = {
            'Logistic Regression': {
                'description': 'Linear model for classification using logistic function',
                'pros': ['Fast training', 'Interpretable', 'No hyperparameter tuning needed'],
                'cons': ['Assumes linear relationship', 'Sensitive to outliers'],
                'best_for': 'Binary classification with linear relationships'
            },
            'Random Forest': {
                'description': 'Ensemble of decision trees with voting',
                'pros': ['Handles non-linear relationships', 'Feature importance', 'Robust to outliers'],
                'cons': ['Can overfit with small datasets', 'Less interpretable'],
                'best_for': 'Most tabular data problems'
            },
            'Support Vector Machine': {
                'description': 'Finds optimal boundary between classes',
                'pros': ['Works well with high-dimensional data', 'Memory efficient'],
                'cons': ['Slow on large datasets', 'Requires feature scaling'],
                'best_for': 'Small to medium datasets with complex boundaries'
            },
            'Linear Regression': {
                'description': 'Linear relationship between features and target',
                'pros': ['Simple and interpretable', 'Fast training', 'No hyperparameters'],
                'cons': ['Assumes linear relationship', 'Sensitive to outliers'],
                'best_for': 'Continuous target with linear relationships'
            },
            'K-Nearest Neighbors': {
                'description': 'Predicts based on k nearest training examples',
                'pros': ['Simple concept', 'No assumptions about data distribution'],
                'cons': ['Slow predictions', 'Sensitive to irrelevant features'],
                'best_for': 'Small datasets with meaningful distance metrics'
            }
        }
        
        return algorithm_info.get(algorithm_name, {
            'description': 'Algorithm information not available',
            'pros': [],
            'cons': [],
            'best_for': 'Various use cases'
        })