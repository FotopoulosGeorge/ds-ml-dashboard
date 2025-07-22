# src/ml/ensemble/pipeline_builder.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.ml.performance_decorator import ml_performance
import warnings
warnings.filterwarnings('ignore')

class PipelineBuilder:
    """
    Build complex ML pipelines with preprocessing, feature selection, and model chaining
    """
    
    def __init__(self):
        self.pipeline = None
        self.pipeline_steps = []
        self.pipeline_config = {}
        
    def render_pipeline_tab(self, df):
        """
        Main interface for building ML pipelines
        """
        st.header("🔧 **ML Pipeline Builder**")
        st.markdown("*Build end-to-end ML pipelines with preprocessing, feature selection, and model training*")
        
        # Data validation
        if df.empty:
            st.warning("⚠️ No data available for pipeline building")
            return
        
        # Pipeline overview
        st.subheader("🏗️ Pipeline Configuration")
        
        # Get column information
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            target_column = st.selectbox(
                "**Target Variable:**",
                all_cols,
                key="pipeline_target"
            )
            
            # Auto-detect problem type
            if target_column:
                problem_type = self._detect_problem_type(df[target_column])
                st.info(f"**Detected Problem:** {problem_type}")
        
        with config_col2:
            feature_columns = st.multiselect(
                "**Feature Columns:**",
                [col for col in all_cols if col != target_column],
                default=[col for col in all_cols if col != target_column][:10],
                key="pipeline_features"
            )
        
        if not feature_columns:
            st.warning("Please select at least one feature column")
            return
        
        # Pipeline steps configuration
        st.subheader("🔗 Pipeline Steps")
        
        # Step 1: Data Preprocessing
        with st.expander("1️⃣ **Data Preprocessing**", expanded=True):
            self._configure_preprocessing_step(numeric_cols, categorical_cols, feature_columns)
        
        # Step 2: Feature Selection
        with st.expander("2️⃣ **Feature Selection**", expanded=True):
            self._configure_feature_selection_step(problem_type)
        
        # Step 3: Model Selection
        with st.expander("3️⃣ **Model Selection**", expanded=True):
            self._configure_model_step(problem_type)
        
        # Step 4: Hyperparameter Tuning
        with st.expander("4️⃣ **Hyperparameter Tuning**", expanded=False):
            self._configure_hyperparameter_tuning()
        
        # Pipeline execution
        st.subheader("🚀 Build & Execute Pipeline")
        
        exec_col1, exec_col2 = st.columns(2)
        
        with exec_col1:
            pipeline_name = st.text_input(
                "**Pipeline Name:**",
                value=f"Pipeline_{problem_type}_{datetime.now().strftime('%H%M')}",
                key="pipeline_name"
            )
        
        with exec_col2:
            validation_strategy = st.selectbox(
                "**Validation Strategy:**",
                ["Train-Test Split", "Cross-Validation", "Hold-out Validation"],
                key="validation_strategy"
            )
        
        # Build pipeline button
        if st.button("🔧 **Build & Execute Pipeline**", type="primary", key="build_pipeline"):
            try:
                with st.spinner('Building and executing pipeline...'):
                    pipeline_result = self._build_and_execute_pipeline(
                        df, target_column, feature_columns, problem_type, validation_strategy
                    )
                
                if pipeline_result:
                    self._display_pipeline_results(pipeline_result, pipeline_name)
                    
                    # Store pipeline
                    if 'ml_pipelines' not in st.session_state:
                        st.session_state.ml_pipelines = {}
                    
                    st.session_state.ml_pipelines[pipeline_name] = {
                        'pipeline': pipeline_result['pipeline'],
                        'config': pipeline_result['config'],
                        'performance': pipeline_result['performance'],
                        'problem_type': problem_type,
                        'created_at': datetime.now()
                    }
                    
                    st.success(f"✅ Pipeline '{pipeline_name}' built and saved!")
                    
            except Exception as e:
                st.error(f"❌ Pipeline execution failed: {str(e)}")
                st.info("💡 Check your configuration and data compatibility")
    
    def _detect_problem_type(self, target_series):
        """Auto-detect if problem is classification or regression"""
        unique_values = target_series.nunique()
        if target_series.dtype == 'object' or unique_values <= 10:
            return 'classification'
        else:
            return 'regression'
    
    def _configure_preprocessing_step(self, numeric_cols, categorical_cols, feature_columns):
        """Configure preprocessing steps"""
        st.markdown("**🧹 Data Preprocessing Configuration**")
        
        # Missing value handling
        missing_col1, missing_col2 = st.columns(2)
        
        with missing_col1:
            numeric_missing_strategy = st.selectbox(
                "**Numeric Missing Values:**",
                ["mean", "median", "constant", "drop"],
                key="numeric_missing"
            )
            
            if numeric_missing_strategy == "constant":
                numeric_fill_value = st.number_input(
                    "Fill value:",
                    value=0.0,
                    key="numeric_fill_value"
                )
        
        with missing_col2:
            categorical_missing_strategy = st.selectbox(
                "**Categorical Missing Values:**",
                ["most_frequent", "constant", "drop"],
                key="categorical_missing"
            )
            
            if categorical_missing_strategy == "constant":
                categorical_fill_value = st.text_input(
                    "Fill value:",
                    value="missing",
                    key="categorical_fill_value"
                )
        
        # Scaling for numeric features
        scaling_col1, scaling_col2 = st.columns(2)
        
        with scaling_col1:
            numeric_features_in_selection = [col for col in feature_columns if col in numeric_cols]
            if numeric_features_in_selection:
                numeric_scaling = st.selectbox(
                    "**Numeric Scaling:**",
                    ["none", "standard", "minmax", "robust"],
                    key="numeric_scaling"
                )
        
        with scaling_col2:
            # Encoding for categorical features
            categorical_features_in_selection = [col for col in feature_columns if col in categorical_cols]
            if categorical_features_in_selection:
                categorical_encoding = st.selectbox(
                    "**Categorical Encoding:**",
                    ["onehot", "label", "target", "frequency"],
                    key="categorical_encoding"
                )
        
        # Store preprocessing config
        st.session_state.preprocessing_config = {
            'numeric_missing': numeric_missing_strategy,
            'categorical_missing': categorical_missing_strategy,
            'numeric_scaling': numeric_scaling if 'numeric_scaling' in locals() else 'none',
            'categorical_encoding': categorical_encoding if 'categorical_encoding' in locals() else 'onehot',
            'numeric_fill_value': numeric_fill_value if 'numeric_fill_value' in locals() else 0,
            'categorical_fill_value': categorical_fill_value if 'categorical_fill_value' in locals() else 'missing'
        }
    
    def _configure_feature_selection_step(self, problem_type):
        """Configure feature selection"""
        st.markdown("**🎯 Feature Selection Configuration**")
        
        feature_selection_enabled = st.checkbox(
            "**Enable Feature Selection**",
            value=False,
            key="enable_feature_selection"
        )
        
        if feature_selection_enabled:
            selection_col1, selection_col2 = st.columns(2)
            
            with selection_col1:
                selection_method = st.selectbox(
                    "**Selection Method:**",
                    ["univariate", "model_based", "recursive", "pca"],
                    format_func=lambda x: {
                        "univariate": "Univariate (SelectKBest)",
                        "model_based": "Model-based (SelectFromModel)",
                        "recursive": "Recursive (RFE)",
                        "pca": "PCA Dimensionality Reduction"
                    }[x],
                    key="selection_method"
                )
            
            with selection_col2:
                if selection_method in ["univariate", "recursive"]:
                    n_features = st.slider(
                        "**Number of Features:**",
                        min_value=1,
                        max_value=20,
                        value=10,
                        key="n_features_select"
                    )
                elif selection_method == "pca":
                    pca_components = st.slider(
                        "**PCA Components:**",
                        min_value=1,
                        max_value=20,
                        value=5,
                        key="pca_components"
                    )
            
            # Advanced feature selection options
            if selection_method == "univariate":
                score_func = st.selectbox(
                    "**Scoring Function:**",
                    ["f_classif", "f_regression", "chi2"] if problem_type == "classification" else ["f_regression"],
                    key="score_func"
                )
            elif selection_method == "model_based":
                selection_estimator = st.selectbox(
                    "**Selection Estimator:**",
                    ["random_forest", "lasso", "tree"],
                    key="selection_estimator"
                )
        
        # Store feature selection config
        st.session_state.feature_selection_config = {
            'enabled': feature_selection_enabled,
            'method': selection_method if feature_selection_enabled else None,
            'n_features': n_features if 'n_features' in locals() else 10,
            'pca_components': pca_components if 'pca_components' in locals() else 5,
            'score_func': score_func if 'score_func' in locals() else 'f_classif',
            'selection_estimator': selection_estimator if 'selection_estimator' in locals() else 'random_forest'
        }
    
    def _configure_model_step(self, problem_type):
        """Configure model selection"""
        st.markdown("**🤖 Model Configuration**")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            if problem_type == "classification":
                model_type = st.selectbox(
                    "**Model Type:**",
                    ["logistic_regression", "random_forest", "svm", "gradient_boosting", "neural_network"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="model_type"
                )
            else:
                model_type = st.selectbox(
                    "**Model Type:**",
                    ["linear_regression", "random_forest", "svr", "gradient_boosting", "neural_network"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="model_type"
                )
        
        with model_col2:
            model_complexity = st.selectbox(
                "**Model Complexity:**",
                ["simple", "moderate", "complex"],
                help="Controls default hyperparameters",
                key="model_complexity"
            )
        
        # Model-specific parameters
        if model_type in ["random_forest"]:
            rf_col1, rf_col2 = st.columns(2)
            with rf_col1:
                n_estimators = st.slider("N Estimators:", 10, 200, 100, key="rf_n_estimators")
            with rf_col2:
                max_depth = st.selectbox("Max Depth:", [None, 5, 10, 15, 20], key="rf_max_depth")
        
        elif model_type in ["gradient_boosting"]:
            gb_col1, gb_col2 = st.columns(2)
            with gb_col1:
                learning_rate = st.slider("Learning Rate:", 0.01, 0.3, 0.1, key="gb_learning_rate")
            with gb_col2:
                n_estimators_gb = st.slider("N Estimators:", 50, 300, 100, key="gb_n_estimators")
        
        # Store model config
        st.session_state.model_config = {
            'type': model_type,
            'complexity': model_complexity,
            'n_estimators': n_estimators if 'n_estimators' in locals() else 100,
            'max_depth': max_depth if 'max_depth' in locals() else None,
            'learning_rate': learning_rate if 'learning_rate' in locals() else 0.1,
            'n_estimators_gb': n_estimators_gb if 'n_estimators_gb' in locals() else 100
        }
    
    def _configure_hyperparameter_tuning(self):
        """Configure hyperparameter tuning"""
        st.markdown("**⚙️ Hyperparameter Tuning Configuration**")
        
        tuning_enabled = st.checkbox(
            "**Enable Hyperparameter Tuning**",
            value=False,
            key="enable_tuning"
        )
        
        if tuning_enabled:
            tuning_col1, tuning_col2 = st.columns(2)
            
            with tuning_col1:
                tuning_method = st.selectbox(
                    "**Tuning Method:**",
                    ["grid_search", "random_search"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="tuning_method"
                )
            
            with tuning_col2:
                cv_folds = st.slider(
                    "**CV Folds:**",
                    min_value=3,
                    max_value=10,
                    value=5,
                    key="tuning_cv_folds"
                )
            
            if tuning_method == "random_search":
                n_iter = st.slider(
                    "**Number of Iterations:**",
                    min_value=10,
                    max_value=100,
                    value=20,
                    key="random_search_iter"
                )
        
        # Store tuning config
        st.session_state.tuning_config = {
            'enabled': tuning_enabled,
            'method': tuning_method if tuning_enabled else None,
            'cv_folds': cv_folds if 'cv_folds' in locals() else 5,
            'n_iter': n_iter if 'n_iter' in locals() else 20
        }
    @ml_performance(
        "ensemble", 
        dataset_param="df", 
        config_params=["problem_type", "validation_strategy"]
    )
    def _build_and_execute_pipeline(self, df, target_column, feature_columns, problem_type, validation_strategy):
        """Build and execute the complete pipeline"""
        
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values in target
        complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[complete_mask]
        y = y[complete_mask]
        
        if len(X) == 0:
            raise ValueError("No complete cases found after removing missing values")
        
        # Get configurations
        preprocess_config = st.session_state.get('preprocessing_config', {})
        feature_config = st.session_state.get('feature_selection_config', {})
        model_config = st.session_state.get('model_config', {})
        tuning_config = st.session_state.get('tuning_config', {})
        
        # Build pipeline steps
        pipeline_steps = []
        
        # 1. Preprocessing
        preprocessor = self._build_preprocessor(X.columns, preprocess_config)
        if preprocessor is not None:
            pipeline_steps.append(('preprocessor', preprocessor))
        
        # 2. Feature Selection
        if feature_config.get('enabled', False):
            feature_selector = self._build_feature_selector(feature_config, problem_type)
            if feature_selector is not None:
                pipeline_steps.append(('feature_selector', feature_selector))
        
        # 3. Model
        model = self._build_model(model_config, problem_type)
        pipeline_steps.append(('model', model))
        
        # Create pipeline
        pipeline = Pipeline(pipeline_steps)
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if problem_type == 'classification' and len(np.unique(y)) > 1 else None
        )
        
        # Hyperparameter tuning
        if tuning_config.get('enabled', False):
            pipeline = self._tune_hyperparameters(pipeline, X_train, y_train, tuning_config, problem_type)
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Evaluate pipeline
        performance = self._evaluate_pipeline(pipeline, X_test, y_test, problem_type, validation_strategy)
        
        # Get feature names after preprocessing
        feature_names = self._get_feature_names_after_preprocessing(pipeline, X.columns)
        
        result = {
            'pipeline': pipeline,
            'config': {
                'preprocessing': preprocess_config,
                'feature_selection': feature_config,
                'model': model_config,
                'tuning': tuning_config
            },
            'performance': performance,
            'problem_type': problem_type,
            'feature_names': feature_names,
            'test_data': (X_test, y_test),
            'training_data': (X_train, y_train)
        }
        
        return result
    
    def _build_preprocessor(self, feature_columns, config):
        """Build preprocessing pipeline"""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        
        transformers = []
        
        # Identify column types
        numeric_features = []
        categorical_features = []
        
        for col in feature_columns:
            # This is simplified - in practice you'd check actual dtypes
            if 'numeric' in str(col).lower() or any(char.isdigit() for char in str(col)):
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        # Numeric pipeline
        if numeric_features:
            numeric_steps = []
            
            # Imputation
            if config.get('numeric_missing') == 'mean':
                numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif config.get('numeric_missing') == 'median':
                numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
            elif config.get('numeric_missing') == 'constant':
                numeric_steps.append(('imputer', SimpleImputer(strategy='constant', 
                                                               fill_value=config.get('numeric_fill_value', 0))))
            
            # Scaling
            if config.get('numeric_scaling') == 'standard':
                numeric_steps.append(('scaler', StandardScaler()))
            elif config.get('numeric_scaling') == 'minmax':
                numeric_steps.append(('scaler', MinMaxScaler()))
            elif config.get('numeric_scaling') == 'robust':
                numeric_steps.append(('scaler', RobustScaler()))
            
            if numeric_steps:
                numeric_pipeline = Pipeline(numeric_steps)
                transformers.append(('numeric', numeric_pipeline, numeric_features))
        
        # Categorical pipeline
        if categorical_features:
            categorical_steps = []
            
            # Imputation
            if config.get('categorical_missing') == 'most_frequent':
                categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            elif config.get('categorical_missing') == 'constant':
                categorical_steps.append(('imputer', SimpleImputer(strategy='constant',
                                                                   fill_value=config.get('categorical_fill_value', 'missing'))))
            
            # Encoding
            if config.get('categorical_encoding') == 'onehot':
                categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            
            if categorical_steps:
                categorical_pipeline = Pipeline(categorical_steps)
                transformers.append(('categorical', categorical_pipeline, categorical_features))
        
        if transformers:
            return ColumnTransformer(transformers, remainder='passthrough')
        
        return None
    
    def _build_feature_selector(self, config, problem_type):
        """Build feature selection step"""
        if not config.get('enabled', False):
            return None
        
        method = config.get('method')
        
        if method == 'univariate':
            score_func = f_classif if problem_type == 'classification' else f_regression
            return SelectKBest(score_func=score_func, k=config.get('n_features', 10))
        
        elif method == 'pca':
            return PCA(n_components=config.get('pca_components', 5))
        
        elif method == 'model_based':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if problem_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            return SelectFromModel(estimator)
        
        elif method == 'recursive':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if problem_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            return RFE(estimator, n_features_to_select=config.get('n_features', 10))
        
        return None
    
    def _build_model(self, config, problem_type):
        """Build model based on configuration"""
        model_type = config.get('type')
        
        if problem_type == 'classification':
            if model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=42, max_iter=1000)
            
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth'),
                    random_state=42
                )
            
            elif model_type == 'svm':
                from sklearn.svm import SVC
                return SVC(random_state=42, probability=True)
            
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=config.get('n_estimators_gb', 100),
                    learning_rate=config.get('learning_rate', 0.1),
                    random_state=42
                )
        
        else:  # regression
            if model_type == 'linear_regression':
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth'),
                    random_state=42
                )
            
            elif model_type == 'svr':
                from sklearn.svm import SVR
                return SVR()
            
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=config.get('n_estimators_gb', 100),
                    learning_rate=config.get('learning_rate', 0.1),
                    random_state=42
                )
        
        # Default fallback
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=42)
    
    def _tune_hyperparameters(self, pipeline, X_train, y_train, tuning_config, problem_type):
        """Perform hyperparameter tuning"""
        # Define parameter grids (simplified)
        param_grid = self._get_parameter_grid(pipeline, problem_type)
        
        if tuning_config.get('method') == 'grid_search':
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=tuning_config.get('cv_folds', 5),
                scoring='accuracy' if problem_type == 'classification' else 'r2',
                n_jobs=-1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=tuning_config.get('n_iter', 20),
                cv=tuning_config.get('cv_folds', 5),
                scoring='accuracy' if problem_type == 'classification' else 'r2',
                n_jobs=-1,
                random_state=42
            )
        
        search.fit(X_train, y_train)
        return search.best_estimator_
    
    def _get_parameter_grid(self, pipeline, problem_type):
        """Get parameter grid for hyperparameter tuning"""
        # Simplified parameter grids
        param_grid = {}
        
        # Check if pipeline has a model step
        step_names = [name for name, _ in pipeline.steps]
        
        if 'model' in step_names:
            model = pipeline.named_steps['model']
            model_type = type(model).__name__
            
            if 'RandomForest' in model_type:
                param_grid.update({
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 5, 10, 15],
                    'model__min_samples_split': [2, 5, 10]
                })
            
            elif 'LogisticRegression' in model_type:
                param_grid.update({
                    'model__C': [0.1, 1.0, 10.0],
                    'model__solver': ['liblinear', 'lbfgs']
                })
            
            elif 'SV' in model_type:  # SVC or SVR
                param_grid.update({
                    'model__C': [0.1, 1.0, 10.0],
                    'model__kernel': ['rbf', 'linear']
                })
        
        # Feature selection parameters
        if 'feature_selector' in step_names:
            selector = pipeline.named_steps['feature_selector']
            if 'SelectKBest' in type(selector).__name__:
                param_grid.update({
                    'feature_selector__k': [5, 10, 15, 20]
                })
            elif 'PCA' in type(selector).__name__:
                param_grid.update({
                    'feature_selector__n_components': [3, 5, 10, 15]
                })
        
        return param_grid
    
    def _evaluate_pipeline(self, pipeline, X_test, y_test, problem_type, validation_strategy):
        """Evaluate pipeline performance"""
        y_pred = pipeline.predict(X_test)
        
        performance = {}
        
        if problem_type == 'classification':
            performance['accuracy'] = accuracy_score(y_test, y_pred)
            
            # Additional metrics
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                performance['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                performance['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                performance['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            except:
                pass
        
        else:  # regression
            performance['r2'] = r2_score(y_test, y_pred)
            performance['mse'] = mean_squared_error(y_test, y_pred)
            performance['rmse'] = np.sqrt(performance['mse'])
            
            # Additional metrics
            try:
                from sklearn.metrics import mean_absolute_error
                performance['mae'] = mean_absolute_error(y_test, y_pred)
            except:
                pass
        
        # Cross-validation
        if validation_strategy in ["Cross-Validation", "Both"]:
            try:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(
                    pipeline, X_test, y_test, cv=5,
                    scoring='accuracy' if problem_type == 'classification' else 'r2'
                )
                performance['cv_mean'] = cv_scores.mean()
                performance['cv_std'] = cv_scores.std()
            except:
                pass
        
        return performance
    
    def _get_feature_names_after_preprocessing(self, pipeline, original_features):
        """Get feature names after preprocessing"""
        try:
            # This is simplified - getting feature names after preprocessing can be complex
            if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
                preprocessor = pipeline.named_steps['preprocessor']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    return preprocessor.get_feature_names_out()
            
            return original_features.tolist()
        except:
            return original_features.tolist()
    
    def _display_pipeline_results(self, pipeline_result, pipeline_name):
        """Display comprehensive pipeline results"""
        st.success("✅ Pipeline built and executed successfully!")
        
        pipeline = pipeline_result['pipeline']
        performance = pipeline_result['performance']
        config = pipeline_result['config']
        
        # Performance overview
        st.subheader("📊 Pipeline Performance")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Pipeline Steps", len(pipeline.steps))
        
        with perf_col2:
            if 'accuracy' in performance:
                st.metric("Accuracy", f"{performance['accuracy']:.4f}")
            elif 'r2' in performance:
                st.metric("R² Score", f"{performance['r2']:.4f}")
        
        with perf_col3:
            if 'cv_mean' in performance:
                st.metric("CV Score", f"{performance['cv_mean']:.4f} ± {performance['cv_std']:.4f}")
        
        with perf_col4:
            st.metric("Problem Type", pipeline_result['problem_type'].title())
        
        # Detailed performance
        st.subheader("📈 Detailed Performance Metrics")
        
        perf_data = []
        for metric, value in performance.items():
            if metric not in ['cv_mean', 'cv_std']:
                perf_data.append({
                    'Metric': metric.upper(),
                    'Value': f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
        
        # Pipeline architecture
        st.subheader("🏗️ Pipeline Architecture")
        
        arch_data = []
        for i, (step_name, step_obj) in enumerate(pipeline.steps, 1):
            arch_data.append({
                'Step': i,
                'Name': step_name.replace('_', ' ').title(),
                'Component': type(step_obj).__name__,
                'Parameters': str(step_obj.get_params())[:100] + "..." if len(str(step_obj.get_params())) > 100 else str(step_obj.get_params())
            })
        
        arch_df = pd.DataFrame(arch_data)
        st.dataframe(arch_df, use_container_width=True)
        
        # Configuration details
        with st.expander("⚙️ **Pipeline Configuration Details**"):
            for config_type, config_details in config.items():
                st.write(f"**{config_type.title()}:**")
                for key, value in config_details.items():
                    st.write(f"  • {key}: {value}")
        
        # Feature importance (if available)
        if hasattr(pipeline.named_steps.get('model'), 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            feature_names = pipeline_result.get('feature_names', [])
            importances = pipeline.named_steps['model'].feature_importances_
            
            if len(feature_names) == len(importances):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Feature Importances'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Save pipeline
        st.subheader("💾 Save Pipeline")
        
        save_col1, save_col2 = st.columns(2)
        
        with save_col1:
            if st.button("💾 **Save as Trained Model**", key="save_pipeline_model"):
                try:
                    model_id = f"pipeline_{pipeline_name}"
                    
                    pipeline_info = {
                        'model': pipeline,
                        'algorithm': "ML Pipeline",
                        'target': 'pipeline_target',
                        'features': pipeline_result.get('feature_names', []),
                        'problem_type': pipeline_result['problem_type'],
                        'test_data': pipeline_result.get('test_data', (None, None)),
                        'model_id': model_id,
                        'pipeline_info': pipeline_result
                    }
                    
                    st.session_state.trained_models[model_id] = pipeline_info
                    st.success(f"✅ Pipeline saved as: {model_id}")
                    st.info("💡 Now available for predictions and analysis")
                    
                except Exception as e:
                    st.error(f"❌ Failed to save: {str(e)}")
        
        with save_col2:
            # Download configuration
            import json
            config_data = {
                'pipeline_name': pipeline_name,
                'steps': [{'name': name, 'type': type(obj).__name__} for name, obj in pipeline.steps],
                'configuration': config,
                'performance': performance,
                'created_at': datetime.now().isoformat()
            }
            
            config_json = json.dumps(config_data, indent=2, default=str)
            
            st.download_button(
                label="📥 **Download Config**",
                data=config_json,
                file_name=f"{pipeline_name}_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )