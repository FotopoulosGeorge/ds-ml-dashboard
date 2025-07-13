# src/ml/automl.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

class AutoMLEngine:
    """
    Automated Machine Learning engine that automatically tries multiple algorithms,
    preprocessing steps, and hyperparameters to find the best model
    """
    
    def __init__(self):
        self.results = []
        self.best_model = None
        self.best_score = None
        self.experiment_log = []
        
    def render_automl_tab(self, df):
        """
        Main AutoML interface
        """
        st.header("🤖 **AutoML - Automated Machine Learning**")
        st.markdown("*Let AI find the best model for your data automatically*")
        
        # Data validation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for AutoML")
            return
        
        # Configuration
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            target_col = st.selectbox(
                "**Target Variable:**",
                numeric_cols,
                key="automl_target"
            )
            
            # Auto-detect problem type
            if target_col:
                problem_type = self._detect_problem_type(df[target_col])
                st.info(f"**Detected Problem:** {problem_type.title()}")
        
        with config_col2:
            available_features = [col for col in numeric_cols if col != target_col]
            selected_features = st.multiselect(
                "**Features (leave empty for auto-selection):**",
                available_features,
                key="automl_features"
            )
            
            if not selected_features:
                selected_features = available_features[:10]  # Auto-select top 10
                st.info(f"🤖 **Auto-selected {len(selected_features)} features**")
        
        # AutoML Configuration
        st.subheader("🔧 AutoML Configuration")
        
        automl_col1, automl_col2, automl_col3 = st.columns(3)
        
        with automl_col1:
            search_strategy = st.selectbox(
                "**Search Strategy:**",
                ["Quick Search", "Thorough Search", "Custom"],
                help="Quick: Fast but basic | Thorough: Comprehensive but slower",
                key="search_strategy"
            )
        
        with automl_col2:
            max_time_minutes = st.slider(
                "**Max Time (minutes):**",
                min_value=1,
                max_value=60,
                value=5 if search_strategy == "Quick Search" else 15,
                key="max_time"
            )
        
        with automl_col3:
            include_preprocessing = st.checkbox(
                "**Auto Preprocessing**",
                value=True,
                help="Automatically try different scaling methods",
                key="auto_preprocessing"
            )
        
        # Advanced options
        with st.expander("⚙️ **Advanced AutoML Settings**"):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                cv_folds = st.slider("Cross-Validation Folds:", 3, 10, 5, key="automl_cv")
                test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05, key="automl_test_size")
            
            with adv_col2:
                include_feature_selection = st.checkbox("Auto Feature Selection", value=True, key="feature_selection")
                random_state = st.number_input("Random State:", value=42, key="automl_random_state")
        
        # Run AutoML
        if st.button("🚀 **Start AutoML Experiment**", type="primary", key="start_automl"):
            with st.spinner('🤖 AutoML is running... This may take a few minutes...'):
                try:
                    # Prepare data
                    X = df[selected_features].copy()
                    y = df[target_col].copy()
                    
                    # Handle missing values
                    if X.isnull().any().any() or y.isnull().any():
                        st.warning("⚠️ Handling missing values...")
                        mask = ~(X.isnull().any(axis=1) | y.isnull())
                        X = X[mask]
                        y = y[mask]
                    
                    # Run AutoML experiment
                    experiment_results = self._run_automl_experiment(
                        X, y, problem_type, search_strategy, max_time_minutes,
                        include_preprocessing, include_feature_selection,
                        cv_folds, test_size, random_state
                    )
                    
                    if experiment_results:
                        self._display_automl_results(experiment_results, target_col, selected_features)
                        
                except Exception as e:
                    st.error(f"❌ AutoML failed: {str(e)}")
                    st.info("💡 Try reducing the number of features or check data quality")
    
    def _detect_problem_type(self, target_series, threshold=10):
        """Auto-detect if problem is classification or regression"""
        clean_target = target_series.dropna()
        unique_values = clean_target.nunique()
        
        if clean_target.dtype == 'object' or unique_values <= threshold:
            return 'classification'
        else:
            return 'regression'
    
    def _get_algorithms_for_problem(self, problem_type, search_strategy):
        """Get algorithms to try based on problem type and search strategy"""
        if problem_type == 'classification':
            if search_strategy == "Quick Search":
                return {
                    'Random Forest': RandomForestClassifier(),
                    'Logistic Regression': LogisticRegression(max_iter=1000),
                    'Gradient Boosting': GradientBoostingClassifier()
                }
            else:  # Thorough Search
                return {
                    'Random Forest': RandomForestClassifier(),
                    'Logistic Regression': LogisticRegression(max_iter=1000),
                    'Gradient Boosting': GradientBoostingClassifier(),
                    'SVM': SVC(probability=True),
                    'Decision Tree': DecisionTreeClassifier(),
                    'K-Nearest Neighbors': KNeighborsClassifier(),
                    'Naive Bayes': GaussianNB()
                }
        
        else:  # regression
            if search_strategy == "Quick Search":
                return {
                    'Random Forest': RandomForestRegressor(),
                    'Linear Regression': LinearRegression(),
                    'Gradient Boosting': GradientBoostingRegressor()
                }
            else:  # Thorough Search
                return {
                    'Random Forest': RandomForestRegressor(),
                    'Linear Regression': LinearRegression(),
                    'Ridge Regression': Ridge(),
                    'Lasso Regression': Lasso(),
                    'Gradient Boosting': GradientBoostingRegressor(),
                    'SVR': SVR(),
                    'Decision Tree': DecisionTreeRegressor(),
                    'K-Nearest Neighbors': KNeighborsRegressor()
                }
    
    def _get_hyperparameter_grids(self, algorithm_name, model, search_strategy):
        """Get hyperparameter grids for different algorithms"""
        if search_strategy == "Quick Search":
            # Simplified grids for quick search
            quick_grids = {
                'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
                'Logistic Regression': {'C': [0.1, 1.0, 10.0]},
                'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]},
                'Linear Regression': {},
                'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
                'Lasso Regression': {'alpha': [0.1, 1.0, 10.0]},
                'SVM': {'C': [0.1, 1.0], 'kernel': ['rbf']},
                'SVR': {'C': [0.1, 1.0], 'kernel': ['rbf']},
                'Decision Tree': {'max_depth': [None, 5, 10]},
                'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
                'Naive Bayes': {}
            }
            return quick_grids.get(algorithm_name, {})
        
        else:  # Thorough search
            thorough_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'Logistic Regression': {
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'solver': ['liblinear', 'lbfgs']
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'Ridge Regression': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                },
                'Lasso Regression': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                },
                'SVM': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'SVR': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'Decision Tree': {
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'K-Nearest Neighbors': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },
                'Naive Bayes': {}
            }
            return thorough_grids.get(algorithm_name, {})
    
    def _get_preprocessing_pipelines(self, include_preprocessing):
        """Get different preprocessing pipelines to try"""
        if not include_preprocessing:
            return [('no_scaling', None)]
        
        return [
            ('no_scaling', None),
            ('standard_scaler', StandardScaler()),
            ('minmax_scaler', MinMaxScaler()),
            ('robust_scaler', RobustScaler())
        ]
    
    def _run_automl_experiment(self, X, y, problem_type, search_strategy, max_time_minutes,
                             include_preprocessing, include_feature_selection, cv_folds, test_size, random_state):
        """
        Run the complete AutoML experiment
        """
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Get algorithms to try
        algorithms = self._get_algorithms_for_problem(problem_type, search_strategy)
        preprocessing_options = self._get_preprocessing_pipelines(include_preprocessing)
        
        results = []
        experiment_log = []
        
        total_combinations = len(algorithms) * len(preprocessing_options)
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_combination = 0
        
        for algo_name, algorithm in algorithms.items():
            for prep_name, preprocessor in preprocessing_options:
                current_combination += 1
                
                # Check time limit
                if time.time() - start_time > max_time_seconds:
                    st.warning(f"⏰ Time limit reached. Stopping after {current_combination-1} combinations.")
                    break
                
                try:
                    status_text.text(f"🔍 Testing: {algo_name} with {prep_name}")
                    
                    # Create pipeline
                    if preprocessor is not None:
                        pipeline_steps = [('preprocessor', preprocessor), ('classifier', algorithm)]
                    else:
                        pipeline_steps = [('classifier', algorithm)]
                    
                    pipeline = Pipeline(pipeline_steps)
                    
                    # Get hyperparameters (adjust for pipeline)
                    param_grid = self._get_hyperparameter_grids(algo_name, algorithm, search_strategy)
                    if param_grid:
                        # Prefix parameters with 'classifier__' for pipeline
                        param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
                    
                    # Hyperparameter search
                    if param_grid:
                        if search_strategy == "Quick Search":
                            search = GridSearchCV(
                                pipeline, param_grid, cv=cv_folds, 
                                scoring='accuracy' if problem_type == 'classification' else 'r2',
                                n_jobs=-1
                            )
                        else:
                            search = RandomizedSearchCV(
                                pipeline, param_grid, cv=cv_folds, n_iter=10,
                                scoring='accuracy' if problem_type == 'classification' else 'r2',
                                n_jobs=-1, random_state=random_state
                            )
                        
                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        best_params = search.best_params_
                        cv_score = search.best_score_
                    else:
                        # No hyperparameters to tune
                        pipeline.fit(X_train, y_train)
                        cv_scores = cross_val_score(
                            pipeline, X_train, y_train, cv=cv_folds,
                            scoring='accuracy' if problem_type == 'classification' else 'r2'
                        )
                        best_model = pipeline
                        best_params = {}
                        cv_score = cv_scores.mean()
                    
                    # Test set evaluation
                    y_pred = best_model.predict(X_test)
                    
                    if problem_type == 'classification':
                        test_score = accuracy_score(y_test, y_pred)
                        metric_name = 'Accuracy'
                    else:
                        test_score = r2_score(y_test, y_pred)
                        metric_name = 'R² Score'
                    
                    # Store results
                    result = {
                        'algorithm': algo_name,
                        'preprocessing': prep_name,
                        'cv_score': cv_score,
                        'test_score': test_score,
                        'best_params': best_params,
                        'model': best_model,
                        'metric_name': metric_name,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                    
                    results.append(result)
                    experiment_log.append(f"✅ {algo_name} + {prep_name}: {metric_name} = {test_score:.4f}")
                    
                except Exception as e:
                    experiment_log.append(f"❌ {algo_name} + {prep_name}: Failed ({str(e)})")
                
                # Update progress
                progress_bar.progress(current_combination / total_combinations)
                
                # Break outer loop if time limit reached
                if time.time() - start_time > max_time_seconds:
                    break
        
        progress_bar.progress(1.0)
        status_text.text("🎉 AutoML experiment completed!")
        
        if not results:
            st.error("❌ No models were successfully trained")
            return None
        
        # Find best model
        best_result = max(results, key=lambda x: x['test_score'])
        
        experiment_summary = {
            'results': results,
            'best_result': best_result,
            'experiment_log': experiment_log,
            'total_time': time.time() - start_time,
            'models_tested': len(results),
            'problem_type': problem_type
        }
        
        return experiment_summary
    
    def _display_automl_results(self, experiment_summary, target_col, features):
        """
        Display comprehensive AutoML results
        """
        results = experiment_summary['results']
        best_result = experiment_summary['best_result']
        experiment_log = experiment_summary['experiment_log']
        total_time = experiment_summary['total_time']
        
        st.success(f"🎉 AutoML Completed! Best model: **{best_result['algorithm']}** with **{best_result['preprocessing']}**")
        
        # Summary metrics
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Models Tested", len(results))
        with summary_col2:
            st.metric("Total Time", f"{total_time/60:.1f} min")
        with summary_col3:
            st.metric(f"Best {best_result['metric_name']}", f"{best_result['test_score']:.4f}")
        with summary_col4:
            improvement = (best_result['test_score'] - min(r['test_score'] for r in results)) / min(r['test_score'] for r in results) * 100
            st.metric("Improvement", f"+{improvement:.1f}%")
        
        # Results comparison table
        st.subheader("📊 Model Comparison")
        
        comparison_data = []
        for result in sorted(results, key=lambda x: x['test_score'], reverse=True):
            comparison_data.append({
                'Rank': len(comparison_data) + 1,
                'Algorithm': result['algorithm'],
                'Preprocessing': result['preprocessing'],
                'CV Score': f"{result['cv_score']:.4f}",
                'Test Score': f"{result['test_score']:.4f}",
                'Parameters': str(result['best_params'])[:50] + "..." if len(str(result['best_params'])) > 50 else str(result['best_params'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance visualization
        st.subheader("📈 Performance Comparison")
        
        viz_data = pd.DataFrame({
            'Algorithm': [f"{r['algorithm']}+{r['preprocessing']}" for r in results],
            'CV Score': [r['cv_score'] for r in results],
            'Test Score': [r['test_score'] for r in results]
        })
        
        fig = px.scatter(
            viz_data,
            x='CV Score',
            y='Test Score',
            hover_data=['Algorithm'],
            title='Cross-Validation vs Test Performance',
            labels={'CV Score': 'Cross-Validation Score', 'Test Score': 'Test Set Score'}
        )
        
        # Add diagonal line (perfect correlation)
        min_score = min(viz_data[['CV Score', 'Test Score']].min())
        max_score = max(viz_data[['CV Score', 'Test Score']].max())
        fig.add_trace(go.Scatter(
            x=[min_score, max_score],
            y=[min_score, max_score],
            mode='lines',
            name='Perfect Correlation',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model evaluation
        st.subheader("🏆 Best Model Evaluation")
        
        best_model = best_result['model']
        
        # Model details
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.markdown(f"""
            **🤖 Best Model Details:**
            - **Algorithm:** {best_result['algorithm']}
            - **Preprocessing:** {best_result['preprocessing']}
            - **{best_result['metric_name']}:** {best_result['test_score']:.4f}
            - **CV Score:** {best_result['cv_score']:.4f}
            """)
        
        with details_col2:
            st.markdown(f"""
            **🎯 Best Parameters:**
            """)
            for param, value in best_result['best_params'].items():
                st.write(f"• **{param}:** {value}")
        
        # Store best model in session state
        model_id = f"automl_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        
        # Prepare test data for storage
        X_test_features = pd.DataFrame(best_result['y_test']).index  # Get the indices
        # We need to reconstruct X_test - this is a limitation we should note
        
        model_info = {
            'model': best_model,
            'algorithm': f"AutoML - {best_result['algorithm']}",
            'target': target_col,
            'features': features,
            'problem_type': experiment_summary['problem_type'],
            'test_data': (None, best_result['y_test']),  # X_test not available in this context
            'model_id': model_id,
            'automl_summary': experiment_summary
        }
        
        st.session_state.trained_models[model_id] = model_info
        
        st.success(f"✅ Best model saved as: **{model_id}**")
        st.info("💡 Use the 'Make Predictions' or 'Model Management' tabs to work with this model")
        
        # Experiment log
        with st.expander("📋 Experiment Log"):
            for log_entry in experiment_log:
                st.write(log_entry)
        
        # Download results
        results_csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 **Download AutoML Results**",
            data=results_csv,
            file_name=f"automl_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )