# src/ml/ml_trainer.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from .ml_evaluator import MLEvaluator
from .ml_utils import MLUtils

class MLTrainer:
    """
    Main ML training interface - handles UI and coordinates ML workflow
    """
    
    def __init__(self):
        self.evaluator = MLEvaluator()
        self.utils = MLUtils()
        
        # Initialize session state for ML
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'ml_results' not in st.session_state:
            st.session_state.ml_results = {}
    
    def get_current_data(self):
        """Get current working data from session state"""
        if 'working_df' in st.session_state and st.session_state.working_df is not None:
            return st.session_state.working_df.copy()
        elif 'base_df' in st.session_state and st.session_state.base_df is not None:
            return st.session_state.base_df.copy()
        else:
            st.error("No data available for ML training")
            return pd.DataFrame()
    
    def render_ml_tab(self):
        """
        Render the complete ML training tab
        """
        st.header("🤖 **Machine Learning**")
        st.markdown("*Train machine learning models on your processed data*")
        
        # Get current data
        current_data = self.get_current_data()
        
        if current_data.empty:
            st.warning("⚠️ No data available for ML training")
            return
        
        # Data overview
        data_col1, data_col2, data_col3, data_col4 = st.columns(4)
        
        with data_col1:
            st.metric("Total Samples", len(current_data))
        with data_col2:
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))
        with data_col3:
            categorical_cols = current_data.select_dtypes(include=['object', 'category']).columns
            st.metric("Categorical Features", len(categorical_cols))
        with data_col4:
            missing_pct = (current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns))) * 100
            st.metric("Missing Data %", f"{missing_pct:.1f}%")
        
        st.markdown("---")
        
        # ML workflow selection
        ml_task = st.selectbox(
            "**Select ML Task:**",
            [
                "🎯 Supervised Learning (Classification/Regression)",
                "🔍 Unsupervised Learning (Clustering)",
                "📊 Model Comparison Dashboard",
                "🔮 Make Predictions"
            ],
            key="ml_task_selection"
        )
        
        st.markdown("---")
        
        try:
            if ml_task == "🎯 Supervised Learning (Classification/Regression)":
                self._supervised_learning_workflow(current_data)
            elif ml_task == "🔍 Unsupervised Learning (Clustering)":
                self._unsupervised_learning_workflow(current_data)
            elif ml_task == "📊 Model Comparison Dashboard":
                self._model_comparison_dashboard()
            elif ml_task == "🔮 Make Predictions":
                self._prediction_interface()
                
        except Exception as e:
            st.error(f"❌ ML Error: {str(e)}")
            st.info("💡 **Tip:** Ensure your data has numeric features and no excessive missing values")
    
    def _supervised_learning_workflow(self, df):
        """Handle supervised learning workflow"""
        st.subheader("🎯 Supervised Learning")
        
        # Target variable selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns (1 target + 1 feature)")
            return
        
        setup_col1, setup_col2 = st.columns(2)
        
        with setup_col1:
            target_col = st.selectbox(
                "**Target Variable (what to predict):**",
                numeric_cols,
                key="target_variable"
            )
            
            # Auto-detect problem type
            if target_col:
                problem_type = self.utils.detect_problem_type(df[target_col])
                st.info(f"**Detected Problem Type:** {problem_type}")
        
        with setup_col2:
            # Feature selection
            available_features = [col for col in numeric_cols if col != target_col]
            selected_features = st.multiselect(
                "**Select Features:**",
                available_features,
                default=available_features[:5],  # Default to first 5
                key="selected_features"
            )
        
        if not selected_features:
            st.warning("Please select at least one feature")
            return
        
        # Train/Test split
        test_size = st.slider("**Test Set Size:**", 0.1, 0.5, 0.2, 0.05, key="test_size")
        
        # Algorithm selection
        algorithms = self.utils.get_available_algorithms(problem_type)
        selected_algorithm = st.selectbox(
            "**Select Algorithm:**",
            list(algorithms.keys()),
            key="selected_algorithm"
        )
        
        # Training parameters
        with st.expander("⚙️ **Training Parameters**"):
            use_cv = st.checkbox("Use Cross-Validation", value=True, key="use_cv")
            if use_cv:
                cv_folds = st.slider("CV Folds:", 3, 10, 5, key="cv_folds")
            
            random_state = st.number_input("Random State:", value=42, key="random_state")
        
        # Training button
        if st.button("🚀 **Train Model**", type="primary", key="train_supervised"):
            with st.spinner('Training model...'):
                try:
                    # Prepare data
                    X = df[selected_features].copy()
                    y = df[target_col].copy()
                    
                    # Handle missing values
                    if X.isnull().any().any() or y.isnull().any():
                        st.warning("⚠️ Handling missing values by dropping rows...")
                        mask = ~(X.isnull().any(axis=1) | y.isnull())
                        X = X[mask]
                        y = y[mask]
                    
                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Train model
                    model_class = algorithms[selected_algorithm]
                    model = model_class(random_state=random_state)
                    model.fit(X_train, y_train)
                    
                    # Store model and results
                    model_id = f"{selected_algorithm}_{target_col}_{len(st.session_state.trained_models)}"
                    
                    model_info = {
                        'model': model,
                        'algorithm': selected_algorithm,
                        'target': target_col,
                        'features': selected_features,
                        'problem_type': problem_type,
                        'test_data': (X_test, y_test),
                        'training_data': (X_train, y_train),
                        'model_id': model_id
                    }
                    
                    st.session_state.trained_models[model_id] = model_info
                    
                    st.success(f"✅ Model trained successfully! Model ID: {model_id}")
                    
                    # Show evaluation
                    self.evaluator.evaluate_model(model_info)
                    
                    # Cross-validation if enabled
                    if use_cv:
                        st.subheader("📊 Cross-Validation Results")
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                        
                        cv_col1, cv_col2, cv_col3 = st.columns(3)
                        with cv_col1:
                            st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
                        with cv_col2:
                            st.metric("Std CV Score", f"{cv_scores.std():.4f}")
                        with cv_col3:
                            st.metric("CV Range", f"{cv_scores.min():.3f} - {cv_scores.max():.3f}")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.info("💡 **Common issues:** Missing values, non-numeric data, or insufficient samples")
    
    def _unsupervised_learning_workflow(self, df):
        """Handle unsupervised learning workflow"""
        st.subheader("🔍 Unsupervised Learning - Clustering")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for clustering")
            return
        
        # Feature selection for clustering
        selected_features = st.multiselect(
            "**Select Features for Clustering:**",
            numeric_cols,
            default=numeric_cols[:3],
            key="clustering_features"
        )
        
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features")
            return
        
        # Clustering parameters
        cluster_col1, cluster_col2 = st.columns(2)
        
        with cluster_col1:
            n_clusters = st.slider("**Number of Clusters:**", 2, 10, 3, key="n_clusters")
        
        with cluster_col2:
            random_state = st.number_input("Random State:", value=42, key="clustering_random_state")
        
        if st.button("🔍 **Perform Clustering**", type="primary", key="perform_clustering"):
            with st.spinner('Performing clustering...'):
                try:
                    # Prepare data
                    X = df[selected_features].copy()
                    
                    # Handle missing values
                    if X.isnull().any().any():
                        st.warning("⚠️ Handling missing values...")
                        X = X.dropna()
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    clusters = kmeans.fit_predict(X)
                    
                    # Add clusters to dataframe
                    X_clustered = X.copy()
                    X_clustered['Cluster'] = clusters
                    
                    # Store results
                    cluster_id = f"KMeans_{n_clusters}clusters_{len(st.session_state.trained_models)}"
                    cluster_info = {
                        'model': kmeans,
                        'algorithm': 'K-Means Clustering',
                        'features': selected_features,
                        'n_clusters': n_clusters,
                        'data_with_clusters': X_clustered,
                        'model_id': cluster_id,
                        'problem_type': 'clustering'
                    }
                    
                    st.session_state.trained_models[cluster_id] = cluster_info
                    
                    st.success(f"✅ Clustering completed! Model ID: {cluster_id}")
                    
                    # Visualize clusters
                    self.evaluator.visualize_clusters(cluster_info)
                    
                except Exception as e:
                    st.error(f"Clustering failed: {str(e)}")
    
    def _model_comparison_dashboard(self):
        """Display model comparison dashboard"""
        st.subheader("📊 Model Comparison Dashboard")
        
        if not st.session_state.trained_models:
            st.info("🤖 No trained models yet. Train some models first!")
            return
        
        # Show all trained models
        models_df = []
        for model_id, model_info in st.session_state.trained_models.items():
            if model_info['problem_type'] != 'clustering':
                models_df.append({
                    'Model ID': model_id,
                    'Algorithm': model_info['algorithm'],
                    'Target': model_info.get('target', 'N/A'),
                    'Features': len(model_info['features']),
                    'Problem Type': model_info['problem_type']
                })
        
        if models_df:
            st.dataframe(pd.DataFrame(models_df), use_container_width=True)
            
            # Model comparison
            selected_models = st.multiselect(
                "**Select models to compare:**",
                [model['Model ID'] for model in models_df],
                key="models_to_compare"
            )
            
            if len(selected_models) > 1:
                if st.button("📊 **Compare Selected Models**", key="compare_models"):
                    self.evaluator.compare_models(selected_models)
        else:
            st.info("🤖 No supervised learning models to compare. Clustering models are shown separately.")
    
    def _prediction_interface(self):
        """Handle making predictions with trained models"""
        st.subheader("🔮 Make Predictions")
        
        supervised_models = {k: v for k, v in st.session_state.trained_models.items() 
                           if v['problem_type'] != 'clustering'}
        
        if not supervised_models:
            st.info("🤖 No trained models available for predictions. Train a model first!")
            return
        
        # Model selection
        selected_model_id = st.selectbox(
            "**Select Model:**",
            list(supervised_models.keys()),
            key="prediction_model"
        )
        
        if selected_model_id:
            model_info = supervised_models[selected_model_id]
            self.evaluator.generate_prediction_interface(model_info)