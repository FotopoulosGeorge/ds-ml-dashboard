# src/ml/ml_trainer.py
import streamlit as st
import os
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
from .pretrained.time_series import TimeSeriesForecaster
from .pretrained.anomaly_detection import AnomalyDetector
from .automl import AutoMLEngine
from .pretrained.pattern_mining import PatternMiner
from src.demo.demo_datasets import DemoDatasets

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
        workflow_category = st.radio(
            "**What would you like to do?**",
            ["🎯 Train New Models", "📊 Analyze Existing Models", "🔮 Use Trained Models"],
            horizontal=True,
            key="ml_workflow_category"
        )

        if workflow_category == "🎯 Train New Models":
            ml_task = st.selectbox(
                "**Select Training Method:**",
                [
                    "🎯 Supervised Learning (Classification/Regression)",
                    "🔍 Unsupervised Learning (Clustering)",
                    "🤖 AutoML (Automated)",
                    "📈 Time Series Forecasting", 
                    "🚨 Anomaly Detection",
                    "🔗 Pattern Mining"
                ],
                key="ml_training_task"
            )
        elif workflow_category == "📊 Analyze Existing Models":
            ml_task = st.selectbox(
                "**Select Analysis:**",
                [
                    "📊 Model Comparison Dashboard",
                    "💾 Model Management",
                    "📈 Performance Analysis"
                ],
                key="ml_analysis_task"
            )
        else:  # Use Trained Models
            ml_task = "🔮 Make Predictions"
        
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
            elif ml_task == "💾 Model Management":
                self._model_management_interface()
            elif ml_task == "📈 Time Series Forecasting":
                ts_forecaster = TimeSeriesForecaster()
                ts_forecaster.render_time_series_tab(current_data)
            elif ml_task == "🚨 Anomaly Detection":
                anomaly_detector = AnomalyDetector()
                anomaly_detector.render_anomaly_detection_tab(current_data)
            elif ml_task == "🤖 AutoML (Automated)":
                automl_engine = AutoMLEngine()
                automl_engine.render_automl_tab(current_data)
            elif ml_task == "🔗 Pattern Mining":
                pattern_miner = PatternMiner()
                pattern_miner.render_pattern_mining_tab(current_data)
                
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
        
        save_model = st.checkbox(
            "💾 Save model to disk (recommended)", 
            value=True, 
            help="Saves model permanently so it won't be lost when session ends",
            key="auto_save_model"
        )

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
                    try:
                        model = model_class(random_state=random_state)
                    except  TypeError:
                        model = model_class()
                        st.info(f"ℹ️ {selected_algorithm} doesn't use random_state parameter")
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
                    if save_model:
                        try:
                            filepath = self.utils.save_model(model, model_id, model_info)
                            st.success(f"✅ Model trained and saved! Model ID: {model_id}")
                            st.info(f"💾 Saved to: {filepath}")
                        except Exception as save_error:
                            st.success(f"✅ Model trained! Model ID: {model_id}")
                            st.warning(f"⚠️ Could not save to disk: {save_error}")
                    else:
                        st.success(f"✅ Model trained! Model ID: {model_id}")
                        st.warning("⚠️ Model not saved - will be lost when session ends")

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

                    st.markdown("---")
                    save_col1, save_col2 = st.columns(2)

                    with save_col1:
                        if st.button("💾 **Save Clustering Model**", key="save_clustering_model"):
                            try:
                                model_name = f"clustering_{cluster_id}"
                                filepath = self.utils.save_model(
                                    model=kmeans, 
                                    model_name=model_name,
                                    model_info=cluster_info
                                )
                                st.success(f"✅ Model saved to: {filepath}")
                                st.info("💡 Saved models can be loaded later from the Model Management section")
                            except Exception as e:
                                st.error(f"❌ Failed to save model: {str(e)}")

                    with save_col2:
                        # Show model info
                        st.info(f"""
                        **Model Details:**
                        • Algorithm: K-Means
                        • Clusters: {n_clusters}
                        • Features: {len(selected_features)}
                        • Samples: {len(X_clustered)}
                        """)
                    
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
        
        if not st.session_state.trained_models:
            st.info("🤖 No trained models available for predictions. Train a model first!")
            return
        
        # Separate models by type for better organization
        supervised_models = {k: v for k, v in st.session_state.trained_models.items() 
                        if v['problem_type'] in ['classification', 'regression']}
        clustering_models = {k: v for k, v in st.session_state.trained_models.items() 
                        if v['problem_type'] == 'clustering'}
        
        # Model type selection
        prediction_type = st.radio(
            "**Select Prediction Type:**",
            [
                "🎯 Supervised Predictions (Classification/Regression)", 
                "🔍 Cluster Assignment",
                "📈 Time Series Forecast",
                "🚨 Anomaly Detection"
            ],
            horizontal=True,
            key="prediction_type_select"
        )
        
        if prediction_type == "🎯 Supervised Predictions (Classification/Regression)":
            if not supervised_models:
                st.info("🤖 No supervised learning models available. Train a classification or regression model first!")
                return
            
            # Model selection for supervised learning
            selected_model_id = st.selectbox(
                "**Select Model:**",
                list(supervised_models.keys()),
                key="supervised_prediction_model"
            )
            
            if selected_model_id:
                model_info = supervised_models[selected_model_id]
                self.evaluator.generate_prediction_interface(model_info)
        
        elif prediction_type == "🔍 Cluster Assignment":
            if not clustering_models:
                st.info("🤖 No clustering models available. Train a clustering model first!")
                return
            
            # Model selection for clustering
            selected_model_id = st.selectbox(
                "**Select Clustering Model:**",
                list(clustering_models.keys()),
                key="clustering_prediction_model"
            )
            
            if selected_model_id:
                model_info = clustering_models[selected_model_id]
                self._generate_clustering_prediction_interface(model_info)

    def _generate_clustering_prediction_interface(self, model_info):
        """Generate interface for cluster assignment predictions"""
        st.subheader(f"🔍 Cluster Assignment - {model_info['algorithm']}")
        
        model = model_info['model']
        features = model_info['features']
        n_clusters = model_info['n_clusters']
        
        st.markdown(f"**Features:** {', '.join(features)}")
        st.markdown(f"**Number of Clusters:** {n_clusters}")
        
        # Input method selection
        input_method = st.radio(
            "**Input Method:**",
            ["📝 Manual Input", "📁 Upload CSV"],
            horizontal=True,
            key="clustering_prediction_input_method"
        )
        
        if input_method == "📝 Manual Input":
            # Create input fields for each feature
            st.subheader("Enter Feature Values:")
            
            input_values = {}
            input_cols = st.columns(min(3, len(features)))
            
            for i, feature in enumerate(features):
                with input_cols[i % 3]:
                    input_values[feature] = st.number_input(
                        f"**{feature}:**",
                        key=f"cluster_pred_input_{feature}",
                        format="%.4f"
                    )
            
            if st.button("🔍 **Predict Cluster**", key="make_cluster_prediction"):
                try:
                    # Create input array
                    input_array = np.array([[input_values[feature] for feature in features]])
                    
                    # Make prediction
                    predicted_cluster = model.predict(input_array)[0]
                    
                    # Get distance to all cluster centers
                    distances = model.transform(input_array)[0]
                    
                    # Display result
                    st.success(f"🎯 **Predicted Cluster:** {predicted_cluster}")
                    
                    # Show distances to all clusters
                    st.subheader("📊 Distance to Cluster Centers:")
                    distance_col1, distance_col2 = st.columns(2)
                    
                    with distance_col1:
                        for i, distance in enumerate(distances):
                            if i == predicted_cluster:
                                st.metric(f"🎯 Cluster {i} (Assigned)", f"{distance:.4f}")
                            else:
                                st.metric(f"Cluster {i}", f"{distance:.4f}")
                    
                    with distance_col2:
                        # Visualize distances
                        distance_df = pd.DataFrame({
                            'Cluster': [f"Cluster {i}" for i in range(len(distances))],
                            'Distance': distances,
                            'Assigned': [i == predicted_cluster for i in range(len(distances))]
                        })
                        
                        fig = px.bar(
                            distance_df, 
                            x='Cluster', 
                            y='Distance',
                            color='Assigned',
                            title='Distance to Each Cluster Center',
                            color_discrete_map={True: 'green', False: 'lightblue'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Cluster prediction failed: {str(e)}")
        
        elif input_method == "📁 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV file with feature values:",
                type=['csv'],
                key="clustering_prediction_csv_upload"
            )
            
            if uploaded_file is not None:
                try:
                    # Read uploaded file
                    new_data = pd.read_csv(uploaded_file)
                    
                    st.subheader("📋 Uploaded Data Preview:")
                    st.dataframe(new_data.head(), use_container_width=True)
                    
                    # Check if all required features are present
                    missing_features = set(features) - set(new_data.columns)
                    
                    if missing_features:
                        st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    else:
                        if st.button("🔍 **Assign Clusters**", key="make_batch_cluster_predictions"):
                            try:
                                # Make predictions
                                X_new = new_data[features]
                                predicted_clusters = model.predict(X_new)
                                distances = model.transform(X_new)
                                
                                # Add predictions to dataframe
                                result_df = new_data.copy()
                                result_df['Predicted_Cluster'] = predicted_clusters
                                
                                # Add distances to nearest cluster center
                                result_df['Distance_to_Center'] = [distances[i][predicted_clusters[i]] 
                                                                for i in range(len(predicted_clusters))]
                                
                                st.subheader("📊 Cluster Assignment Results:")
                                st.dataframe(result_df, use_container_width=True)
                                
                                # Show cluster distribution
                                cluster_counts = pd.DataFrame(predicted_clusters, columns=['Cluster']).value_counts().reset_index()
                                cluster_counts.columns = ['Cluster', 'Count']
                                
                                st.subheader("📈 Cluster Distribution:")
                                fig = px.pie(cluster_counts, values='Count', names='Cluster', 
                                        title='Distribution of Predicted Clusters')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download predictions
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 **Download Cluster Assignments**",
                                    data=csv,
                                    file_name=f"cluster_predictions_{model_info['model_id']}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            except Exception as e:
                                st.error(f"Batch cluster assignment failed: {str(e)}")
                
                except Exception as e:
                    st.error(f"Failed to read CSV: {str(e)}")

    def _model_management_interface(self):
        """Handle model loading and management"""
        st.subheader("💾 Model Management")
        is_deployed = DemoDatasets.is_deployed()
        # Show session models
        if st.session_state.trained_models:
            st.subheader("🧠 Models in Current Session")
            
            session_models = []
            for model_id, model_info in st.session_state.trained_models.items():
                session_models.append({
                    'Model ID': model_id,
                    'Algorithm': model_info['algorithm'],
                    'Type': model_info['problem_type'],
                    'Features': len(model_info['features']),
                    'Target': model_info.get('target', 'N/A')
                })
            
            session_df = pd.DataFrame(session_models)
            st.dataframe(session_df, use_container_width=True)
        else:
            st.info("No models in current session")
        
        st.markdown("---")
        # Handle saved models differently in demo vs local mode
        if is_deployed:
            # Demo mode: Explain limitations
            st.subheader("💿 Model Persistence in Demo Mode")
            st.info("""
            🌐 **Demo Mode Limitations:**
            - Models can't be permanently saved to cloud server
            - Models exist only during your current session  
            - Download model info as JSON for reference
            - For full model saving/loading, run app locally
            """)
            
            if st.session_state.trained_models:
                st.markdown("**💡 Want to save these models permanently?**")
                st.code("git clone [your-repo-url] && streamlit run dashboard.py")
        else:
            # Show saved models
            st.subheader("💿 Saved Models on Disk")
            
            try:
                saved_models = self.utils.list_saved_models()
                
                if saved_models:
                    saved_models_df = pd.DataFrame([{
                        'Filename': model['filename'],
                        'Created': model['created'].strftime('%Y-%m-%d %H:%M'),
                        'Size (MB)': f"{model['size_mb']:.2f}"
                    } for model in saved_models])
                    
                    st.dataframe(saved_models_df, use_container_width=True)
                    
                    # Model loading interface
                    st.subheader("📂 Load Saved Model")
                    
                    selected_file = st.selectbox(
                        "Select model to load:",
                        [model['filename'] for model in saved_models],
                        key="load_model_select"
                    )
                    
                    load_col1, load_col2 = st.columns(2)
                    
                    with load_col1:
                        if st.button("📂 **Load Model**", key="load_saved_model"):
                            try:
                                selected_model = next(m for m in saved_models if m['filename'] == selected_file)
                                model, model_info = self.utils.load_model(selected_model['filepath'])
                                
                                # Add to session state
                                loaded_model_id = f"loaded_{selected_file.replace('.joblib', '')}"
                                model_info['model'] = model
                                model_info['model_id'] = loaded_model_id
                                
                                st.session_state.trained_models[loaded_model_id] = model_info
                                
                                st.success(f"✅ Model loaded successfully as: {loaded_model_id}")
                                st.info("💡 Model is now available in current session for predictions and analysis")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to load model: {str(e)}")
                    
                    with load_col2:
                        if st.button("🗑️ **Delete Selected**", key="delete_saved_model"):
                            try:
                                selected_model = next(m for m in saved_models if m['filename'] == selected_file)
                                os.remove(selected_model['filepath'])
                                st.success(f"✅ Deleted {selected_file}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to delete: {str(e)}")
                
                else:
                    st.info("No saved models found")
                    st.caption("💡 Train and save models to see them here")
            
            except Exception as e:
                st.error(f"❌ Error accessing saved models: {str(e)}")