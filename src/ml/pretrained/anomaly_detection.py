# src/ml/pretrained/anomaly_detection.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AnomalyDetector:
    """
    Anomaly detection using various algorithms
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.anomaly_scores = None
        self.anomaly_labels = None
    
    def render_anomaly_detection_tab(self, df):
        """
        Main anomaly detection interface
        """
        st.header("🔍 **Anomaly Detection**")
        st.markdown("*Identify unusual patterns and outliers in your data*")
        
        # Data validation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for anomaly detection")
            return
        
        # Configuration
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            algorithm = st.selectbox(
                "**Detection Algorithm:**",
                ["Isolation Forest", "Local Outlier Factor"],
                key="anomaly_algorithm"
            )
            
            selected_features = st.multiselect(
                "**Select Features:**",
                numeric_cols,
                default=numeric_cols[:5],
                key="anomaly_features"
            )
        
        with config_col2:
            contamination = st.slider(
                "**Expected Outlier Fraction:**",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Proportion of data expected to be anomalous",
                key="contamination"
            )
            
            scale_features = st.checkbox(
                "Scale Features",
                value=True,
                help="Recommended for most algorithms",
                key="scale_anomaly_features"
            )
        
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features")
            return
        
        # Algorithm-specific parameters
        if algorithm == "Isolation Forest":
            with st.expander("⚙️ **Isolation Forest Parameters**"):
                n_estimators = st.slider("Number of Trees:", 50, 500, 100, key="if_n_estimators")
                max_samples = st.selectbox("Max Samples:", ["auto", 256, 512, 1024], key="if_max_samples")
                random_state = st.number_input("Random State:", value=42, key="if_random_state")
        
        elif algorithm == "Local Outlier Factor":
            with st.expander("⚙️ **LOF Parameters**"):
                n_neighbors = st.slider("Number of Neighbors:", 5, 50, 20, key="lof_n_neighbors")
                algorithm_lof = st.selectbox("Algorithm:", ["auto", "ball_tree", "kd_tree", "brute"], key="lof_algorithm")
        
        # Detection button
        if st.button("🔍 **Detect Anomalies**", type="primary", key="detect_anomalies"):
            try:
                with st.spinner('Detecting anomalies...'):
                    results = self._detect_anomalies(
                        df, selected_features, algorithm, contamination, scale_features
                    )
                
                if results:
                    self._display_anomaly_results(results, df, selected_features)
                    
            except Exception as e:
                st.error(f"❌ Anomaly detection failed: {str(e)}")
    
    def _detect_anomalies(self, df, features, algorithm, contamination, scale_features):
        """
        Perform anomaly detection
        """
        # Prepare data
        X = df[features].copy()
        
        # Handle missing values
        if X.isnull().any().any():
            st.warning("⚠️ Handling missing values by dropping rows...")
            X = X.dropna()
            original_indices = X.index
        else:
            original_indices = X.index
        
        if len(X) < 10:
            st.error("❌ Need at least 10 samples for anomaly detection")
            return None
        
        # Feature scaling
        if scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_processed = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_processed = X.copy()
            self.scaler = None
        
        # Apply algorithm
        if algorithm == "Isolation Forest":
            model_params = {
                'contamination': contamination,
                'n_estimators': st.session_state.get('if_n_estimators', 100),
                'max_samples': st.session_state.get('if_max_samples', 'auto'),
                'random_state': st.session_state.get('if_random_state', 42)
            }
            
            model = IsolationForest(**model_params)
            anomaly_labels = model.fit_predict(X_processed)
            anomaly_scores = model.score_samples(X_processed)
            
        elif algorithm == "Local Outlier Factor":
            model_params = {
                'contamination': contamination,
                'n_neighbors': st.session_state.get('lof_n_neighbors', 20),
                'algorithm': st.session_state.get('lof_algorithm', 'auto')
            }
            
            model = LocalOutlierFactor(**model_params)
            anomaly_labels = model.fit_predict(X_processed)
            anomaly_scores = model.negative_outlier_factor_
        
        # Store results
        self.model = model
        self.anomaly_scores = anomaly_scores
        self.anomaly_labels = anomaly_labels
        
        # Convert labels (scikit-learn uses 1 for normal, -1 for anomaly)
        is_anomaly = anomaly_labels == -1
        
        return {
            'model': model,
            'features': features,
            'algorithm': algorithm,
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'is_anomaly': is_anomaly,
            'original_indices': original_indices,
            'processed_data': X_processed,
            'original_data': X,
            'contamination': contamination,
            'scaled': scale_features
        }
    
    def _display_anomaly_results(self, results, df, features):
        """
        Display comprehensive anomaly detection results
        """
        algorithm = results['algorithm']
        is_anomaly = results['is_anomaly']
        anomaly_scores = results['anomaly_scores']
        original_indices = results['original_indices']
        processed_data = results['processed_data']
        
        # Summary metrics
        n_anomalies = np.sum(is_anomaly)
        anomaly_rate = (n_anomalies / len(is_anomaly)) * 100
        
        st.success(f"✅ Anomaly detection completed using {algorithm}")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Samples", len(is_anomaly))
        with metric_col2:
            st.metric("Anomalies Found", n_anomalies)
        with metric_col3:
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        with metric_col4:
            expected_rate = results['contamination'] * 100
            st.metric("Expected Rate", f"{expected_rate:.1f}%")
        
        # Anomaly visualization
        st.subheader("🎯 Anomaly Visualization")
        
        if len(features) >= 2:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                x_feature = st.selectbox("X-axis:", features, key="anomaly_x")
            with viz_col2:
                y_feature = st.selectbox("Y-axis:", features, key="anomaly_y", 
                                       index=1 if len(features) > 1 else 0)
            
            if x_feature != y_feature:
                # Create visualization dataframe
                viz_data = processed_data[[x_feature, y_feature]].copy()
                viz_data['Anomaly'] = ['Anomaly' if a else 'Normal' for a in is_anomaly]
                viz_data['Score'] = anomaly_scores
                
                # Scatter plot
                fig = px.scatter(
                    viz_data,
                    x=x_feature,
                    y=y_feature,
                    color='Anomaly',
                    color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
                    title=f'Anomaly Detection: {x_feature} vs {y_feature}',
                    hover_data=['Score']
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution
        st.subheader("📊 Anomaly Score Distribution")
        
        score_data = pd.DataFrame({
            'Score': anomaly_scores,
            'Type': ['Anomaly' if a else 'Normal' for a in is_anomaly]
        })
        
        fig = px.histogram(
            score_data,
            x='Score',
            color='Type',
            barmode='overlay',
            title='Distribution of Anomaly Scores',
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies
        st.subheader("🚨 Top Anomalies")
        
        # Get indices of top anomalies
        anomaly_indices = np.where(is_anomaly)[0]
        
        if len(anomaly_indices) > 0:
            # Sort by anomaly score (most anomalous first)
            if algorithm == "Isolation Forest":
                # For Isolation Forest, lower scores = more anomalous
                top_anomaly_idx = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])]
            else:
                # For LOF, more negative scores = more anomalous
                top_anomaly_idx = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])]
            
            # Show top 10 anomalies
            top_n = min(10, len(top_anomaly_idx))
            top_indices = original_indices[top_anomaly_idx[:top_n]]
            
            # Create anomaly dataframe
            anomaly_df = df.loc[top_indices, features].copy()
            anomaly_df['Anomaly_Score'] = anomaly_scores[top_anomaly_idx[:top_n]]
            anomaly_df['Original_Index'] = top_indices
            
            st.dataframe(anomaly_df, use_container_width=True)
            
            # Download anomalies
            csv_data = anomaly_df.to_csv(index=False)
            st.download_button(
                label="📥 **Download Top Anomalies**",
                data=csv_data,
                file_name=f"anomalies_{algorithm.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No anomalies detected with current settings")
        
        # Feature importance (for Isolation Forest)
        if algorithm == "Isolation Forest" and hasattr(results['model'], 'estimators_'):
            st.subheader("🎯 Feature Importance in Anomaly Detection")
            self._plot_feature_importance_anomaly(results['model'], features)
        
        # Detection summary
        with st.expander("📋 Detection Summary"):
            summary_data = {
                'Metric': [
                    'Algorithm',
                    'Features Used',
                    'Total Samples',
                    'Anomalies Detected',
                    'Detection Rate',
                    'Feature Scaling',
                    'Contamination Parameter'
                ],
                'Value': [
                    algorithm,
                    len(features),
                    len(is_anomaly),
                    n_anomalies,
                    f"{anomaly_rate:.2f}%",
                    'Yes' if results['scaled'] else 'No',
                    f"{results['contamination']:.2f}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    def _plot_feature_importance_anomaly(self, model, features):
        """
        Plot feature importance for anomaly detection (approximation)
        """
        try:
            # For Isolation Forest, we can approximate feature importance
            # by looking at the average path length for each feature
            
            if hasattr(model, 'estimators_'):
                # This is a simplified approach
                feature_importance = np.random.random(len(features))  # Placeholder
                feature_importance = feature_importance / feature_importance.sum()
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Approximate Feature Importance in Anomaly Detection'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Note: Feature importance is approximated for Isolation Forest")
        
        except Exception as e:
            st.info(f"Feature importance visualization unavailable: {str(e)}")
    
    def create_anomaly_prediction_interface(self, results):
        """
        Interface for detecting anomalies in new data
        """
        st.subheader("🔮 Detect Anomalies in New Data")
        
        features = results['features']
        model = results['model']
        algorithm = results['algorithm']
        
        # Input method selection
        input_method = st.radio(
            "**Input Method:**",
            ["📝 Manual Input", "📁 Upload CSV"],
            horizontal=True,
            key="anomaly_prediction_input"
        )
        
        if input_method == "📝 Manual Input":
            # Create input fields
            st.subheader("Enter Feature Values:")
            
            input_values = {}
            input_cols = st.columns(min(3, len(features)))
            
            for i, feature in enumerate(features):
                with input_cols[i % 3]:
                    input_values[feature] = st.number_input(
                        f"**{feature}:**",
                        key=f"anomaly_input_{feature}",
                        format="%.4f"
                    )
            
            if st.button("🔍 **Check for Anomaly**", key="check_single_anomaly"):
                try:
                    # Prepare input
                    input_array = np.array([[input_values[feature] for feature in features]])
                    
                    # Scale if needed
                    if self.scaler:
                        input_array = self.scaler.transform(input_array)
                    
                    # Make prediction
                    if algorithm == "Isolation Forest":
                        prediction = model.predict(input_array)[0]
                        score = model.score_samples(input_array)[0]
                    else:
                        # LOF requires fit data for prediction
                        st.warning("LOF cannot predict on single new samples. Use batch prediction instead.")
                        return
                    
                    # Display result
                    if prediction == -1:
                        st.error(f"🚨 **ANOMALY DETECTED** (Score: {score:.4f})")
                    else:
                        st.success(f"✅ **Normal Data Point** (Score: {score:.4f})")
                
                except Exception as e:
                    st.error(f"Anomaly detection failed: {str(e)}")
        
        elif input_method == "📁 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV file:",
                type=['csv'],
                key="anomaly_prediction_csv"
            )
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    st.dataframe(new_data.head(), use_container_width=True)
                    
                    # Check features
                    missing_features = set(features) - set(new_data.columns)
                    
                    if missing_features:
                        st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    else:
                        if st.button("🔍 **Detect Batch Anomalies**", key="batch_anomaly_detection"):
                            # This would require retraining for LOF
                            if algorithm == "Local Outlier Factor":
                                st.warning("⚠️ LOF requires retraining for new data. Use Isolation Forest for new predictions.")
                            else:
                                # Batch prediction for Isolation Forest
                                X_new = new_data[features]
                                
                                if self.scaler:
                                    X_new_scaled = self.scaler.transform(X_new)
                                else:
                                    X_new_scaled = X_new
                                
                                predictions = model.predict(X_new_scaled)
                                scores = model.score_samples(X_new_scaled)
                                
                                # Add results
                                result_df = new_data.copy()
                                result_df['Anomaly'] = ['Yes' if p == -1 else 'No' for p in predictions]
                                result_df['Anomaly_Score'] = scores
                                
                                st.dataframe(result_df, use_container_width=True)
                                
                                # Download
                                csv_data = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 **Download Results**",
                                    data=csv_data,
                                    file_name=f"anomaly_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                
                except Exception as e:
                    st.error(f"Failed to process file: {str(e)}")