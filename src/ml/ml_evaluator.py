# src/ml/ml_evaluator.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import seaborn as sns
import matplotlib.pyplot as plt

class MLEvaluator:
    """
    Handles all ML evaluation, metrics, and visualization
    """
    
    def __init__(self):
        pass
    
    def evaluate_model(self, model_info):
        """
        Main evaluation method - routes to appropriate evaluation type
        """
        problem_type = model_info['problem_type']
        
        if problem_type == 'classification':
            self._evaluate_classification(model_info)
        elif problem_type == 'regression':
            self._evaluate_regression(model_info)
        elif problem_type == 'clustering':
            self._evaluate_clustering(model_info)
        elif problem_type == 'ensemble':
            self._evaluate_ensemble(model_info)
    
    def _evaluate_classification(self, model_info):
        """Comprehensive classification evaluation"""
        st.subheader("📊 Classification Results")
        
        model = model_info['model']
        X_test, y_test = model_info['test_data']
        algorithm = model_info['algorithm']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X_test)
            # Convert to probabilities for binary classification
            if len(np.unique(y_test)) == 2:
                from sklearn.utils.fixes import expit
                y_pred_proba = np.column_stack([1 - expit(decision_scores), expit(decision_scores)])
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("🎯 Accuracy", f"{accuracy:.4f}")
        with metric_col2:
            st.metric("🔍 Precision", f"{precision:.4f}")
        with metric_col3:
            st.metric("📡 Recall", f"{recall:.4f}")
        with metric_col4:
            st.metric("⚖️ F1-Score", f"{f1:.4f}")
        
        # Detailed visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Confusion Matrix
            st.subheader("🔥 Confusion Matrix")
            self._plot_confusion_matrix(y_test, y_pred)
        
        with viz_col2:
            # ROC Curve (for binary classification)
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                st.subheader("📈 ROC Curve")
                self._plot_roc_curve(y_test, y_pred_proba)
            else:
                st.subheader("📊 Prediction Distribution")
                self._plot_prediction_distribution(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            self._plot_feature_importance(model, model_info['features'])
        elif hasattr(model, 'coef_'):
            st.subheader("🎯 Feature Coefficients")
            self._plot_feature_coefficients(model, model_info['features'])
        
        # Classification report
        with st.expander("📋 Detailed Classification Report"):
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
    
    def _evaluate_regression(self, model_info):
        """Comprehensive regression evaluation"""
        st.subheader("📊 Regression Results")
        
        model = model_info['model']
        X_test, y_test = model_info['test_data']
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("📊 R² Score", f"{r2:.4f}")
        with metric_col2:
            st.metric("📏 RMSE", f"{rmse:.4f}")
        with metric_col3:
            st.metric("📐 MAE", f"{mae:.4f}")
        with metric_col4:
            st.metric("🎯 MSE", f"{mse:.4f}")
        
        # Visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Actual vs Predicted
            st.subheader("🎯 Actual vs Predicted")
            self._plot_actual_vs_predicted(y_test, y_pred)
        
        with viz_col2:
            # Residuals plot
            st.subheader("📊 Residuals Plot")
            self._plot_residuals(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            self._plot_feature_importance(model, model_info['features'])
        elif hasattr(model, 'coef_'):
            st.subheader("🎯 Feature Coefficients")
            self._plot_feature_coefficients(model, model_info['features'])

    def _evaluate_ensemble(self, model_info):
        """Evaluate ensemble models"""
        st.subheader("🔗 Ensemble Results")
        
        ensemble_type = model_info.get('ensemble_info', {}).get('method', 'Unknown')
        st.info(f"**Ensemble Type:** {ensemble_type}")
        
        # Check if it's a pipeline, chain, or standard ensemble
        if 'pipeline_info' in model_info:
            self._evaluate_pipeline_ensemble(model_info)
        elif 'stacking_info' in model_info:
            self._evaluate_stacking_ensemble(model_info)
        elif 'chain_result' in model_info:
            self._evaluate_chain_ensemble(model_info)
        else:
            # Standard ensemble evaluation
            base_problem_type = model_info.get('base_problem_type', 'classification')
            if base_problem_type == 'classification':
                self._evaluate_classification(model_info)
            else:
                self._evaluate_regression(model_info)
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix using plotly"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create labels
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            xaxis={'tickmode': 'array', 'tickvals': list(range(len(labels))), 'ticktext': labels},
            yaxis={'tickmode': 'array', 'tickvals': list(range(len(labels))), 'ticktext': labels}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve for binary classification"""
        # Get probabilities for positive class
        y_scores = y_pred_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_prediction_distribution(self, y_true, y_pred):
        """Plot distribution of predictions vs actual"""
        df_plot = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        fig = px.histogram(
            df_plot, 
            x=['Actual', 'Predicted'],
            barmode='group',
            title='Distribution: Actual vs Predicted'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_actual_vs_predicted(self, y_true, y_pred):
        """Plot actual vs predicted values for regression"""
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_residuals(self, y_true, y_pred):
        """Plot residuals for regression analysis"""
        residuals = y_true - y_pred
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='green', size=6, opacity=0.6)
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Residuals Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals (Actual - Predicted)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models"""
        importance = model.feature_importances_
        
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            df_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_feature_coefficients(self, model, feature_names):
        """Plot feature coefficients for linear models"""
        if hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                # Multi-class classification
                coef = model.coef_[0]  # Take first class for visualization
            else:
                coef = model.coef_
            
            df_coef = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef
            }).sort_values('Coefficient', ascending=True)
            
            fig = px.bar(
                df_coef,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Feature Coefficients',
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def visualize_clusters(self, cluster_info):
        """Visualize clustering results"""
        st.subheader("🔍 Clustering Results")
        
        data_with_clusters = cluster_info['data_with_clusters']
        features = cluster_info['features']
        n_clusters = cluster_info['n_clusters']
        
        # Cluster summary
        cluster_summary = data_with_clusters.groupby('Cluster').size().reset_index(name='Count')
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.subheader("📊 Cluster Sizes")
            fig = px.bar(cluster_summary, x='Cluster', y='Count', title='Samples per Cluster')
            st.plotly_chart(fig, use_container_width=True)
        
        with summary_col2:
            st.subheader("📋 Cluster Summary")
            st.dataframe(cluster_summary, use_container_width=True)
        
        # Visualization options
        if len(features) >= 2:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                x_feature = st.selectbox("X-axis feature:", features, key="cluster_x")
            with viz_col2:
                y_feature = st.selectbox("Y-axis feature:", features, key="cluster_y", 
                                       index=1 if len(features) > 1 else 0)
            
            # 2D scatter plot
            if x_feature != y_feature:
                fig = px.scatter(
                    data_with_clusters,
                    x=x_feature,
                    y=y_feature,
                    color='Cluster',
                    title=f'Clusters: {x_feature} vs {y_feature}',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 3D visualization if we have 3+ features
        if len(features) >= 3:
            with st.expander("🌐 3D Cluster Visualization"):
                z_feature = st.selectbox("Z-axis feature:", features, key="cluster_z", 
                                       index=2 if len(features) > 2 else 0)
                
                if len(set([x_feature, y_feature, z_feature])) == 3:
                    fig = px.scatter_3d(
                        data_with_clusters,
                        x=x_feature,
                        y=y_feature,
                        z=z_feature,
                        color='Cluster',
                        title=f'3D Clusters: {x_feature}, {y_feature}, {z_feature}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def compare_models(self, selected_model_ids):
        """Compare multiple trained models"""
        st.subheader("📊 Model Comparison")
        
        comparison_data = []
        
        for model_id in selected_model_ids:
            model_info = st.session_state.trained_models[model_id]
            model = model_info['model']
            X_test, y_test = model_info['test_data']
            
            y_pred = model.predict(X_test)
            
            if model_info['problem_type'] == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                comparison_data.append({
                    'Model ID': model_id,
                    'Algorithm': model_info['algorithm'],
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })
            
            elif model_info['problem_type'] == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                comparison_data.append({
                    'Model ID': model_id,
                    'Algorithm': model_info['algorithm'],
                    'R² Score': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MSE': mse
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Find best model
            if 'Accuracy' in comparison_df.columns:
                best_model_idx = comparison_df['Accuracy'].idxmax()
                best_metric = 'Accuracy'
            elif 'R² Score' in comparison_df.columns:
                best_model_idx = comparison_df['R² Score'].idxmax()
                best_metric = 'R² Score'
            
            best_model_id = comparison_df.loc[best_model_idx, 'Model ID']
            best_score = comparison_df.loc[best_model_idx, best_metric]
            
            st.success(f"🏆 **Best Model:** {best_model_id} with {best_metric}: {best_score:.4f}")
    
    def generate_prediction_interface(self, model_info):
        """Generate interface for making predictions"""
        st.subheader(f"🔮 Make Predictions - {model_info['algorithm']}")
        
        model = model_info['model']
        features = model_info['features']
        target = model_info['target']
        
        st.markdown(f"**Target Variable:** {target}")
        st.markdown(f"**Features:** {', '.join(features)}")
        
        # Input method selection
        input_method = st.radio(
            "**Input Method:**",
            ["📝 Manual Input", "📁 Upload CSV"],
            horizontal=True,
            key="prediction_input_method"
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
                        key=f"pred_input_{feature}",
                        format="%.4f",
                        step=1.0
                    )
            
            if st.button("🔮 **Make Prediction**", key="make_single_prediction"):
                try:
                    # Create input array
                    input_array = np.array([[input_values[feature] for feature in features]])
                    
                    # Make prediction
                    prediction = model.predict(input_array)[0]
                    
                    # Display result
                    st.success(f"🎯 **Predicted {target}:** {prediction:.4f}")
                    
                    # Show prediction probability if available (classification)
                    if hasattr(model, 'predict_proba') and model_info['problem_type'] == 'classification':
                        proba = model.predict_proba(input_array)[0]
                        classes = model.classes_
                        
                        st.subheader("📊 Prediction Probabilities:")
                        for class_label, prob in zip(classes, proba):
                            st.metric(f"Class {class_label}", f"{prob:.4f}")
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        
        elif input_method == "📁 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV file with feature values:",
                type=['csv'],
                key="prediction_csv_upload"
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
                        if st.button("🔮 **Make Batch Predictions**", key="make_batch_predictions"):
                            try:
                                # Make predictions
                                X_new = new_data[features]
                                predictions = model.predict(X_new)
                                
                                # Add predictions to dataframe
                                result_df = new_data.copy()
                                result_df[f'Predicted_{target}'] = predictions
                                
                                st.subheader("📊 Prediction Results:")
                                st.dataframe(result_df, use_container_width=True)
                                
                                # Download predictions
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 **Download Predictions**",
                                    data=csv,
                                    file_name=f"predictions_{model_info['model_id']}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            except Exception as e:
                                st.error(f"Batch prediction failed: {str(e)}")
                
                except Exception as e:
                    st.error(f"Failed to read CSV: {str(e)}")