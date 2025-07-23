# src/ml/ensemble/stacking.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.base import clone
import warnings
from src.ml.performance_decorator import ml_performance
warnings.filterwarnings('ignore')

class StackingEnsemble:
    """
    Stacked generalization where a meta-learner learns to combine base model predictions
    """
    
    def __init__(self):
        self.stacking_model = None
        self.base_models = []
        self.meta_learner = None
        self.cv_predictions = None
        
    def render_stacking_tab(self, df):
        """
        Main interface for stacking ensemble
        """
        st.header("🏗️ **Stacking Ensemble (Meta-Learning)**")
        st.markdown("*Use a meta-learner to optimally combine predictions from multiple base models*")
        
        # Check for available trained models
        if 'trained_models' not in st.session_state or not st.session_state.trained_models:
            st.warning("⚠️ No trained models available for stacking")
            st.info("💡 **First train multiple diverse models** in the supervised learning section, then return here to stack them")
            return
        
        # Filter models by problem type
        available_models = st.session_state.trained_models
        classification_models = {k: v for k, v in available_models.items() if v['problem_type'] == 'classification'}
        regression_models = {k: v for k, v in available_models.items() if v['problem_type'] == 'regression'}
        
        # Problem type selection
        st.subheader("🎯 Stacking Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            if classification_models and regression_models:
                problem_type = st.selectbox(
                    "**Problem Type:**",
                    ["Classification", "Regression"],
                    key="stacking_problem_type"
                )
            elif classification_models:
                problem_type = "Classification"
                st.info("**Detected:** Classification models available")
            elif regression_models:
                problem_type = "Regression"
                st.info("**Detected:** Regression models available")
            else:
                st.error("No compatible models found")
                return
        
        with config_col2:
            stack_method = st.selectbox(
                "**Stacking Method:**",
                ["Standard Stacking", "Multi-Level Stacking", "Blending"],
                help="Standard: Single meta-learner | Multi-Level: Multiple layers | Blending: Hold-out validation",
                key="stack_method"
            )
        
        # Select models for ensemble
        target_models = classification_models if problem_type == "Classification" else regression_models
        
        if len(target_models) < 2:
            st.warning(f"⚠️ Need at least 2 {problem_type.lower()} models for stacking. Found {len(target_models)}.")
            return
        
        # Base models selection
        st.subheader("🔧 Base Models Selection")
        
        selected_base_models = st.multiselect(
            f"**Select Base {problem_type} Models:**",
            list(target_models.keys()),
            default=list(target_models.keys())[:min(4, len(target_models))],
            help="Diverse models typically work better for stacking",
            key="selected_base_models"
        )
        
        if len(selected_base_models) < 2:
            st.warning("Please select at least 2 base models")
            return
        
        # Display base model information
        st.markdown("**📋 Selected Base Models:**")
        base_model_data = []
        for model_name in selected_base_models:
            model_info = target_models[model_name]
            base_model_data.append({
                'Model': model_name,
                'Algorithm': model_info['algorithm'],
                'Features': len(model_info['features']),
                'Type': model_info['problem_type']
            })
        
        base_models_df = pd.DataFrame(base_model_data)
        st.dataframe(base_models_df, use_container_width=True)
        
        # Meta-learner configuration
        st.subheader("🧠 Meta-Learner Configuration")
        
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        
        with meta_col1:
            if problem_type == "Classification":
                meta_learner_type = st.selectbox(
                    "**Meta-Learner:**",
                    ["Logistic Regression", "Random Forest", "Neural Network (Simple)"],
                    key="meta_learner_type"
                )
            else:
                meta_learner_type = st.selectbox(
                    "**Meta-Learner:**",
                    ["Linear Regression", "Random Forest", "Neural Network (Simple)"],
                    key="meta_learner_type"
                )
        
        with meta_col2:
            cv_folds = st.slider(
                "**CV Folds:**",
                min_value=3,
                max_value=10,
                value=5,
                help="Cross-validation folds for generating meta-features",
                key="stacking_cv_folds"
            )
        
        with meta_col3:
            use_probabilities = st.checkbox(
                "**Use Probabilities**",
                value=True if problem_type == "Classification" else False,
                help="Use class probabilities instead of predictions (classification only)",
                key="use_probabilities"
            )
        
        # Advanced stacking options
        with st.expander("⚙️ **Advanced Stacking Options**"):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                passthrough = st.checkbox(
                    "**Feature Passthrough**",
                    value=False,
                    help="Include original features along with base model predictions",
                    key="feature_passthrough"
                )
                
                stack_cv_strategy = st.selectbox(
                    "**CV Strategy:**",
                    ["Stratified" if problem_type == "Classification" else "Standard", "Time Series", "Custom"],
                    key="cv_strategy"
                )
            
            with adv_col2:
                final_estimator_params = st.text_area(
                    "**Meta-Learner Parameters (JSON):**",
                    value="{}",
                    help="Custom parameters for meta-learner in JSON format",
                    key="meta_params"
                )
        
        # Stacking execution
        st.subheader("🚀 Create Stacking Ensemble")
        
        exec_col1, exec_col2 = st.columns(2)
        
        with exec_col1:
            stacking_name = st.text_input(
                "**Stacking Ensemble Name:**",
                value=f"Stacking_{len(selected_base_models)}models_{datetime.now().strftime('%H%M')}",
                key="stacking_name"
            )
        
        with exec_col2:
            evaluation_metrics = st.multiselect(
                "**Evaluation Metrics:**",
                ["Accuracy", "Precision", "Recall", "F1-Score"] if problem_type == "Classification" 
                else ["R² Score", "MSE", "MAE", "RMSE"],
                default=["Accuracy"] if problem_type == "Classification" else ["R² Score"],
                key="eval_metrics"
            )
        
        if st.button("🏗️ **Create Stacking Ensemble**", type="primary", key="create_stacking"):
            try:
                with st.spinner('Creating stacking ensemble...'):
                    stacking_result = self._create_stacking_ensemble(
                        selected_base_models, target_models, meta_learner_type,
                        problem_type, cv_folds, stack_method, df
                    )
                
                if stacking_result:
                    self._display_stacking_results(stacking_result, stacking_name, evaluation_metrics)
                    
                    # Store stacking ensemble
                    if 'stacking_ensembles' not in st.session_state:
                        st.session_state.stacking_ensembles = {}
                    
                    st.session_state.stacking_ensembles[stacking_name] = {
                        'stacking_model': stacking_result['stacking_model'],
                        'base_models': selected_base_models,
                        'meta_learner': meta_learner_type,
                        'method': stack_method,
                        'problem_type': problem_type,
                        'performance': stacking_result['performance'],
                        'created_at': datetime.now()
                    }
                    
                    st.success(f"✅ Stacking ensemble '{stacking_name}' created and saved!")
                    
            except Exception as e:
                st.error(f"❌ Stacking ensemble creation failed: {str(e)}")
                st.info("💡 Check model compatibility and ensure sufficient data")
    

    @ml_performance(
        "ensemble", 
        dataset_param="df", 
        config_params=["selected_base_models", "meta_learner_type", "cv_folds", "stack_method"]
    )
    def _create_stacking_ensemble(self, selected_base_models, target_models, meta_learner_type, 
                                problem_type, cv_folds, stack_method, df):
        """Create the stacking ensemble"""
        
        # Prepare base models
        base_estimators = []
        for model_name in selected_base_models:
            model_info = target_models[model_name]
            model = clone(model_info['model'])  # Clone to avoid modifying original
            base_estimators.append((model_name, model))
        
        # Configure meta-learner
        meta_learner = self._get_meta_learner(meta_learner_type, problem_type)
        
        # Get training data (from first model)
        first_model_info = target_models[selected_base_models[0]]
        
        if 'training_data' in first_model_info:
            X_train, y_train = first_model_info['training_data']
        else:
            # Fallback: use current data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Insufficient numeric features for stacking")
            
            target_col = numeric_cols[-1]
            feature_cols = numeric_cols[:-1]
            
            X_train = df[feature_cols].dropna()
            y_train = df[target_col].loc[X_train.index]
        
        # Create stacking ensemble
        if problem_type == "Classification":
            stacking_model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=cv_folds,
                stack_method='predict_proba' if st.session_state.get('use_probabilities', True) else 'predict',
                passthrough=st.session_state.get('feature_passthrough', False)
            )
        else:
            stacking_model = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=cv_folds,
                passthrough=st.session_state.get('feature_passthrough', False)
            )
        
        # Fit the stacking ensemble
        stacking_model.fit(X_train, y_train)
        
        # Evaluate the ensemble
        performance = self._evaluate_stacking_ensemble(
            stacking_model, selected_base_models, target_models, X_train, y_train, problem_type
        )
        
        # Generate cross-validation predictions for analysis
        cv_predictions = self._generate_cv_predictions(
            base_estimators, X_train, y_train, cv_folds, problem_type
        )
        
        result = {
            'stacking_model': stacking_model,
            'base_models': selected_base_models,
            'meta_learner': meta_learner_type,
            'problem_type': problem_type,
            'performance': performance,
            'cv_predictions': cv_predictions,
            'training_data': (X_train, y_train)
        }
        
        return result
    
    def _get_meta_learner(self, meta_learner_type, problem_type):
        """Get configured meta-learner"""
        try:
            import json
            meta_params = json.loads(st.session_state.get('meta_params', '{}'))
        except:
            meta_params = {}
        
        if problem_type == "Classification":
            if meta_learner_type == "Logistic Regression":
                return LogisticRegression(random_state=42, **meta_params)
            elif meta_learner_type == "Random Forest":
                default_params = {'n_estimators': 100, 'random_state': 42}
                default_params.update(meta_params)
                return RandomForestClassifier(**default_params)
            elif meta_learner_type == "Neural Network (Simple)":
                try:
                    from sklearn.neural_network import MLPClassifier
                    default_params = {'hidden_layer_sizes': (50,), 'random_state': 42, 'max_iter': 500}
                    default_params.update(meta_params)
                    return MLPClassifier(**default_params)
                except ImportError:
                    st.warning("Neural network not available, using Logistic Regression")
                    return LogisticRegression(random_state=42)
        else:  # Regression
            if meta_learner_type == "Linear Regression":
                return LinearRegression(**meta_params)
            elif meta_learner_type == "Random Forest":
                default_params = {'n_estimators': 100, 'random_state': 42}
                default_params.update(meta_params)
                return RandomForestRegressor(**default_params)
            elif meta_learner_type == "Neural Network (Simple)":
                try:
                    from sklearn.neural_network import MLPRegressor
                    default_params = {'hidden_layer_sizes': (50,), 'random_state': 42, 'max_iter': 500}
                    default_params.update(meta_params)
                    return MLPRegressor(**default_params)
                except ImportError:
                    st.warning("Neural network not available, using Linear Regression")
                    return LinearRegression()
        
        # Default fallback
        return LogisticRegression(random_state=42) if problem_type == "Classification" else LinearRegression()
    
    def _evaluate_stacking_ensemble(self, stacking_model, selected_base_models, target_models, X_train, y_train, problem_type):
        """Evaluate stacking ensemble performance"""
        performance = {
            'stacking_scores': {},
            'base_model_scores': {},
            'meta_learner_importance': {},
            'cross_validation': {}
        }
        
        # Get test data for evaluation
        first_model_info = target_models[selected_base_models[0]]
        if 'test_data' in first_model_info:
            X_test, y_test = first_model_info['test_data']
            
            # Evaluate stacking ensemble
            y_pred_stacking = stacking_model.predict(X_test)
            
            if problem_type == "Classification":
                stacking_score = accuracy_score(y_test, y_pred_stacking)
                performance['stacking_scores']['accuracy'] = stacking_score
                
                # Additional classification metrics
                if hasattr(stacking_model, 'predict_proba'):
                    y_proba = stacking_model.predict_proba(X_test)
                    from sklearn.metrics import roc_auc_score, log_loss
                    try:
                        if len(np.unique(y_test)) == 2:  # Binary classification
                            auc_score = roc_auc_score(y_test, y_proba[:, 1])
                            performance['stacking_scores']['auc'] = auc_score
                        logloss = log_loss(y_test, y_proba)
                        performance['stacking_scores']['log_loss'] = logloss
                    except:
                        pass
            else:
                r2_score_val = r2_score(y_test, y_pred_stacking)
                mse_score = mean_squared_error(y_test, y_pred_stacking)
                performance['stacking_scores']['r2'] = r2_score_val
                performance['stacking_scores']['mse'] = mse_score
                performance['stacking_scores']['rmse'] = np.sqrt(mse_score)
            
            # Evaluate individual base models for comparison
            for model_name in selected_base_models:
                model_info = target_models[model_name]
                model = model_info['model']
                y_pred_base = model.predict(X_test)
                
                if problem_type == "Classification":
                    base_score = accuracy_score(y_test, y_pred_base)
                else:
                    base_score = r2_score(y_test, y_pred_base)
                
                performance['base_model_scores'][model_name] = base_score
        
        # Cross-validation performance
        from sklearn.model_selection import cross_val_score
        cv_folds = st.session_state.get('stacking_cv_folds', 5)
        
        try:
            if problem_type == "Classification":
                cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            else:
                cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=cv_folds, scoring='r2')
            
            performance['cross_validation'] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
        except Exception as e:
            st.warning(f"Cross-validation evaluation failed: {str(e)}")
        
        return performance
    
    def _generate_cv_predictions(self, base_estimators, X_train, y_train, cv_folds, problem_type):
        """Generate cross-validation predictions for meta-feature analysis"""
        cv_predictions = {}
        
        try:
            # Use appropriate CV strategy
            if problem_type == "Classification":
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for name, estimator in base_estimators:
                if problem_type == "Classification" and hasattr(estimator, 'predict_proba'):
                    # Use probabilities for classification
                    cv_pred = cross_val_predict(estimator, X_train, y_train, cv=cv, method='predict_proba')
                else:
                    # Use predictions
                    cv_pred = cross_val_predict(estimator, X_train, y_train, cv=cv)
                
                cv_predictions[name] = cv_pred
        
        except Exception as e:
            st.warning(f"CV predictions generation failed: {str(e)}")
        
        return cv_predictions
    
    def _display_stacking_results(self, stacking_result, stacking_name, evaluation_metrics):
        """Display comprehensive stacking results"""
        st.success("✅ Stacking ensemble created successfully!")
        
        performance = stacking_result['performance']
        
        # Performance overview
        st.subheader("📊 Stacking Performance")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Base Models", len(stacking_result['base_models']))
        
        with perf_col2:
            st.metric("Meta-Learner", stacking_result['meta_learner'])
        
        with perf_col3:
            if 'accuracy' in performance['stacking_scores']:
                st.metric("Stacking Accuracy", f"{performance['stacking_scores']['accuracy']:.4f}")
            elif 'r2' in performance['stacking_scores']:
                st.metric("Stacking R² Score", f"{performance['stacking_scores']['r2']:.4f}")
        
        with perf_col4:
            if 'cross_validation' in performance:
                cv_mean = performance['cross_validation']['mean_score']
                cv_std = performance['cross_validation']['std_score']
                st.metric("CV Score", f"{cv_mean:.4f} ± {cv_std:.4f}")
        
        # Detailed performance metrics
        st.subheader("📈 Detailed Performance")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("**🏗️ Stacking Ensemble Scores:**")
            for metric, score in performance['stacking_scores'].items():
                st.write(f"• **{metric.upper()}:** {score:.4f}")
        
        with detail_col2:
            if performance['base_model_scores']:
                st.markdown("**🔧 Base Model Scores:**")
                for model, score in performance['base_model_scores'].items():
                    st.write(f"• **{model}:** {score:.4f}")
        
        # Performance comparison visualization
        if performance['base_model_scores']:
            st.subheader("🏆 Stacking vs Base Models")
            
            # Create comparison data
            comparison_data = []
            
            # Add base model scores
            for model_name, score in performance['base_model_scores'].items():
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Base Model',
                    'Score': score
                })
            
            # Add stacking score
            stacking_score = (performance['stacking_scores'].get('accuracy') or 
                            performance['stacking_scores'].get('r2'))
            if stacking_score:
                comparison_data.append({
                    'Model': 'Stacking Ensemble',
                    'Type': 'Stacking',
                    'Score': stacking_score
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Score', ascending=False)
            
            # Visualization
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Score',
                color='Type',
                title='Base Models vs Stacking Ensemble Performance',
                color_discrete_map={'Base Model': 'lightblue', 'Stacking': 'red'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show improvement
            if stacking_score:
                best_base_score = max(performance['base_model_scores'].values())
                improvement = stacking_score - best_base_score
                improvement_pct = (improvement / best_base_score) * 100
                
                if improvement > 0:
                    st.success(f"🎯 **Improvement:** +{improvement:.4f} ({improvement_pct:.2f}%) over best base model")
                else:
                    st.warning(f"⚠️ **Performance:** {improvement:.4f} ({improvement_pct:.2f}%) vs best base model")
        
        # Cross-validation analysis
        if 'cross_validation' in performance:
            st.subheader("🔄 Cross-Validation Analysis")
            
            cv_scores = performance['cross_validation']['scores']
            cv_df = pd.DataFrame({
                'Fold': range(1, len(cv_scores) + 1),
                'Score': cv_scores
            })
            
            cv_col1, cv_col2 = st.columns(2)
            
            with cv_col1:
                fig = px.line(
                    cv_df,
                    x='Fold',
                    y='Score',
                    title='Cross-Validation Scores by Fold',
                    markers=True
                )
                fig.add_hline(y=np.mean(cv_scores), line_dash="dash", 
                            annotation_text=f"Mean: {np.mean(cv_scores):.4f}")
                st.plotly_chart(fig, use_container_width=True)
            
            with cv_col2:
                st.markdown("**📊 CV Statistics:**")
                st.write(f"• **Mean:** {np.mean(cv_scores):.4f}")
                st.write(f"• **Std:** {np.std(cv_scores):.4f}")
                st.write(f"• **Min:** {np.min(cv_scores):.4f}")
                st.write(f"• **Max:** {np.max(cv_scores):.4f}")
                
                cv_stability = np.std(cv_scores) / np.mean(cv_scores) * 100
                st.write(f"• **Stability:** {cv_stability:.2f}% CV")
        
        # Meta-features analysis
        if stacking_result['cv_predictions']:
            self._analyze_meta_features(stacking_result['cv_predictions'], stacking_result['problem_type'])
        
        # Model architecture details
        with st.expander("🏗️ **Stacking Architecture Details**"):
            st.write(f"**Base Models:** {len(stacking_result['base_models'])}")
            for i, model_name in enumerate(stacking_result['base_models'], 1):
                st.write(f"  {i}. {model_name}")
            
            st.write(f"**Meta-Learner:** {stacking_result['meta_learner']}")
            st.write(f"**Problem Type:** {stacking_result['problem_type']}")
            
            stacking_model = stacking_result['stacking_model']
            if hasattr(stacking_model, 'stack_method_'):
                st.write(f"**Stack Method:** {stacking_model.stack_method_}")
            if hasattr(stacking_model, 'passthrough'):
                st.write(f"**Feature Passthrough:** {stacking_model.passthrough}")
        
        # Save options
        st.subheader("💾 Save Stacking Ensemble")
        
        save_col1, save_col2 = st.columns(2)
        
        with save_col1:
            if st.button("💾 **Save as Trained Model**", key="save_stacking_model"):
                try:
                    model_id = f"stacking_{stacking_name}"
                    
                    stacking_info = {
                        'model': stacking_result['stacking_model'],
                        'algorithm': f"Stacking Ensemble ({stacking_result['meta_learner']})",
                        'target': 'stacking_target',
                        'features': [],
                        'problem_type': stacking_result['problem_type'].lower(),
                        'test_data': stacking_result.get('training_data', (None, None)),
                        'model_id': model_id,
                        'stacking_info': stacking_result
                    }
                    
                    st.session_state.trained_models[model_id] = stacking_info
                    st.success(f"✅ Stacking ensemble saved as: {model_id}")
                    st.info("💡 Now available for predictions and further analysis")
                    
                except Exception as e:
                    st.error(f"❌ Failed to save: {str(e)}")
        
        with save_col2:
            # Download configuration
            config_data = {
                'stacking_name': stacking_name,
                'base_models': stacking_result['base_models'],
                'meta_learner': stacking_result['meta_learner'],
                'performance': performance,
                'created_at': datetime.now().isoformat()
            }
            
            import json
            config_json = json.dumps(config_data, indent=2, default=str)
            
            st.download_button(
                label="📥 **Download Config**",
                data=config_json,
                file_name=f"{stacking_name}_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    def _analyze_meta_features(self, cv_predictions, problem_type):
        """Analyze meta-features generated by base models"""
        st.subheader("🔍 Meta-Features Analysis")
        
        if not cv_predictions:
            st.info("No meta-features available for analysis")
            return
        
        # Meta-feature statistics
        meta_stats = []
        for model_name, predictions in cv_predictions.items():
            if predictions.ndim > 1:
                # Probabilities (classification)
                meta_stats.append({
                    'Model': model_name,
                    'Shape': str(predictions.shape),
                    'Mean': f"{np.mean(predictions):.4f}",
                    'Std': f"{np.std(predictions):.4f}",
                    'Type': 'Probabilities' if problem_type == "Classification" else 'Predictions'
                })
            else:
                # Predictions
                meta_stats.append({
                    'Model': model_name,
                    'Shape': str(predictions.shape),
                    'Mean': f"{np.mean(predictions):.4f}",
                    'Std': f"{np.std(predictions):.4f}",
                    'Type': 'Predictions'
                })
        
        meta_stats_df = pd.DataFrame(meta_stats)
        st.dataframe(meta_stats_df, use_container_width=True)
        
        # Correlation between meta-features
        if len(cv_predictions) > 1:
            st.markdown("**🔗 Meta-Feature Correlations:**")
            
            # Create correlation matrix for meta-features
            meta_features_df = pd.DataFrame()
            
            for model_name, predictions in cv_predictions.items():
                if predictions.ndim == 1:
                    meta_features_df[model_name] = predictions
                else:
                    # For probabilities, use the positive class (index 1) or first column
                    meta_features_df[model_name] = predictions[:, -1] if predictions.shape[1] > 1 else predictions[:, 0]
            
            if not meta_features_df.empty:
                corr_matrix = meta_features_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Meta-Feature Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # High correlation warning
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.9:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if high_corr_pairs:
                    st.warning("⚠️ **High correlations detected:**")
                    for model1, model2, corr in high_corr_pairs:
                        st.write(f"• {model1} ↔ {model2}: {corr:.3f}")
                    st.info("💡 Consider using more diverse base models for better stacking performance")