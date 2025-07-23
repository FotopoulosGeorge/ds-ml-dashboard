# src/ml/ensemble/ensemble_methods.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.base import clone
import warnings
from src.ml.performance_decorator import ml_performance
warnings.filterwarnings('ignore')

class EnsembleMethods:
    """
    Classical ensemble methods: Voting, Bagging, Boosting
    """
    
    def __init__(self):
        self.ensemble_model = None
        self.ensemble_type = None
        self.base_models = []
        
    def render_ensemble_tab(self, df):
        """
        Main interface for ensemble methods
        """
        st.header("🗳️ **Ensemble Methods**")
        st.markdown("*Combine multiple models to improve predictions through voting, bagging, and boosting*")
        
        # Check for available trained models
        if 'trained_models' not in st.session_state or not st.session_state.trained_models:
            st.warning("⚠️ No trained models available for ensemble")
            st.info("💡 **First train multiple models** in the supervised learning section, then return here to combine them")
            return
        
        # Filter models by problem type
        available_models = st.session_state.trained_models
        classification_models = {k: v for k, v in available_models.items() if v['problem_type'] == 'classification'}
        regression_models = {k: v for k, v in available_models.items() if v['problem_type'] == 'regression'}
        
        # Ensemble method selection
        st.subheader("🎯 Ensemble Method Selection")
        
        method_col1, method_col2 = st.columns(2)
        
        with method_col1:
            ensemble_method = st.selectbox(
                "**Ensemble Method:**",
                ["Voting Ensemble", "Bagging Ensemble", "Custom Weighted Ensemble"],
                key="ensemble_method"
            )
        
        with method_col2:
            if classification_models and regression_models:
                problem_type = st.selectbox(
                    "**Problem Type:**",
                    ["Classification", "Regression"],
                    key="ensemble_problem_type"
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
        
        # Select models for ensemble
        target_models = classification_models if problem_type == "Classification" else regression_models
        
        if len(target_models) < 2:
            st.warning(f"⚠️ Need at least 2 {problem_type.lower()} models for ensemble. Found {len(target_models)}.")
            return
        
        st.subheader("🔧 Ensemble Configuration")
        
        # Model selection
        selected_models = st.multiselect(
            f"**Select {problem_type} Models:**",
            list(target_models.keys()),
            default=list(target_models.keys())[:min(3, len(target_models))],
            help=f"Choose {problem_type.lower()} models to combine in the ensemble",
            key="selected_ensemble_models"
        )
        
        if len(selected_models) < 2:
            st.warning("Please select at least 2 models")
            return
        
        # Method-specific configuration
        if ensemble_method == "Voting Ensemble":
            self._configure_voting_ensemble(selected_models, target_models, problem_type)
        elif ensemble_method == "Bagging Ensemble":
            self._configure_bagging_ensemble(selected_models, target_models, problem_type)
        elif ensemble_method == "Custom Weighted Ensemble":
            self._configure_weighted_ensemble(selected_models, target_models, problem_type)
        
        # Ensemble execution
        if len(selected_models) >= 2:
            st.subheader("🚀 Create Ensemble")
            
            exec_col1, exec_col2 = st.columns(2)
            
            with exec_col1:
                ensemble_name = st.text_input(
                    "**Ensemble Name:**",
                    value=f"{ensemble_method.replace(' ', '')}_{len(selected_models)}models_{datetime.now().strftime('%H%M')}",
                    key="ensemble_name"
                )
            
            with exec_col2:
                evaluation_method = st.selectbox(
                    "**Evaluation Method:**",
                    ["Cross-Validation", "Hold-out Test", "Both"],
                    key="eval_method"
                )
            button_text = "🎯 **Create Ensemble**"
            button_help = "Create the ensemble model"

            if st.button(button_text, type="primary", key="create_ensemble", help=button_help):
                try:
                    with st.spinner('Creating ensemble...'):
                        ensemble_result = self._create_ensemble(
                            selected_models, target_models, ensemble_method, 
                            problem_type, evaluation_method, df
                        )

                        if ensemble_result:
                            self._display_ensemble_results(ensemble_result, ensemble_name)
                            
                            # Store ensemble
                            if 'ensemble_models' not in st.session_state:
                                st.session_state.ensemble_models = {}
                            
                            st.session_state.ensemble_models[ensemble_name] = {
                                'ensemble_model': ensemble_result['ensemble_model'],
                                'base_models': selected_models,
                                'method': ensemble_method,
                                'problem_type': problem_type,
                                'performance': ensemble_result['performance'],
                                'created_at': datetime.now()
                            }
                            
                    st.success(f"✅ Ensemble '{ensemble_name}' created successfully!")
                            
                except Exception as e:
                    st.error(f"❌ Ensemble creation failed: {str(e)}")
                    st.info("💡 Check model compatibility and data requirements")                   
    
    def _configure_voting_ensemble(self, selected_models, target_models, problem_type):
        """Configure voting ensemble parameters"""
        st.markdown("**🗳️ Voting Ensemble Configuration**")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            if problem_type == "Classification":
                voting_type = st.selectbox(
                    "**Voting Type:**",
                    ["hard", "soft"],
                    format_func=lambda x: "Hard Voting (Majority)" if x == "hard" else "Soft Voting (Probability)",
                    help="Hard: Majority vote | Soft: Average probabilities",
                    key="voting_type"
                )
            else:
                voting_type = "hard"  # Only option for regression
                st.info("**Voting Type:** Average predictions (regression)")
        
        with config_col2:
            equal_weights = st.checkbox(
                "**Equal Weights**",
                value=True,
                help="Give all models equal importance",
                key="equal_weights"
            )
        
        # Show model weights configuration
        if not equal_weights:
            st.markdown("**⚖️ Model Weights:**")
            weights = {}
            weight_cols = st.columns(min(3, len(selected_models)))
            
            for i, model_name in enumerate(selected_models):
                with weight_cols[i % 3]:
                    model_info = target_models[model_name]
                    weight = st.slider(
                        f"{model_info['algorithm']}:",
                        min_value=0.1,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        key=f"weight_{model_name}"
                    )
                    weights[model_name] = weight
            
            # Store weights in session state
            st.session_state.ensemble_weights = weights
        
        # Show voting ensemble info
        with st.expander("ℹ️ **Voting Ensemble Information**"):
            st.markdown(f"""
            **Method:** Voting Ensemble
            **Models:** {len(selected_models)} base models
            **Type:** {voting_type if problem_type == "Classification" else "averaging"}
            
            **How it works:**
            - Each model makes independent predictions
            - {"Final prediction is majority vote" if voting_type == "hard" else "Final prediction is averaged probability"} 
            - {"Weights determine model importance" if not equal_weights else "All models have equal influence"}
            
            **Benefits:**
            - Reduces overfitting
            - More robust predictions
            - Combines different model strengths
            """)
    
    def _configure_bagging_ensemble(self, selected_models, target_models, problem_type):
        """Configure bagging ensemble parameters"""
        st.markdown("**🎒 Bagging Ensemble Configuration**")
        
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            n_estimators = st.slider(
                "**Number of Estimators:**",
                min_value=3,
                max_value=50,
                value=10,
                help="Number of models to train with different data samples",
                key="bagging_n_estimators"
            )
        
        with config_col2:
            max_samples = st.slider(
                "**Sample Fraction:**",
                min_value=0.3,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Fraction of samples to use for each model",
                key="bagging_max_samples"
            )
        
        with config_col3:
            bootstrap = st.checkbox(
                "**Bootstrap Sampling**",
                value=True,
                help="Sample with replacement",
                key="bagging_bootstrap"
            )
        
        # Base model selection for bagging
        if len(selected_models) == 1:
            st.info(f"**Base Model:** {target_models[selected_models[0]]['algorithm']}")
        else:
            base_model_for_bagging = st.selectbox(
                "**Base Model for Bagging:**",
                selected_models,
                help="Single model type to use for bagging (will create multiple instances)",
                key="bagging_base_model"
            )
            st.session_state.bagging_base_model = base_model_for_bagging
        
        # Show bagging info
        with st.expander("ℹ️ **Bagging Ensemble Information**"):
            st.markdown(f"""
            **Method:** Bagging (Bootstrap Aggregating)
            **Estimators:** {n_estimators}
            **Sample Size:** {max_samples * 100:.0f}% of data
            **Bootstrap:** {"Yes" if bootstrap else "No"}
            
            **How it works:**
            - Train multiple instances of the same model
            - Each model sees different subset of data
            - {"Sample with replacement" if bootstrap else "Sample without replacement"}
            - Final prediction is average of all models
            
            **Benefits:**
            - Reduces variance
            - Parallel training possible
            - Works well with high-variance models
            """)
    
    def _configure_weighted_ensemble(self, selected_models, target_models, problem_type):
        """Configure custom weighted ensemble"""
        st.markdown("**⚖️ Custom Weighted Ensemble Configuration**")
        
        st.info("Configure custom weights based on model performance or domain knowledge")
        
        # Weight configuration method
        weight_method = st.selectbox(
            "**Weight Assignment:**",
            ["Manual", "Performance-Based", "Cross-Validation Score"],
            key="weight_method"
        )
        
        weights = {}
        
        if weight_method == "Manual":
            st.markdown("**🎛️ Manual Weight Configuration:**")
            weight_cols = st.columns(min(3, len(selected_models)))
            
            for i, model_name in enumerate(selected_models):
                with weight_cols[i % 3]:
                    model_info = target_models[model_name]
                    weight = st.slider(
                        f"**{model_info['algorithm']}**:",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0/len(selected_models),
                        step=0.05,
                        key=f"manual_weight_{model_name}"
                    )
                    weights[model_name] = weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
                st.info(f"**Normalized weights:** {', '.join([f'{k}: {v:.2f}' for k, v in weights.items()])}")
            else:
                st.error("Total weight cannot be zero")
        
        elif weight_method == "Performance-Based":
            st.markdown("**📊 Performance-Based Weights:**")
            st.info("Weights will be calculated based on individual model performance on test data")
            
            # Show model performance if available
            for model_name in selected_models:
                model_info = target_models[model_name]
                if 'test_data' in model_info:
                    X_test, y_test = model_info['test_data']
                    model = model_info['model']
                    y_pred = model.predict(X_test)
                    
                    if problem_type == "Classification":
                        score = accuracy_score(y_test, y_pred)
                        metric = "Accuracy"
                    else:
                        score = r2_score(y_test, y_pred)
                        metric = "R² Score"
                    
                    st.metric(f"{model_info['algorithm']}", f"{score:.4f}", help=metric)
                    weights[model_name] = score
            
            # Normalize performance-based weights
            if weights:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}
        
        st.session_state.custom_weights = weights
        
        # Show weight summary
        if weights:
            weight_df = pd.DataFrame([
                {'Model': k, 'Algorithm': target_models[k]['algorithm'], 'Weight': f"{v:.3f}"}
                for k, v in weights.items()
            ])
            st.dataframe(weight_df, use_container_width=True)
    @ml_performance(
        "ensemble", 
        dataset_param="df", 
        config_params=["ensemble_method", "selected_models", "evaluation_method"]
    )
    def _create_ensemble(self, selected_models, target_models, ensemble_method, problem_type, evaluation_method, df):
        """Create the actual ensemble model"""
        
        # Prepare base models
        base_models = []
        for model_name in selected_models:
            model_info = target_models[model_name]
            model = model_info['model']
            
            # Clone the model to avoid modifying the original
            cloned_model = clone(model)
            base_models.append((model_name, cloned_model))
        
        # Create ensemble based on method
        if ensemble_method == "Voting Ensemble":
            ensemble_model = self._create_voting_ensemble(base_models, problem_type)
        elif ensemble_method == "Bagging Ensemble":
            ensemble_model = self._create_bagging_ensemble(base_models, problem_type)
        elif ensemble_method == "Custom Weighted Ensemble":
            ensemble_model = self._create_weighted_ensemble(base_models, problem_type)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        # Prepare training data (use data from first model)
        first_model_info = target_models[selected_models[0]]
        
        if 'training_data' in first_model_info:
            X_train, y_train = first_model_info['training_data']
        else:
            # Fallback: use current data with numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Insufficient numeric features for ensemble training")
            
            target_col = numeric_cols[-1]  # Use last numeric column as target
            feature_cols = numeric_cols[:-1]
            
            X_train = df[feature_cols].dropna()
            y_train = df[target_col].loc[X_train.index]
        
        # Fit the ensemble
        ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        performance = self._evaluate_ensemble(
            ensemble_model, selected_models, target_models, evaluation_method
        )
        
        result = {
            'ensemble_model': ensemble_model,
            'base_models': selected_models,
            'method': ensemble_method,
            'problem_type': problem_type,
            'performance': performance,
            'training_data': (X_train, y_train)
        }
        
        return result
    
    def _create_voting_ensemble(self, base_models, problem_type):
        """Create voting ensemble"""
        voting_type = st.session_state.get('voting_type', 'hard')
        
        if problem_type == "Classification":
            ensemble = VotingClassifier(
                estimators=base_models,
                voting=voting_type
            )
        else:
            ensemble = VotingRegressor(
                estimators=base_models
            )
        
        return ensemble
    
    def _create_bagging_ensemble(self, base_models, problem_type):
        """Create bagging ensemble"""
        n_estimators = st.session_state.get('bagging_n_estimators', 10)
        max_samples = st.session_state.get('bagging_max_samples', 0.8)
        bootstrap = st.session_state.get('bagging_bootstrap', True)
        
        # Use first model as base estimator for bagging
        base_estimator = base_models[0][1]
        
        if problem_type == "Classification":
            ensemble = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                bootstrap=bootstrap,
                random_state=42
            )
        else:
            ensemble = BaggingRegressor(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                bootstrap=bootstrap,
                random_state=42
            )
        
        return ensemble
    
    def _create_weighted_ensemble(self, base_models, problem_type):
        """Create custom weighted ensemble (simplified implementation)"""
        weights = st.session_state.get('custom_weights', {})
        
        # For weighted ensemble, we'll use VotingClassifier/Regressor with custom weights
        # Note: This is a simplified implementation
        model_weights = [weights.get(name, 1.0) for name, _ in base_models]
        
        if problem_type == "Classification":
            ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft'  # Use soft voting for weighted
            )
        else:
            ensemble = VotingRegressor(
                estimators=base_models,
                weights=model_weights
            )
        
        return ensemble
    
    def _evaluate_ensemble(self, ensemble_model, selected_models, target_models, evaluation_method):
        """Evaluate ensemble performance"""
        performance = {
            'ensemble_scores': {},
            'individual_scores': {},
            'comparison': {}
        }
        
        # Get test data from first model
        first_model_info = target_models[selected_models[0]]
        if 'test_data' in first_model_info:
            X_test, y_test = first_model_info['test_data']
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble_model.predict(X_test)
            
            if first_model_info['problem_type'] == 'classification':
                ensemble_score = accuracy_score(y_test, y_pred_ensemble)
                metric_name = 'accuracy'
            else:
                ensemble_score = r2_score(y_test, y_pred_ensemble)
                metric_name = 'r2_score'
            
            performance['ensemble_scores'][metric_name] = ensemble_score
            
            # Evaluate individual models for comparison
            for model_name in selected_models:
                model_info = target_models[model_name]
                model = model_info['model']
                y_pred_individual = model.predict(X_test)
                
                if first_model_info['problem_type'] == 'classification':
                    individual_score = accuracy_score(y_test, y_pred_individual)
                else:
                    individual_score = r2_score(y_test, y_pred_individual)
                
                performance['individual_scores'][model_name] = individual_score
            
            # Calculate improvement
            best_individual = max(performance['individual_scores'].values())
            improvement = ensemble_score - best_individual
            performance['comparison']['improvement'] = improvement
            performance['comparison']['best_individual'] = best_individual
        
        return performance
    
    def _display_ensemble_results(self, ensemble_result, ensemble_name):
        """Display comprehensive ensemble results"""
        st.success("✅ Ensemble created successfully!")
        
        performance = ensemble_result['performance']
        
        # Performance overview
        st.subheader("📊 Ensemble Performance")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Base Models", len(ensemble_result['base_models']))
        
        with perf_col2:
            if 'accuracy' in performance['ensemble_scores']:
                st.metric("Ensemble Accuracy", f"{performance['ensemble_scores']['accuracy']:.4f}")
            elif 'r2_score' in performance['ensemble_scores']:
                st.metric("Ensemble R² Score", f"{performance['ensemble_scores']['r2_score']:.4f}")
        
        with perf_col3:
            if 'best_individual' in performance['comparison']:
                st.metric("Best Individual", f"{performance['comparison']['best_individual']:.4f}")
        
        with perf_col4:
            if 'improvement' in performance['comparison']:
                improvement = performance['comparison']['improvement']
                st.metric("Improvement", f"{improvement:.4f}", 
                         delta=f"{'📈' if improvement > 0 else '📉'}")
        
        # Individual vs Ensemble comparison
        if performance['individual_scores']:
            st.subheader("🏆 Model Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, score in performance['individual_scores'].items():
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Individual',
                    'Score': score
                })
            
            # Add ensemble score
            ensemble_score = (performance['ensemble_scores'].get('accuracy') or 
                            performance['ensemble_scores'].get('r2_score'))
            if ensemble_score:
                comparison_data.append({
                    'Model': f"{ensemble_result['method']} Ensemble",
                    'Type': 'Ensemble',
                    'Score': ensemble_score
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Score', ascending=False)
            
            # Display comparison table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Score',
                color='Type',
                title='Individual Models vs Ensemble Performance',
                color_discrete_map={'Individual': 'lightblue', 'Ensemble': 'orange'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Ensemble details
        with st.expander("🔧 **Ensemble Configuration Details**"):
            st.write(f"**Method:** {ensemble_result['method']}")
            st.write(f"**Problem Type:** {ensemble_result['problem_type']}")
            st.write(f"**Base Models:** {', '.join(ensemble_result['base_models'])}")
            
            if hasattr(ensemble_result['ensemble_model'], 'estimators_'):
                st.write(f"**Number of Estimators:** {len(ensemble_result['ensemble_model'].estimators_)}")
            
            if hasattr(ensemble_result['ensemble_model'], 'weights'):
                weights = ensemble_result['ensemble_model'].weights
                if weights:
                    st.write(f"**Weights:** {weights}")
        
        # Save ensemble for later use
        st.subheader("💾 Save Ensemble")
        
        save_col1, save_col2 = st.columns(2)
        
        with save_col1:
            if st.button("💾 **Save Ensemble Model**", key="save_ensemble"):
                try:
                    # Add to trained models for general use
                    model_id = f"ensemble_{ensemble_name}"
                    
                    # Create model info similar to individual models
                    ensemble_info = {
                        'model': ensemble_result['ensemble_model'],
                        'algorithm': f"Ensemble - {ensemble_result['method']}",
                        'target': 'ensemble_target',  # Generic target
                        'features': [],  # Will be filled when used
                        'problem_type': ensemble_result['problem_type'].lower(),
                        'test_data': ensemble_result.get('training_data', (None, None)),
                        'model_id': model_id,
                        'ensemble_info': ensemble_result
                    }
                    
                    st.session_state.trained_models[model_id] = ensemble_info
                    st.success(f"✅ Ensemble saved as trained model: {model_id}")
                    st.info("💡 You can now use this ensemble for predictions in the ML section")
                    
                except Exception as e:
                    st.error(f"❌ Failed to save ensemble: {str(e)}")
        
        with save_col2:
            # Download ensemble configuration
            config_data = {
                'ensemble_name': ensemble_name,
                'method': ensemble_result['method'],
                'base_models': ensemble_result['base_models'],
                'performance': performance,
                'created_at': datetime.now().isoformat()
            }
            
            import json
            config_json = json.dumps(config_data, indent=2, default=str)
            
            st.download_button(
                label="📥 **Download Config**",
                data=config_json,
                file_name=f"{ensemble_name}_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                help="Download ensemble configuration for reference"
            )