# src/ml/ensemble/model_chaining.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import warnings
from src.ml.performance_decorator import ml_performance
warnings.filterwarnings('ignore')

class ModelChainer:
    """
    Sequential model chaining where output of one model feeds into the next
    """
    
    def __init__(self):
        self.chain = []
        self.chain_metadata = {}
        self.fitted_chain = None
        self.performance_history = []
    
    def render_chaining_tab(self, df):
        """
        Main interface for sequential model chaining
        """
        st.header("🔗 **Sequential Model Chaining**")
        st.markdown("*Create chains where the output of one model feeds into the next*")
        
        # Check for available trained models
        if 'trained_models' not in st.session_state or not st.session_state.trained_models:
            st.warning("⚠️ No trained models available for chaining")
            st.info("💡 **First train some models** in the supervised learning section, then return here to chain them")
            return
        
        # Get available models
        available_models = st.session_state.trained_models
        model_options = list(available_models.keys())
        
        # Chain configuration section
        st.subheader("🔧 Chain Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            chain_type = st.selectbox(
                "**Chain Type:**",
                ["Sequential Prediction", "Hierarchical Classification", "Multi-Stage Refinement"],
                help="Sequential: A→B→C | Hierarchical: Broad→Specific | Refinement: Coarse→Fine",
                key="chain_type"
            )
        
        with config_col2:
            max_chain_length = st.slider(
                "**Maximum Chain Length:**",
                min_value=2,
                max_value=min(10, len(model_options)),
                value=3,
                key="max_chain_length"
            )
        
        # Chain builder interface
        st.subheader("🏗️ Build Your Chain")
        
        chain_models = []
        chain_valid = True
        
        for i in range(max_chain_length):
            with st.expander(f"🔗 **Chain Position {i+1}**", expanded=i < 2):
                pos_col1, pos_col2 = st.columns(2)
                
                with pos_col1:
                    if i == 0:
                        # First model - can be any model
                        available_options = model_options
                        help_text = "First model in the chain - receives original input data"
                    else:
                        # Subsequent models - should match previous output type
                        prev_model_info = available_models[chain_models[i-1]] if i-1 < len(chain_models) else None
                        if prev_model_info:
                            prev_problem_type = prev_model_info['problem_type']
                            # Filter compatible models
                            available_options = [
                                name for name, info in available_models.items() 
                                if self._models_compatible(prev_model_info, info)
                            ]
                            help_text = f"Must be compatible with previous model output ({prev_problem_type})"
                        else:
                            available_options = model_options
                            help_text = "Select a model for this position"
                    
                    selected_model = st.selectbox(
                        f"Model {i+1}:",
                        ["None"] + available_options,
                        key=f"chain_model_{i}",
                        help=help_text
                    )
                    
                    if selected_model != "None":
                        chain_models.append(selected_model)
                        
                        # Show model info
                        model_info = available_models[selected_model]
                        st.info(f"**{model_info['algorithm']}** | {model_info['problem_type']} | {len(model_info['features'])} features")
                    else:
                        break
                
                with pos_col2:
                    if selected_model != "None":
                        # Data flow configuration
                        if i > 0:
                            flow_type = st.selectbox(
                                "Data Flow:",
                                ["Prediction Only", "Prediction + Original Features", "Custom Selection"],
                                key=f"flow_type_{i}",
                                help="How to combine previous predictions with original data"
                            )
                            
                            if flow_type == "Custom Selection":
                                # Allow user to select which features to pass forward
                                prev_features = available_models[chain_models[i-1]]['features']
                                selected_features = st.multiselect(
                                    "Features to include:",
                                    prev_features,
                                    default=prev_features[:3],
                                    key=f"custom_features_{i}"
                                )
        
        # Validate chain
        if len(chain_models) < 2:
            chain_valid = False
            st.warning("⚠️ Need at least 2 models to create a chain")
        
        # Chain summary
        if chain_valid and chain_models:
            st.subheader("📋 Chain Summary")
            
            summary_data = []
            for i, model_name in enumerate(chain_models):
                model_info = available_models[model_name]
                summary_data.append({
                    'Position': i + 1,
                    'Model': model_name,
                    'Algorithm': model_info['algorithm'],
                    'Type': model_info['problem_type'],
                    'Features': len(model_info['features'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Visualize chain flow
            self._visualize_chain_flow(chain_models, available_models)
        
        # Execute chain
        if chain_valid and chain_models and len(chain_models) >= 2:
            st.subheader("🚀 Execute Chain")
            
            exec_col1, exec_col2 = st.columns(2)
            
            with exec_col1:
                test_data_source = st.selectbox(
                    "**Test Data:**",
                    ["Use Model Test Sets", "Current Working Data", "Upload New Data"],
                    key="test_data_source"
                )
            
            with exec_col2:
                chain_name = st.text_input(
                    "**Chain Name:**",
                    value=f"Chain_{len(chain_models)}models_{datetime.now().strftime('%H%M')}",
                    key="chain_name"
                )
            
            if st.button("🔗 **Execute Model Chain**", type="primary", key="execute_chain"):
                try:
                    with st.spinner('Executing model chain...'):
                        chain_result = self._execute_model_chain(
                            chain_models, available_models, df, test_data_source
                        )
                    
                    if chain_result:
                        self._display_chain_results(chain_result, chain_name)
                        
                        # Store chain for future use
                        if 'model_chains' not in st.session_state:
                            st.session_state.model_chains = {}
                        
                        st.session_state.model_chains[chain_name] = {
                            'chain_models': chain_models,
                            'chain_result': chain_result,
                            'created_at': datetime.now(),
                            'type': 'sequential'
                        }
                        
                        st.success(f"✅ Chain '{chain_name}' created and saved!")
                        
                except Exception as e:
                    st.error(f"❌ Chain execution failed: {str(e)}")
                    st.info("💡 Check model compatibility and data requirements")
    
    def _models_compatible(self, model1_info, model2_info):
        """
        Check if two models can be chained together
        """
        # For now, simple compatibility: regression can feed into regression/classification
        # Classification can feed into classification
        
        model1_type = model1_info['problem_type']
        model2_type = model2_info['problem_type']
        
        if model1_type == 'regression':
            return True  # Regression output can feed into any model
        elif model1_type == 'classification' and model2_type == 'classification':
            return True  # Classification to classification
        elif model1_type == 'clustering':
            return model2_type in ['classification', 'regression']  # Cluster labels can be features
        
        return False
    
    def _visualize_chain_flow(self, chain_models, available_models):
        """
        Create visual representation of the model chain
        """
        st.subheader("🔄 Chain Flow Visualization")
        
        # Create a simple flow diagram using plotly
        fig = go.Figure()
        
        # Create nodes for each model
        x_positions = list(range(len(chain_models)))
        y_position = [0] * len(chain_models)
        
        # Add nodes
        for i, model_name in enumerate(chain_models):
            model_info = available_models[model_name]
            
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=50, color='lightblue', line=dict(width=2, color='darkblue')),
                text=f"{model_info['algorithm']}<br>({model_info['problem_type']})",
                textposition="middle center",
                name=f"Model {i+1}",
                showlegend=False
            ))
        
        # Add arrows between nodes
        for i in range(len(chain_models) - 1):
            fig.add_annotation(
                x=i + 0.5,
                y=0,
                text="→",
                showarrow=False,
                font=dict(size=20, color='darkgreen')
            )
        
        fig.update_layout(
            title="Model Chain Flow",
            xaxis=dict(title="Chain Position", showgrid=False, zeroline=False),
            yaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            height=200,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    

    @ml_performance(
        "ensemble", 
        dataset_param="df", 
        config_params=["chain_models", "test_data_source"]
    )
    def _execute_model_chain(self, chain_models, available_models, df, test_data_source):
        """
        Execute the complete model chain
        """
        # Get test data based on source
        if test_data_source == "Current Working Data":
            test_data = df.copy()
        else:
            # Use the first model's test data as starting point
            first_model_info = available_models[chain_models[0]]
            if 'test_data' in first_model_info:
                X_test, y_test = first_model_info['test_data']
                test_data = X_test.copy()
            else:
                raise ValueError("No test data available for first model")
        
        # Execute chain step by step
        chain_results = {
            'models': chain_models,
            'step_results': [],
            'final_predictions': None,
            'step_predictions': [],
            'performance_metrics': []
        }
        
        current_data = test_data.copy()
        
        for i, model_name in enumerate(chain_models):
            model_info = available_models[model_name]
            model = model_info['model']
            
            step_result = {
                'model_name': model_name,
                'position': i + 1,
                'input_shape': current_data.shape,
                'model_features': model_info['features']
            }
            
            try:
                # Prepare input data for this model
                if i == 0:
                    # First model uses original features
                    required_features = model_info['features']
                    if all(feat in current_data.columns for feat in required_features):
                        model_input = current_data[required_features]
                    else:
                        # Use available numeric features if exact match not found
                        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                        model_input = current_data[numeric_cols.tolist()[:len(required_features)]]
                else:
                    # Subsequent models: combine predictions with original features
                    prev_predictions = chain_results['step_predictions'][-1]
                    
                    # Create feature matrix with previous prediction
                    if isinstance(prev_predictions, np.ndarray):
                        if prev_predictions.ndim == 1:
                            prev_pred_df = pd.DataFrame({'prev_prediction': prev_predictions})
                        else:
                            prev_pred_df = pd.DataFrame(prev_predictions, columns=[f'prev_pred_{j}' for j in range(prev_predictions.shape[1])])
                    else:
                        prev_pred_df = pd.DataFrame({'prev_prediction': prev_predictions})
                    
                    # Combine with original features (simplified approach)
                    numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        combined_features = pd.concat([
                            prev_pred_df.reset_index(drop=True),
                            current_data[numeric_cols].reset_index(drop=True)
                        ], axis=1)
                    else:
                        combined_features = prev_pred_df
                    
                    # Select features needed for this model
                    required_features = model_info['features']
                    n_features_needed = len(required_features)
                    
                    if combined_features.shape[1] >= n_features_needed:
                        model_input = combined_features.iloc[:, :n_features_needed]
                        model_input.columns = required_features
                    else:
                        # Pad with zeros if not enough features
                        model_input = combined_features.copy()
                        for j in range(combined_features.shape[1], n_features_needed):
                            model_input[required_features[j]] = 0
                
                # Make prediction
                predictions = model.predict(model_input)
                step_result['predictions'] = predictions
                step_result['output_shape'] = predictions.shape
                
                # Store step results
                chain_results['step_predictions'].append(predictions)
                chain_results['step_results'].append(step_result)
                
                # Evaluate if we have ground truth
                if i == 0 and 'test_data' in model_info:
                    X_test, y_test = model_info['test_data']
                    if model_info['problem_type'] == 'classification':
                        accuracy = accuracy_score(y_test, predictions)
                        step_result['accuracy'] = accuracy
                    elif model_info['problem_type'] == 'regression':
                        mse = mean_squared_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        step_result['mse'] = mse
                        step_result['r2'] = r2
            
            except Exception as e:
                step_result['error'] = str(e)
                st.warning(f"⚠️ Error in model {i+1} ({model_name}): {str(e)}")
        
        # Final predictions are from the last model
        if chain_results['step_predictions']:
            chain_results['final_predictions'] = chain_results['step_predictions'][-1]
        
        return chain_results
    
    def _display_chain_results(self, chain_result, chain_name):
        """
        Display comprehensive chain execution results
        """
        st.success("✅ Model chain executed successfully!")
        
        # Chain overview
        st.subheader("📊 Chain Execution Summary")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Chain Length", len(chain_result['models']))
        with summary_col2:
            st.metric("Steps Executed", len(chain_result['step_results']))
        with summary_col3:
            if chain_result['final_predictions'] is not None:
                st.metric("Final Predictions", len(chain_result['final_predictions']))
            else:
                st.metric("Final Predictions", "N/A")
        with summary_col4:
            errors = sum(1 for step in chain_result['step_results'] if 'error' in step)
            st.metric("Errors", errors, delta=f"{'❌' if errors > 0 else '✅'}")
        
        # Step-by-step results
        st.subheader("🔗 Step-by-Step Results")
        
        for i, step in enumerate(chain_result['step_results']):
            with st.expander(f"📍 **Step {step['position']}: {step['model_name']}**"):
                step_col1, step_col2 = st.columns(2)
                
                with step_col1:
                    st.write(f"**Input Shape:** {step['input_shape']}")
                    st.write(f"**Output Shape:** {step.get('output_shape', 'N/A')}")
                    
                    if 'error' in step:
                        st.error(f"❌ **Error:** {step['error']}")
                
                with step_col2:
                    # Performance metrics if available
                    if 'accuracy' in step:
                        st.metric("Accuracy", f"{step['accuracy']:.4f}")
                    if 'r2' in step:
                        st.metric("R² Score", f"{step['r2']:.4f}")
                    if 'mse' in step:
                        st.metric("MSE", f"{step['mse']:.4f}")
                
                # Show sample predictions
                if 'predictions' in step:
                    predictions = step['predictions']
                    st.write("**Sample Predictions:**")
                    if isinstance(predictions, np.ndarray):
                        sample_preds = predictions[:5]
                        st.code(f"{sample_preds}")
        
        # Final predictions analysis
        if chain_result['final_predictions'] is not None:
            st.subheader("🎯 Final Chain Predictions")
            
            final_preds = chain_result['final_predictions']
            
            # Basic statistics
            if isinstance(final_preds, np.ndarray) and final_preds.dtype in ['int64', 'float64']:
                pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                
                with pred_col1:
                    st.metric("Mean", f"{np.mean(final_preds):.4f}")
                with pred_col2:
                    st.metric("Std", f"{np.std(final_preds):.4f}")
                with pred_col3:
                    st.metric("Min", f"{np.min(final_preds):.4f}")
                with pred_col4:
                    st.metric("Max", f"{np.max(final_preds):.4f}")
                
                # Distribution plot
                fig = px.histogram(
                    x=final_preds,
                    nbins=30,
                    title="Final Prediction Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download predictions
            pred_df = pd.DataFrame({'Chain_Predictions': final_preds})
            csv_data = pred_df.to_csv(index=False)
            
            st.download_button(
                label="📥 **Download Chain Predictions**",
                data=csv_data,
                file_name=f"{chain_name}_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Chain performance comparison
        self._plot_chain_performance(chain_result)
    
    def _plot_chain_performance(self, chain_result):
        """
        Plot performance evolution through the chain
        """
        st.subheader("📈 Chain Performance Evolution")
        
        # Extract performance metrics from each step
        step_names = []
        accuracies = []
        r2_scores = []
        
        for step in chain_result['step_results']:
            step_names.append(f"Step {step['position']}")
            
            if 'accuracy' in step:
                accuracies.append(step['accuracy'])
            else:
                accuracies.append(None)
            
            if 'r2' in step:
                r2_scores.append(step['r2'])
            else:
                r2_scores.append(None)
        
        # Plot if we have metrics
        if any(acc is not None for acc in accuracies) or any(r2 is not None for r2 in r2_scores):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if any(acc is not None for acc in accuracies):
                valid_accuracies = [(i, acc) for i, acc in enumerate(accuracies) if acc is not None]
                if valid_accuracies:
                    indices, values = zip(*valid_accuracies)
                    fig.add_trace(
                        go.Scatter(x=[step_names[i] for i in indices], y=values, 
                                 name="Accuracy", line=dict(color='blue')),
                        secondary_y=False
                    )
            
            if any(r2 is not None for r2 in r2_scores):
                valid_r2 = [(i, r2) for i, r2 in enumerate(r2_scores) if r2 is not None]
                if valid_r2:
                    indices, values = zip(*valid_r2)
                    fig.add_trace(
                        go.Scatter(x=[step_names[i] for i in indices], y=values, 
                                 name="R² Score", line=dict(color='red')),
                        secondary_y=True
                    )
            
            fig.update_layout(title="Performance Through Chain Steps")
            fig.update_xaxes(title="Chain Step")
            fig.update_yaxes(title="Accuracy", secondary_y=False)
            fig.update_yaxes(title="R² Score", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance metrics available for visualization")
    
    def save_chain_configuration(self, chain_models, chain_name):
        """Save chain configuration for later use"""
        config = {
            'chain_name': chain_name,
            'models': chain_models,
            'created_at': datetime.now().isoformat(),
            'type': 'sequential'
        }
        
        # Store in session state
        if 'saved_chains' not in st.session_state:
            st.session_state.saved_chains = {}
        
        st.session_state.saved_chains[chain_name] = config
        
        return config
    
    def load_chain_configuration(self, chain_name):
        """Load a saved chain configuration"""
        if 'saved_chains' in st.session_state:
            return st.session_state.saved_chains.get(chain_name)
        return None