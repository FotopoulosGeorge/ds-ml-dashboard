# feature_engineering.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from functools import cached_property

class FeatureEngineer:
    """Handles all feature engineering transformations"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = list(df.columns)
        self.new_features = []
        self._column_cache = {}
    # Add these methods to your existing feature_engineering.py

    # Replace your incomplete render_feature_engineering_tab method in feature_engineering.py with this complete version:

    def render_feature_engineering_tab(self):
        """
        Render the complete feature engineering tab UI
        """
        st.header("🔧 **Feature Engineering**")
        st.markdown("*Transform your data to improve machine learning model performance*")
        
        # Show current data info
        data_col1, data_col2, data_col3, data_col4 = st.columns(4)
        
        with data_col1:
            st.metric("Original Features", len(self.original_columns))
        with data_col2:
            st.metric("Current Features", len(self.df.columns))
        with data_col3:
            st.metric("New Features", len(self.new_features))
        with data_col4:
            st.metric("Data Points", len(self.df))
        
        st.markdown("---")
        
        # Feature engineering technique selection
        fe_technique = st.selectbox(
            "**Select Technique:**",
            ["🔢 Numerical Transformations", "🔍 Feature Summary"],
            key="fe_technique"
        )
        
        st.markdown("---")
        
        if fe_technique == "🔢 Numerical Transformations":
            st.subheader("🔢 Numerical Transformations")
            
            column_info = self.get_column_info()
            numeric_cols = column_info['numeric']
            
            if not numeric_cols:
                st.warning("No numeric columns available for transformation")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select column:", numeric_cols, key="fe_col")
                    transform_type = st.selectbox("Transformation:", 
                                                self.get_available_numerical_transforms(), key="fe_transform")
                
                with col2:
                    info = self.get_transform_info(transform_type)
                    st.info(f"**{transform_type}:** {info['description']}")
                    
                    # Handle special parameters
                    kwargs = {}
                    if transform_type == "Quantile Binning":
                        kwargs['n_bins'] = st.slider("Number of bins:", 2, 20, 5, key="fe_bins")
                    
                    new_name = st.text_input("New column name:", 
                                            value=f"{selected_col}_{transform_type.lower().replace(' ', '_')}", 
                                            key="fe_new_name")
                
                if st.button("🔧 Apply Transformation", key="fe_apply"):
                    try:
                        created_col = self.apply_numerical_transform(selected_col, transform_type, new_name, **kwargs)
                        st.success(f"✅ Created feature: {created_col}")
                        
                        # Show preview of new feature
                        if len(self.df.columns) >= 2:
                            preview_cols = [selected_col, created_col]
                            preview_df = self.df[preview_cols].head(10)
                            st.dataframe(preview_df, use_container_width=True)
                        
                        # Update working data automatically
                        st.session_state.working_df = self.df.copy()
                        st.info("💡 Feature added to working data! Check other tabs to see it.")
                        
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
        
        elif fe_technique == "🔍 Feature Summary":
            st.subheader("🔍 Feature Engineering Summary")
            
            # Get feature summary
            summary = self.get_feature_summary()
            
            # Display metrics
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Original Features", summary['original_features'])
            with summary_col2:
                st.metric("New Features", summary['new_features'])
            with summary_col3:
                st.metric("Total Features", summary['total_features'])
            
            # Show new features list
            if summary['new_features'] > 0:
                st.subheader("🆕 New Features Created")
                
                new_features_df = pd.DataFrame({
                    'Feature Name': summary['feature_list'],
                    'Data Type': [str(self.df[col].dtype) for col in summary['feature_list']],
                    'Non-Null Count': [self.df[col].count() for col in summary['feature_list']]
                })
                
                st.dataframe(new_features_df, use_container_width=True)
                
                # Show sample of new features
                if st.checkbox("Show sample data with new features", key="show_new_features_sample"):
                    # Show original columns + new features
                    sample_cols = self.original_columns[:3] + summary['feature_list']
                    available_cols = [col for col in sample_cols if col in self.df.columns]
                    if available_cols:
                        sample_df = self.df[available_cols].head(10)
                        st.dataframe(sample_df, use_container_width=True)
            else:
                st.info("No new features created yet. Use numerical transformations to create features.")
        
        # Apply changes section
        if len(self.new_features) > 0:
            st.markdown("---")
            apply_col1, apply_col2 = st.columns(2)
            
            with apply_col1:
                if st.button("✅ **Update Working Data**", type="primary", key="fe_apply_working"):
                    # Update the working dataframe in session state
                    st.session_state.working_df = self.df.copy()
                    st.success(f"✅ Applied {len(self.new_features)} new features to working data!")
                    st.info("💡 Switch to other tabs to see the new features in visualizations and exports")
            
            with apply_col2:
                if st.button("🔄 **Reset Features**", key="fe_reset"):
                    # Reset to original data but keep the current working data structure
                    current_data = st.session_state.working_df.copy()
                    # Remove only the new features, keep original + any filters
                    cols_to_keep = [col for col in current_data.columns if col not in self.new_features]
                    if cols_to_keep:
                        reset_data = current_data[cols_to_keep]
                        st.session_state.fe_handler = FeatureEngineer(reset_data)
                        st.success("🔄 Reset to original features")
                        st.rerun()
    # Add this helper method too:
    def get_current_data_from_session():
        """Static method to get current data from session state"""
        if 'working_df' in st.session_state and st.session_state.working_df is not None:
            return st.session_state.working_df.copy()
        elif 'base_df' in st.session_state and st.session_state.base_df is not None:
            return st.session_state.base_df.copy()
        else:
            return pd.DataFrame()
    # Numerical transformations
    def apply_numerical_transform(self, column, transform_type, new_col_name=None, **kwargs):
        """
        Apply numerical transformations to a specified column
        
        Parameters:
        -----------
        column : str
            Name of the column to transform
        transform_type : str
            Type of transformation ('Log Transform', 'Square Root', etc.)
        new_col_name : str, optional
            Name for the new column. If None, auto-generates name
        **kwargs : dict
            Additional parameters (e.g., n_bins for quantile binning)
        
        Returns:
        --------
        str : Name of the created column
        
        Raises:
        -------
        ValueError : If column doesn't exist or isn't numeric
        """
    
        # Validation
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        # Auto-generate column name if not provided
        if new_col_name is None:
            new_col_name = f"{column}_{transform_type.lower().replace(' ', '_')}"
        
        # Get the data
        data = self.df[column].copy()
        
        try:
            if transform_type == "Log Transform":
                # Handle negative values by adding constant
                min_val = data.min()
                if min_val <= 0:
                    # Add constant to make all values positive
                    transformed = np.log(data + abs(min_val) + 1)
                else:
                    transformed = np.log(data)
            
            elif transform_type == "Square Root":
                # Take absolute value to handle negative numbers
                transformed = np.sqrt(data.abs())
            
            elif transform_type == "Square":
                transformed = data ** 2
            
            elif transform_type == "Reciprocal":
                # Avoid division by zero by replacing 0 with NaN
                transformed = 1 / data.replace(0, np.nan)
            
            elif transform_type == "StandardScaler":
                # Z-score normalization: (x - mean) / std
                mean_val = data.mean()
                std_val = data.std()
                if std_val == 0:
                    raise ValueError("Cannot standardize: standard deviation is zero")
                transformed = (data - mean_val) / std_val
            
            elif transform_type == "MinMaxScaler":
                # Scale to [0, 1]: (x - min) / (max - min)
                min_val = data.min()
                max_val = data.max()
                if min_val == max_val:
                    raise ValueError("Cannot scale: all values are identical")
                transformed = (data - min_val) / (max_val - min_val)
            
            elif transform_type == "RobustScaler":
                # Scale using median and IQR: (x - median) / IQR
                median_val = data.median()
                q75 = data.quantile(0.75)
                q25 = data.quantile(0.25)
                iqr = q75 - q25
                if iqr == 0:
                    raise ValueError("Cannot robust scale: IQR is zero")
                transformed = (data - median_val) / iqr
            
            elif transform_type == "Quantile Binning":
                # Convert to discrete bins based on quantiles
                n_bins = kwargs.get('n_bins', 5)
                transformed = pd.qcut(data, q=n_bins, labels=False, duplicates='drop')
            
            else:
                raise ValueError(f"Unknown transformation type: {transform_type}")
            
            # Add the new column to dataframe
            self.df[new_col_name] = transformed
            
            # Track the new feature
            if new_col_name not in self.new_features:
                self.new_features.append(new_col_name)
            
            return new_col_name
        
        except Exception as e:
            raise ValueError(f"Transformation failed: {str(e)}")


    # Helper method to get available transformations
    def get_available_numerical_transforms(self):
        """Returns list of available numerical transformations"""
        return [
            "Log Transform",
            "Square Root", 
            "Square",
            "Reciprocal",
            "StandardScaler",
            "MinMaxScaler", 
            "RobustScaler",
            "Quantile Binning"
        ]


    # Helper method to get transformation info
    def get_transform_info(self, transform_type):
        """Returns description and requirements for each transformation"""
        info = {
            "Log Transform": {
                "description": "Natural logarithm transformation. Handles negative values automatically.",
                "use_case": "Reduce skewness, compress large ranges",
                "requirements": "Numeric data"
            },
            "Square Root": {
                "description": "Square root transformation. Uses absolute values for negative numbers.",
                "use_case": "Reduce right skewness, moderate compression", 
                "requirements": "Numeric data"
            },
            "StandardScaler": {
                "description": "Z-score normalization (mean=0, std=1)",
                "use_case": "When features have different scales",
                "requirements": "Numeric data, std > 0"
            },
            "MinMaxScaler": {
                "description": "Scale to range [0,1]",
                "use_case": "When you need bounded values",
                "requirements": "Numeric data, min ≠ max"
            },
            "Quantile Binning": {
                "description": "Convert to discrete bins based on quantiles",
                "use_case": "Create categorical features from continuous",
                "requirements": "Numeric data",
                "parameters": "n_bins (default: 5)"
            }
        }
        return info.get(transform_type, {"description": "No info available"})
       
        

    
    # Text feature extraction  
    def extract_text_features(self, column, features_list):
        """Extract text-based features"""
        # Implementation here
        pass
    
    # Date/time features
    def extract_date_features(self, column, features_list):
        """Extract temporal features"""
        # Implementation here
        pass
    
    # Categorical encoding
    def encode_categorical(self, column, encoding_type, target_col=None):
        """Various categorical encoding methods"""
        # Implementation here
        pass
    
    # Advanced features
    def create_interaction_features(self, col1, col2):
        """Create interaction terms"""
        # Implementation here
        pass
    
    def create_ratio_features(self, numerator, denominator):
        """Create ratio features"""
        # Implementation here
        pass
    
    # Statistical features
    def create_statistical_features(self, group_col, agg_col, functions):
        """Create group-wise statistical features"""
        # Implementation here
        pass
    
    def _update_column_cache(self):
        """Update column type cache when dataframe changes"""
        self._column_cache = {
            'numeric': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
            'all': self.df.columns.tolist()
        }

    def get_column_info(self):
        """Get all column types at once - computed only when needed"""
        if not self._column_cache or len(self._column_cache.get('all', [])) != len(self.df.columns):
            self._update_column_cache()
        return self._column_cache

    # Utility methods
    def get_feature_summary(self):
        """Return summary of created features"""
        return {
            'original_features': len(self.original_columns),
            'new_features': len(self.new_features), 
            'total_features': len(self.df.columns),
            'feature_list': self.new_features
        }
    
    def reset_to_original(self, original_df):
        """Reset to original dataset"""
        self.df = original_df.copy()
        self.new_features = []
        return self.df