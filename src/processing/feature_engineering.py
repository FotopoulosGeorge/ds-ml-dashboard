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
            [
                "🔢 Numerical Transformations", 
                "🧮 Column Operations",
                "📝 Text Features",
                "📅 Date Features", 
                "🏷️ Categorical Encoding",
                "🔍 Feature Summary"
                ],
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
        
        elif fe_technique == "🧮 Column Operations":  # ← ADD THIS NEW SECTION
            self._render_column_operations()

        elif fe_technique == "📝 Text Features":
            st.subheader("📝 Text Feature Extraction")
            
            column_info = self.get_column_info()
            text_cols = column_info['categorical']  # Object/string columns
            
            if not text_cols:
                st.warning("No text columns available")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select text column:", text_cols, key="text_col")
                    text_features = st.multiselect(
                        "Features to extract:",
                        ["length", "word_count", "char_count", "uppercase_count", "digit_count", "special_char_count"],
                        default=["length", "word_count"],
                        key="text_features"
                    )
                
                with col2:
                    st.info("**Text Features:**\n• Length: Character count\n• Word count: Number of words\n• Char count: Non-space characters\n• Uppercase/digits/special chars")
                
                if st.button("📝 Extract Text Features", key="text_apply"):
                    try:
                        created = self.extract_text_features(selected_col, text_features)
                        st.success(f"✅ Created {len(created)} text features")
                        st.session_state.working_df = self.df.copy()
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")

        elif fe_technique == "📅 Date Features":
            st.subheader("📅 Date/Time Feature Extraction")
            
            column_info = self.get_column_info()
            date_cols = column_info['datetime']
            
            if not date_cols:
                st.warning("No datetime columns available")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select date column:", date_cols, key="date_col")
                    date_features = st.multiselect(
                        "Features to extract:",
                        ["year", "month", "day", "dayofweek", "quarter", "hour", "is_weekend", "days_since_epoch"],
                        default=["year", "month", "dayofweek"],
                        key="date_features"
                    )
                
                if st.button("📅 Extract Date Features", key="date_apply"):
                    try:
                        created = self.extract_date_features(selected_col, date_features)
                        st.success(f"✅ Created {len(created)} date features")
                        st.session_state.working_df = self.df.copy()
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")

        elif fe_technique == "🏷️ Categorical Encoding":
            st.subheader("🏷️ Categorical Encoding")
            
            column_info = self.get_column_info()
            cat_cols = column_info['categorical']
            numeric_cols = column_info['numeric']
            
            if not cat_cols:
                st.warning("No categorical columns available")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select categorical column:", cat_cols, key="cat_col")
                    encoding_type = st.selectbox(
                        "Encoding method:",
                        ["One-Hot Encoding", "Label Encoding", "Frequency Encoding", "Target Encoding"],
                        key="encoding_type"
                    )
                
                with col2:
                    if encoding_type == "Target Encoding":
                        target_col = st.selectbox("Select target column:", numeric_cols, key="target_col")
                    else:
                        target_col = None
                    
                    st.info(f"**{encoding_type}:**\n• Creates new encoded features")
                
                if st.button("🏷️ Apply Encoding", key="encoding_apply"):
                    try:
                        created = self.encode_categorical(selected_col, encoding_type, target_col)
                        st.success(f"✅ Created {len(created)} encoded features")
                        st.session_state.working_df = self.df.copy()
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
       
        
    def _render_column_operations(self):
        """
        NEW METHOD: Render the column operations interface
        """
        st.subheader("🧮 Column Operations")
        st.markdown("*Combine columns using mathematical operations*")
        
        column_info = self.get_column_info()
        numeric_cols = column_info['numeric']
        all_cols = column_info['all']
        
        if len(all_cols) < 2:
            st.warning("Need at least 2 columns for operations")
            return
        
        # Operation type selection
        operation_type = st.selectbox(
            "**Operation Type:**",
            ["🔢 Numerical Operations", "📝 Text Combinations"],
            key="operation_type"
        )
        
        if operation_type == "🔢 Numerical Operations":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for numerical operations")
                return
                
            # UI for numerical operations
            op_col1, op_col2, op_col3 = st.columns(3)
            
            with op_col1:
                col1 = st.selectbox("**First Column:**", numeric_cols, key="num_col1")
            
            with op_col2:
                operation = st.selectbox(
                    "**Operation:**",
                    ["multiply", "add", "subtract", "divide"],
                    format_func=lambda x: {
                        "multiply": "Multiply (×)",
                        "add": "Add (+)", 
                        "subtract": "Subtract (-)",
                        "divide": "Divide (÷)"
                    }[x],
                    key="num_operation"
                )
            
            with op_col3:
                col2 = st.selectbox("**Second Column:**", numeric_cols, key="num_col2")
            
            # New column name
            default_name = f"{col1}_{operation}_{col2}"
            new_col_name = st.text_input(
                "**New Column Name:**",
                value=default_name,
                key="num_new_name"
            )
            
            # Preview the operation
            operation_symbols = {
                "multiply": "×", "add": "+", "subtract": "-", "divide": "÷"
            }
            st.info(f"**Formula:** {new_col_name} = {col1} {operation_symbols[operation]} {col2}")
            
            if st.button("🧮 **Create Feature**", type="primary", key="create_numerical"):
                try:
                    created_feature = self.create_interaction_features(col1, col2, operation)
                    # Rename if user specified different name
                    if created_feature != new_col_name and new_col_name:
                        self.df[new_col_name] = self.df[created_feature]
                        self.df.drop(columns=[created_feature], inplace=True)
                        # Update the new_features list
                        if created_feature in self.new_features:
                            self.new_features.remove(created_feature)
                        if new_col_name not in self.new_features:
                            self.new_features.append(new_col_name)
                        created_feature = new_col_name
                    
                    st.success(f"✅ Created feature: **{created_feature}**")
                    
                    # Show preview
                    preview_cols = [col1, col2, created_feature]
                    preview_df = self.df[preview_cols].head(5)
                    st.dataframe(preview_df, use_container_width=True)
                    
                    # Update working data
                    st.session_state.working_df = self.df.copy()
                    st.info("💡 Feature added to working data! Check other tabs to see it.")
                    
                except Exception as e:
                    st.error(f"❌ Operation failed: {str(e)}")
        
        elif operation_type == "📝 Text Combinations":
            # UI for text combinations
            text_col1, text_col2 = st.columns(2)
            
            with text_col1:
                col1 = st.selectbox("**First Column:**", all_cols, key="text_col1")
            
            with text_col2:
                col2 = st.selectbox("**Second Column:**", all_cols, key="text_col2")
            
            # Separator and new name
            separator_col, name_col = st.columns(2)
            
            with separator_col:
                separator = st.text_input("**Separator:**", value="_", key="text_separator")
            
            with name_col:
                default_name = f"{col1}_concat_{col2}"
                new_col_name = st.text_input(
                    "**New Column Name:**",
                    value=default_name,
                    key="text_new_name"
                )
            
            # Preview
            st.info(f"**Formula:** {new_col_name} = {col1} + '{separator}' + {col2}")
            
            if st.button("🔗 **Combine Columns**", type="primary", key="create_text"):
                try:
                    created_feature = self.create_interaction_features(col1, col2, "concat")
                    
                    # If user wants custom separator, recreate with custom separator
                    if separator != "_":
                        self.df[new_col_name] = (
                            self.df[col1].astype(str) + separator + self.df[col2].astype(str)
                        )
                        # Remove the default one and update tracking
                        if created_feature in self.df.columns:
                            self.df.drop(columns=[created_feature], inplace=True)
                        if created_feature in self.new_features:
                            self.new_features.remove(created_feature)
                        if new_col_name not in self.new_features:
                            self.new_features.append(new_col_name)
                        created_feature = new_col_name
                    elif created_feature != new_col_name and new_col_name:
                        # Just rename
                        self.df[new_col_name] = self.df[created_feature]
                        self.df.drop(columns=[created_feature], inplace=True)
                        if created_feature in self.new_features:
                            self.new_features.remove(created_feature)
                        if new_col_name not in self.new_features:
                            self.new_features.append(new_col_name)
                        created_feature = new_col_name
                    
                    st.success(f"✅ Created feature: **{created_feature}**")
                    
                    # Show preview
                    preview_cols = [col1, col2, created_feature]
                    preview_df = self.df[preview_cols].head(5)
                    st.dataframe(preview_df, use_container_width=True)
                    
                    # Update working data
                    st.session_state.working_df = self.df.copy()
                    st.info("💡 Feature added to working data! Check other tabs to see it.")
                    
                except Exception as e:
                    st.error(f"❌ Combination failed: {str(e)}")
        
    # Text feature extraction  
    def extract_text_features(self, column, features_list):
        """Extract text-based features from a text column"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if not pd.api.types.is_string_dtype(self.df[column]) and self.df[column].dtype != 'object':
            raise ValueError(f"Column '{column}' is not text data")
        
        data = self.df[column].fillna('')  # Handle NaN values
        created_features = []
        
        if 'length' in features_list:
            feature_name = f"{column}_length"
            self.df[feature_name] = data.str.len()
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'word_count' in features_list:
            feature_name = f"{column}_word_count"
            self.df[feature_name] = data.str.split().str.len()
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'char_count' in features_list:
            feature_name = f"{column}_char_count"
            self.df[feature_name] = data.str.replace(' ', '').str.len()
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'uppercase_count' in features_list:
            feature_name = f"{column}_uppercase_count"
            self.df[feature_name] = data.str.count(r'[A-Z]')
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'digit_count' in features_list:
            feature_name = f"{column}_digit_count"
            self.df[feature_name] = data.str.count(r'\d')
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'special_char_count' in features_list:
            feature_name = f"{column}_special_chars"
            self.df[feature_name] = data.str.count(r'[^a-zA-Z0-9\s]')
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        return created_features
    
    # Date/time features
    def extract_date_features(self, column, features_list):
        """Extract temporal features from datetime column"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if not pd.api.types.is_datetime64_any_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not datetime data")
        
        data = self.df[column]
        created_features = []
        
        if 'year' in features_list:
            feature_name = f"{column}_year"
            self.df[feature_name] = data.dt.year
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'month' in features_list:
            feature_name = f"{column}_month"
            self.df[feature_name] = data.dt.month
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'day' in features_list:
            feature_name = f"{column}_day"
            self.df[feature_name] = data.dt.day
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'dayofweek' in features_list:
            feature_name = f"{column}_dayofweek"
            self.df[feature_name] = data.dt.dayofweek
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'quarter' in features_list:
            feature_name = f"{column}_quarter"
            self.df[feature_name] = data.dt.quarter
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'hour' in features_list:
            feature_name = f"{column}_hour"
            self.df[feature_name] = data.dt.hour
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'is_weekend' in features_list:
            feature_name = f"{column}_is_weekend"
            self.df[feature_name] = (data.dt.dayofweek >= 5).astype(int)
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        if 'days_since_epoch' in features_list:
            feature_name = f"{column}_days_since_epoch"
            epoch = pd.Timestamp('1970-01-01')
            self.df[feature_name] = (data - epoch).dt.days
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        return created_features
        
    # Categorical encoding
    def encode_categorical(self, column, encoding_type, target_col=None):
        """Various categorical encoding methods"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        data = self.df[column]
        created_features = []
        
        if encoding_type == "One-Hot Encoding":
            # Create dummy variables
            dummies = pd.get_dummies(data, prefix=column, dummy_na=True)
            for col in dummies.columns:
                self.df[col] = dummies[col]
                created_features.append(col)
                if col not in self.new_features:
                    self.new_features.append(col)
        
        elif encoding_type == "Label Encoding":
            feature_name = f"{column}_label_encoded"
            # Simple integer encoding
            unique_vals = data.unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.df[feature_name] = data.map(mapping)
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        elif encoding_type == "Frequency Encoding":
            feature_name = f"{column}_frequency"
            freq_map = data.value_counts().to_dict()
            self.df[feature_name] = data.map(freq_map)
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        elif encoding_type == "Target Encoding" and target_col:
            if target_col not in self.df.columns:
                raise ValueError(f"Target column '{target_col}' not found")
            
            feature_name = f"{column}_target_encoded"
            # Calculate mean target value for each category
            target_means = self.df.groupby(column)[target_col].mean()
            self.df[feature_name] = data.map(target_means)
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        return created_features
    
    # Advanced features
    def create_interaction_features(self, col1, col2, interaction_type="multiply"):
        """Create interaction terms between two columns"""
        if col1 not in self.df.columns or col2 not in self.df.columns:
            raise ValueError("One or both columns not found")
        
        data1 = self.df[col1]
        data2 = self.df[col2]
        
        # Check if both are numeric for mathematical operations
        if interaction_type in ["multiply", "add", "subtract", "divide"]:
            if not (pd.api.types.is_numeric_dtype(data1) and pd.api.types.is_numeric_dtype(data2)):
                raise ValueError("Both columns must be numeric for mathematical interactions")
        
        if interaction_type == "multiply":
            feature_name = f"{col1}_x_{col2}"
            self.df[feature_name] = data1 * data2
        
        elif interaction_type == "add":
            feature_name = f"{col1}_plus_{col2}"
            self.df[feature_name] = data1 + data2
        
        elif interaction_type == "subtract":
            feature_name = f"{col1}_minus_{col2}"
            self.df[feature_name] = data1 - data2
        
        elif interaction_type == "divide":
            feature_name = f"{col1}_div_{col2}"
            # Avoid division by zero
            self.df[feature_name] = data1 / data2.replace(0, np.nan)
        
        elif interaction_type == "concat":
            feature_name = f"{col1}_concat_{col2}"
            self.df[feature_name] = data1.astype(str) + "_" + data2.astype(str)
        
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
        
        if feature_name not in self.new_features:
            self.new_features.append(feature_name)
        
        return feature_name
    
    def create_ratio_features(self, numerator, denominator, handle_zero="nan"):
        """Create ratio features between two numeric columns"""
        if numerator not in self.df.columns or denominator not in self.df.columns:
            raise ValueError("Column not found")
        
        if not (pd.api.types.is_numeric_dtype(self.df[numerator]) and 
                pd.api.types.is_numeric_dtype(self.df[denominator])):
            raise ValueError("Both columns must be numeric")
        
        feature_name = f"{numerator}_to_{denominator}_ratio"
        
        num_data = self.df[numerator]
        den_data = self.df[denominator]
        
        if handle_zero == "nan":
            # Replace zero denominators with NaN
            self.df[feature_name] = num_data / den_data.replace(0, np.nan)
        elif handle_zero == "small_value":
            # Replace zero with small value
            self.df[feature_name] = num_data / den_data.replace(0, 1e-10)
        elif handle_zero == "skip":
            # Only calculate where denominator is not zero
            mask = den_data != 0
            self.df[feature_name] = np.nan
            self.df.loc[mask, feature_name] = num_data[mask] / den_data[mask]
        
        if feature_name not in self.new_features:
            self.new_features.append(feature_name)
        
        return feature_name
    
    # Advance feature creation
    def create_aggregation_features(self, group_col, agg_col, functions):
        """Create group-wise statistical features"""
        if group_col not in self.df.columns or agg_col not in self.df.columns:
            raise ValueError("Column not found")
        
        if not pd.api.types.is_numeric_dtype(self.df[agg_col]):
            raise ValueError(f"Aggregation column '{agg_col}' must be numeric")
        
        created_features = []
        
        for func in functions:
            if func == "mean":
                feature_name = f"{agg_col}_mean_by_{group_col}"
                group_means = self.df.groupby(group_col)[agg_col].mean()
                self.df[feature_name] = self.df[group_col].map(group_means)
            
            elif func == "std":
                feature_name = f"{agg_col}_std_by_{group_col}"
                group_stds = self.df.groupby(group_col)[agg_col].std()
                self.df[feature_name] = self.df[group_col].map(group_stds)
            
            elif func == "count":
                feature_name = f"{group_col}_count"
                group_counts = self.df.groupby(group_col).size()
                self.df[feature_name] = self.df[group_col].map(group_counts)
            
            elif func == "rank":
                feature_name = f"{agg_col}_rank_in_{group_col}"
                self.df[feature_name] = self.df.groupby(group_col)[agg_col].rank()
            
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        return created_features

    def create_lag_features(self, column, periods=[1], sort_col=None):
        """Create lag/lead features (useful for time series)"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        created_features = []
        df_copy = self.df.copy()
        
        if sort_col:
            df_copy = df_copy.sort_values(sort_col)
        
        for period in periods:
            if period > 0:
                feature_name = f"{column}_lag_{period}"
                df_copy[feature_name] = df_copy[column].shift(period)
            else:
                feature_name = f"{column}_lead_{abs(period)}"
                df_copy[feature_name] = df_copy[column].shift(period)
            
            created_features.append(feature_name)
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)
        
        # Update the main dataframe with new features
        for feature in created_features:
            self.df[feature] = df_copy[feature]
        
        return created_features

    def create_outlier_features(self, column, method="iqr"):
        """Create features that identify outliers"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' must be numeric")
        
        data = self.df[column]
        created_features = []
        
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Binary outlier indicator
            feature_name = f"{column}_is_outlier_iqr"
            self.df[feature_name] = ((data < lower_bound) | (data > upper_bound)).astype(int)
            created_features.append(feature_name)
            
            # Distance from bounds
            feature_name = f"{column}_outlier_distance"
            self.df[feature_name] = np.where(
                data < lower_bound, lower_bound - data,
                np.where(data > upper_bound, data - upper_bound, 0)
            )
            created_features.append(feature_name)
        
        elif method == "zscore":
            mean_val = data.mean()
            std_val = data.std()
            z_scores = np.abs((data - mean_val) / std_val)
            
            feature_name = f"{column}_is_outlier_zscore"
            self.df[feature_name] = (z_scores > 3).astype(int)
            created_features.append(feature_name)
            
            feature_name = f"{column}_zscore"
            self.df[feature_name] = z_scores
            created_features.append(feature_name)
        
        for feature in created_features:
            if feature not in self.new_features:
                self.new_features.append(feature)
        
        return created_features

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