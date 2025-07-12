# data_preview.py - Independent preview module
import streamlit as st
import pandas as pd
import numpy as np

class DataPreview:
    """
    Independent preview class that shows data from session state
    """
    
    def __init__(self):
        pass
    
    def get_current_data(self):
        """Get current working data from session state"""
        if 'working_df' in st.session_state and st.session_state.working_df is not None:
            return st.session_state.working_df.copy()
        elif 'base_df' in st.session_state and st.session_state.base_df is not None:
            return st.session_state.base_df.copy()
        else:
            st.error("No data available for preview")
            return pd.DataFrame()
    
    def render_preview_tab(self):
        """
        Render the complete data preview tab
        """
        st.header("🧹 **Data Preview & Quality**")
        
        # Get current data
        current_data = self.get_current_data()
        
        if current_data.empty:
            st.warning("⚠️ No data available for preview")
            return
        
        st.markdown(f"*Previewing **{len(current_data):,} rows** and **{len(current_data.columns)}** columns*")
        
        # Preview options
        preview_type = st.selectbox(
            "**Select Preview Type:**",
            [
                "🔍 Data Sample",
                "📊 Column Summary",
                "🧹 Data Quality Check",
                "🔢 Value Counts",
                "📈 Quick Insights"
            ],
            key="preview_type"
        )
        
        st.markdown("---")
        
        try:
            if preview_type == "🔍 Data Sample":
                self._data_sample(current_data)
            elif preview_type == "📊 Column Summary":
                self._column_summary(current_data)
            elif preview_type == "🧹 Data Quality Check":
                self._data_quality_check(current_data)
            elif preview_type == "🔢 Value Counts":
                self._value_counts(current_data)
            elif preview_type == "📈 Quick Insights":
                self._quick_insights(current_data)
                
        except Exception as e:
            st.error(f"❌ Error generating preview: {str(e)}")
    
    def _data_sample(self, df):
        """Show data sample with various options"""
        st.subheader("🔍 Data Sample")
        
        # Sample options
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            sample_type = st.selectbox(
                "Sample type:",
                ["First rows", "Last rows", "Random sample"],
                key="sample_type"
            )
        
        with sample_col2:
            n_rows = st.selectbox(
                "Number of rows:",
                [5, 10, 20, 50, 100],
                index=1,
                key="sample_rows"
            )
        
        with sample_col3:
            show_info = st.checkbox("Show column info", value=True, key="show_info")
        
        # Generate sample
        if sample_type == "First rows":
            sample_df = df.head(n_rows)
        elif sample_type == "Last rows":
            sample_df = df.tail(n_rows)
        else:  # Random sample
            n_rows = min(n_rows, len(df))
            sample_df = df.sample(n_rows, random_state=42)
        
        # Display sample
        st.dataframe(sample_df, use_container_width=True)
        st.caption(f"Showing {len(sample_df)} of {len(df):,} total rows")
        
        # Column info
        if show_info:
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
    
    def _column_summary(self, df):
        """Detailed column summary"""
        st.subheader("📊 Column Summary")
        
        # Column selection
        selected_columns = st.multiselect(
            "Select columns to analyze:",
            df.columns.tolist(),
            default=df.columns.tolist()[:5],  # Default to first 5 columns
            key="column_summary_select"
        )
        
        if not selected_columns:
            st.warning("Please select at least one column")
            return
        
        for col in selected_columns:
            with st.expander(f"📊 **{col}** ({df[col].dtype})"):
                col_data = df[col]
                
                # Basic info
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                
                with info_col1:
                    st.metric("Total Values", len(col_data))
                with info_col2:
                    st.metric("Non-Null", col_data.count())
                with info_col3:
                    st.metric("Null Values", col_data.isnull().sum())
                with info_col4:
                    st.metric("Unique Values", col_data.nunique())
                
                # Type-specific analysis
                if pd.api.types.is_numeric_dtype(col_data):
                    st.markdown("**📈 Numeric Analysis:**")
                    numeric_col1, numeric_col2 = st.columns(2)
                    
                    with numeric_col1:
                        st.write(f"• **Mean:** {col_data.mean():.2f}")
                        st.write(f"• **Median:** {col_data.median():.2f}")
                        st.write(f"• **Min:** {col_data.min():.2f}")
                    
                    with numeric_col2:
                        st.write(f"• **Max:** {col_data.max():.2f}")
                        st.write(f"• **Std:** {col_data.std():.2f}")
                        st.write(f"• **Range:** {col_data.max() - col_data.min():.2f}")
                
                elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                    st.markdown("**📝 Text Analysis:**")
                    
                    # Most common values
                    top_values = col_data.value_counts().head(5)
                    if len(top_values) > 0:
                        st.write("**Top 5 values:**")
                        for val, count in top_values.items():
                            pct = (count / len(col_data)) * 100
                            st.write(f"• **{val}:** {count} ({pct:.1f}%)")
                
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    st.markdown("**📅 Date Analysis:**")
                    date_col1, date_col2 = st.columns(2)
                    
                    with date_col1:
                        st.write(f"• **Earliest:** {col_data.min()}")
                        st.write(f"• **Latest:** {col_data.max()}")
                    
                    with date_col2:
                        date_range = col_data.max() - col_data.min()
                        st.write(f"• **Range:** {date_range.days} days")
    
    def _data_quality_check(self, df):
        """Comprehensive data quality check"""
        st.subheader("🧹 Data Quality Check")
        
        # Overall quality score
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        quality_score = max(0, 100 - ((missing_cells + duplicate_rows) / total_cells * 100))
        
        # Quality metrics
        quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
        
        with quality_col1:
            st.metric("Quality Score", f"{quality_score:.1f}%", 
                     delta=None, delta_color="normal")
        with quality_col2:
            st.metric("Missing Values", f"{missing_cells:,}", 
                     delta=None, delta_color="inverse")
        with quality_col3:
            st.metric("Duplicate Rows", f"{duplicate_rows:,}", 
                     delta=None, delta_color="inverse")
        with quality_col4:
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            st.metric("Completeness", f"{completeness:.1f}%")
        
        # Detailed quality issues
        st.markdown("---")
        st.subheader("🔍 Quality Issues by Column")
        
        quality_issues = []
        
        for col in df.columns:
            col_data = df[col]
            issues = []
            
            # Missing values
            missing_count = col_data.isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                issues.append(f"{missing_count} missing ({missing_pct:.1f}%)")
            
            # Duplicate values (for categorical)
            if col_data.dtype in ['object', 'category']:
                if col_data.nunique() < len(col_data) * 0.5:  # High duplication
                    dup_pct = (1 - col_data.nunique() / len(col_data)) * 100
                    issues.append(f"High duplication ({dup_pct:.1f}%)")
            
            # Outliers (for numeric)
            if pd.api.types.is_numeric_dtype(col_data):
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    outlier_pct = (outliers / len(df)) * 100
                    issues.append(f"{outliers} outliers ({outlier_pct:.1f}%)")
            
            quality_issues.append({
                'Column': col,
                'Data Type': str(col_data.dtype),
                'Issues': '; '.join(issues) if issues else 'No issues',
                'Issue Count': len(issues)
            })
        
        quality_df = pd.DataFrame(quality_issues)
        st.dataframe(quality_df, use_container_width=True)
        
        # Quick fixes suggestions
        if missing_cells > 0 or duplicate_rows > 0:
            st.subheader("💡 Suggested Actions")
            
            if missing_cells > 0:
                st.write("🔧 **Missing Values:**")
                st.write("• Consider removing rows/columns with high missing percentages")
                st.write("• Fill missing values with mean/median (numeric) or mode (categorical)")
            
            if duplicate_rows > 0:
                st.write("🔧 **Duplicate Rows:**")
                st.write("• Consider removing duplicate rows to improve data quality")
                st.write("• Verify if duplicates are legitimate or data entry errors")
    
    def _value_counts(self, df):
        """Show value counts for categorical columns"""
        st.subheader("🔢 Value Counts")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            st.warning("No categorical columns available for value counts")
            return
        
        selected_col = st.selectbox(
            "Select column for value counts:",
            categorical_cols,
            key="value_counts_col"
        )
        
        if selected_col:
            col_data = df[selected_col]
            
            # Options
            options_col1, options_col2 = st.columns(2)
            
            with options_col1:
                top_n = st.slider("Show top N values:", 5, 50, 10, key="value_counts_n")
            with options_col2:
                show_percentage = st.checkbox("Show percentages", value=True, key="show_percentage")
            
            # Calculate value counts
            value_counts = col_data.value_counts().head(top_n)
            
            if show_percentage:
                percentages = (value_counts / len(col_data) * 100).round(2)
                value_counts_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': percentages.values
                })
            else:
                value_counts_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values
                })
            
            st.dataframe(value_counts_df, use_container_width=True)
            
            # Summary
            total_unique = col_data.nunique()
            st.caption(f"Showing top {len(value_counts)} of {total_unique} unique values")
    
    def _quick_insights(self, df):
        """Generate quick insights about the data"""
        st.subheader("📈 Quick Insights")
        
        insights = []
        
        # Data size insights
        if len(df) > 100000:
            insights.append("📊 **Large Dataset:** This is a substantial dataset with over 100K rows")
        elif len(df) < 100:
            insights.append("📊 **Small Dataset:** This is a small dataset with fewer than 100 rows")
        
        # Missing data insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            insights.append(f"❓ **High Missing Data:** {missing_pct:.1f}% of data is missing")
        elif missing_pct == 0:
            insights.append("✅ **Complete Data:** No missing values found")
        
        # Numeric columns insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > len(df.columns) * 0.8:
            insights.append("🔢 **Numeric Heavy:** Most columns are numeric - good for mathematical analysis")
        
        # Categorical insights
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                insights.append(f"🏷️ **High Cardinality:** Column '{col}' has many unique values")
            elif unique_ratio < 0.1:
                insights.append(f"🏷️ **Low Cardinality:** Column '{col}' has few unique values")
        
        # Duplicate insights
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            insights.append(f"🔄 **Duplicates Found:** {duplicate_count} duplicate rows detected")
        
        # Date columns insights
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            insights.append(f"📅 **Time Series Potential:** {len(date_cols)} date column(s) found")
        
        # Display insights
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("No specific insights detected. Your data looks standard!")
        
        # Data overview table
        st.subheader("📋 Data Overview")
        
        overview_data = {
            'Metric': [
                'Total Rows',
                'Total Columns', 
                'Numeric Columns',
                'Text Columns',
                'Date Columns',
                'Missing Values',
                'Duplicate Rows',
                'Memory Usage (MB)'
            ],
            'Value': [
                f"{len(df):,}",
                len(df.columns),
                len(numeric_cols),
                len(categorical_cols),
                len(date_cols),
                f"{df.isnull().sum().sum():,}",
                f"{duplicate_count:,}",
                f"{df.memory_usage(deep=True).sum() / (1024**2):.2f}"
            ]
        }
        
        overview_df = pd.DataFrame(overview_data)
        st.dataframe(overview_df, use_container_width=True, hide_index=True)