# data_filter.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date

class DataFilter:
    """
    Independent filtering class that handles all data filtering logic
    Returns filtered dataframes without scope issues
    """
    
    def __init__(self, df):
        """Initialize with original dataframe"""
        self.original_df = df.copy()
        self.filtered_df = df.copy()
        self.filter_info = {
            'active': False,
            'type': None,
            'original_count': len(df),
            'filtered_count': len(df),
            'filters_applied': []
        }
    
    def get_column_info(self):
        """Get column type information"""
        return {
            'numeric': self.original_df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': self.original_df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime': self.original_df.select_dtypes(include=['datetime64']).columns.tolist(),
            'all': self.original_df.columns.tolist()
        }
    
    def render_filter_ui(self):
        """
        Render the filtering UI and return filtered dataframe
        Returns: (filtered_df, filter_applied)
        """
        st.markdown("*Filter data using simple controls or advanced queries*")
        
        # Filter type selection
        filter_type = st.radio(
            "Choose filter mode:",
            ["🎛️ Simple Filters", "⚡ Query Builder"],
            horizontal=True
        )
        
        filtered_df = self.original_df.copy()
        filters_applied = False
        applied_filter_details = []
        
        if filter_type == "🎛️ Simple Filters":
            filtered_df, filters_applied, applied_filter_details = self._simple_filters()
        else:
            filtered_df, filters_applied, applied_filter_details = self._query_builder()
        
        # Update internal state
        self.filtered_df = filtered_df
        self.filter_info = {
            'active': filters_applied,
            'type': filter_type,
            'original_count': len(self.original_df),
            'filtered_count': len(filtered_df),
            'filters_applied': applied_filter_details
        }
        
        # Show filter results
        if filters_applied:
            reduction_pct = ((len(self.original_df) - len(filtered_df)) / len(self.original_df) * 100)
            st.info(f"🔍 **Filtered:** {len(filtered_df):,} rows ({reduction_pct:.1f}% reduction)")
        
        return filtered_df, filters_applied
    
    def _simple_filters(self):
        """Handle simple filtering UI"""
        column_info = self.get_column_info()
        
        filter_columns = st.multiselect(
            "Select columns to filter:",
            options=self.original_df.columns.tolist(),
            help="Choose columns to create filters for"
        )
        
        if not filter_columns:
            return self.original_df.copy(), False, []
        
        filtered_df = self.original_df.copy()
        filters_applied = False
        applied_filter_details = []
        
        filter_cols = st.columns(min(2, len(filter_columns)))
        
        for i, col in enumerate(filter_columns):
            with filter_cols[i % 2]:
                st.write(f"**{col}** *({self.original_df[col].dtype})*")
                
                if self.original_df[col].dtype in ['object', 'category']:
                    filtered_df, col_filtered, filter_detail = self._handle_categorical_filter(filtered_df, col)
                elif self.original_df[col].dtype in ['int64', 'float64']:
                    filtered_df, col_filtered, filter_detail = self._handle_numeric_filter(filtered_df, col)
                elif 'datetime' in str(self.original_df[col].dtype):
                    filtered_df, col_filtered, filter_detail = self._handle_datetime_filter(filtered_df, col)
                
                if col_filtered:
                    filters_applied = True
                    applied_filter_details.append(filter_detail)
        
        return filtered_df, filters_applied, applied_filter_details
    
    def _handle_categorical_filter(self, df, col):
        """Handle categorical column filtering"""
        unique_values = self.original_df[col].unique()
        
        if len(unique_values) <= 50:
            selected_values = st.multiselect(
                "Select values:",
                options=unique_values,
                default=unique_values,
                key=f"filter_{col}"
            )
            if len(selected_values) < len(unique_values):
                df = df[df[col].isin(selected_values)]
                return df, True, f"{col}: {len(selected_values)}/{len(unique_values)} values"
        else:
            search_term = st.text_input(f"Search in {col}:", key=f"search_{col}")
            if search_term:
                df = df[df[col].str.contains(search_term, case=False, na=False)]
                return df, True, f"{col}: contains '{search_term}'"
        
        return df, False, ""
    
    def _handle_numeric_filter(self, df, col):
        """Handle numeric column filtering"""
        min_val, max_val = float(self.original_df[col].min()), float(self.original_df[col].max())
        selected_range = st.slider(
            "Value range:",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            key=f"range_{col}"
        )
        if selected_range != (min_val, max_val):
            df = df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]
            return df, True, f"{col}: {selected_range[0]} to {selected_range[1]}"
        
        return df, False, ""
    
    def _handle_datetime_filter(self, df, col):
        """Handle datetime column filtering"""
        min_date = self.original_df[col].min().date()
        max_date = self.original_df[col].max().date()
        
        date_range = st.date_input(
            "Date range:",
            value=(min_date, max_date),
            key=f"date_{col}"
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            if start_date != min_date or end_date != max_date:
                df = df[(df[col].dt.date >= start_date) & (df[col].dt.date <= end_date)]
                return df, True, f"{col}: {start_date} to {end_date}"
        
        return df, False, ""
    
    def _query_builder(self):
        """Handle query builder filtering"""
        query_text = st.text_area(
            "Write a pandas query:",
            help="Example: sales > 1000 and category.str.contains('tech')",
            placeholder="column_name > 100 and other_column == 'value'",
            height=100
        )
        
        if query_text:
            try:
                filtered_df = self.original_df.query(query_text)
                st.success(f"✅ Query applied: {len(filtered_df):,} rows returned")
                return filtered_df, True, [f"Query: {query_text}"]
            except Exception as e:
                st.error(f"Query error: {str(e)}")
                return self.original_df.copy(), False, []
        
        # Query examples
        with st.expander("💡 Query Examples"):
            st.code("""
# Numeric conditions
sales > 1000 and quantity >= 5
price.between(10, 100)

# Text patterns  
category.str.contains('electronics', case=False)
name.str.startswith('A')

# Date conditions
date >= '2023-01-01'
date.dt.year == 2023

# Combined conditions
(revenue > 10000) or (region == 'North')
            """)
        
        return self.original_df.copy(), False, []
    
    def get_filtered_data(self):
        """Get current filtered dataframe"""
        return self.filtered_df.copy()
    
    def get_filter_info(self):
        """Get current filter information"""
        return self.filter_info.copy()
    
    def clear_filters(self):
        """Clear all filters and return original data"""
        self.filtered_df = self.original_df.copy()
        self.filter_info = {
            'active': False,
            'type': None,
            'original_count': len(self.original_df),
            'filtered_count': len(self.original_df),
            'filters_applied': []
        }
        return self.filtered_df.copy()
    
    def render_filter_summary(self):
        """Render filter summary if filters are active"""
        if self.filter_info['active']:
            filter_col1, filter_col2 = st.columns([4, 1])
            
            with filter_col1:
                reduction_pct = ((self.filter_info['original_count'] - self.filter_info['filtered_count']) / self.filter_info['original_count'] * 100)
                st.markdown(f"""
                🔍 **Active Filter:** {self.filter_info['type']} • 
                Showing **{self.filter_info['filtered_count']:,}** of **{self.filter_info['original_count']:,}** rows 
                ({reduction_pct:.1f}% filtered)
                """)
            
            with filter_col2:
                if st.button("❌ Clear All Filters", key="clear_filters"):
                    return True  # Signal to clear filters
        
        return False  # No clear action