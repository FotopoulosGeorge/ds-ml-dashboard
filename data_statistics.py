# data_statistics.py - Independent statistics module
import streamlit as st
import pandas as pd
import numpy as np

class DataStatistics:
    """
    Independent statistics class that generates insights from session state data
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
            st.error("No data available for statistics")
            return pd.DataFrame()
    
    def get_column_info(self, df):
        """Get column type information"""
        if df.empty:
            return {'numeric': [], 'categorical': [], 'datetime': [], 'all': []}
        
        return {
            'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'all': df.columns.tolist()
        }
    
    def render_statistics_tab(self):
        """
        Render the complete statistics tab
        """
        st.header("📈 **Data Statistics & Insights**")
        
        # Get current data
        current_data = self.get_current_data()
        
        if current_data.empty:
            st.warning("⚠️ No data available for statistics")
            return
        
        st.markdown(f"*Analyzing **{len(current_data):,} rows** and **{len(current_data.columns)}** columns*")
        
        # Get column information
        columns = self.get_column_info(current_data)
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "**Select Analysis Type:**",
            [
                "📊 Descriptive Statistics",
                "👥 Group Analysis", 
                "❓ Missing Data Report",
                "🔢 Data Types Summary",
                "📈 Distribution Analysis"
            ],
            key="stats_analysis_type"
        )
        
        st.markdown("---")
        
        try:
            if analysis_type == "📊 Descriptive Statistics":
                self._descriptive_statistics(current_data, columns)
            elif analysis_type == "👥 Group Analysis":
                self._group_analysis(current_data, columns)
            elif analysis_type == "❓ Missing Data Report":
                self._missing_data_report(current_data)
            elif analysis_type == "🔢 Data Types Summary":
                self._data_types_summary(current_data)
            elif analysis_type == "📈 Distribution Analysis":
                self._distribution_analysis(current_data, columns)
                
        except Exception as e:
            st.error(f"❌ Error generating statistics: {str(e)}")
    
    def _descriptive_statistics(self, df, columns):
        """Generate descriptive statistics"""
        st.subheader("📊 Descriptive Statistics")
        
        if not columns['numeric']:
            st.warning("No numeric columns available for descriptive statistics")
            return
        
        # Overall statistics
        desc_stats = df[columns['numeric']].describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Additional insights
        st.subheader("🔍 Key Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("**📈 Highest Values:**")
            for col in columns['numeric'][:3]:
                max_val = df[col].max()
                st.metric(f"Max {col}", f"{max_val:,.2f}")
        
        with insights_col2:
            st.markdown("**📉 Lowest Values:**")
            for col in columns['numeric'][:3]:
                min_val = df[col].min()
                st.metric(f"Min {col}", f"{min_val:,.2f}")
    
    def _group_analysis(self, df, columns):
        """Generate group-by analysis"""
        st.subheader("👥 Group Analysis")
        
        if not columns['categorical']:
            st.warning("No categorical columns available for grouping")
            return
        
        if not columns['numeric']:
            st.warning("No numeric columns available for analysis")
            return
        
        group_col1, group_col2, group_col3 = st.columns(3)
        
        with group_col1:
            group_by_col = st.selectbox(
                "**Group by column:**", 
                columns['categorical'],
                key="group_by_col"
            )
        
        with group_col2:
            metric_col = st.selectbox(
                "**Analyze column:**",
                columns['numeric'],
                key="metric_col"
            )
        
        with group_col3:
            agg_func = st.selectbox(
                "**Aggregation:**",
                ["mean", "sum", "count", "median", "std", "min", "max"],
                key="agg_func"
            )
        
        if group_by_col and metric_col:
            try:
                if agg_func == "count":
                    grouped_stats = df.groupby(group_by_col)[metric_col].count().reset_index()
                    grouped_stats.columns = [group_by_col, f"{agg_func}_{metric_col}"]
                else:
                    grouped_stats = df.groupby(group_by_col)[metric_col].agg(agg_func).reset_index()
                    grouped_stats.columns = [group_by_col, f"{agg_func}_{metric_col}"]
                
                st.dataframe(grouped_stats, use_container_width=True)
                
                # Summary
                st.caption(f"📊 Showing {agg_func} of {metric_col} grouped by {group_by_col}")
                
            except Exception as e:
                st.error(f"Error in group analysis: {str(e)}")
    
    def _missing_data_report(self, df):
        """Generate missing data report"""
        st.subheader("❓ Missing Data Report")
        
        # Calculate missing data
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count()
        })
        
        # Filter to show only columns with missing data
        missing_data_filtered = missing_data[missing_data['Missing Count'] > 0]
        
        if len(missing_data_filtered) > 0:
            st.dataframe(missing_data_filtered, use_container_width=True)
            
            # Summary metrics
            total_missing = missing_data['Missing Count'].sum()
            total_cells = len(df) * len(df.columns)
            missing_pct = (total_missing / total_cells) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Missing Values", f"{total_missing:,}")
            with col2:
                st.metric("Columns with Missing", len(missing_data_filtered))
            with col3:
                st.metric("Overall Missing %", f"{missing_pct:.2f}%")
        else:
            st.success("🎉 No missing data found!")
            st.dataframe(missing_data[['Column', 'Data Type', 'Non-Null Count']], use_container_width=True)
    
    def _data_types_summary(self, df):
        """Generate data types summary"""
        st.subheader("🔢 Data Types Summary")
        
        # Create summary
        type_summary = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else 'N/A' for col in df.columns]
        })
        
        st.dataframe(type_summary, use_container_width=True)
        
        # Type distribution
        type_counts = df.dtypes.value_counts()
        
        st.subheader("📊 Type Distribution")
        type_col1, type_col2 = st.columns(2)
        
        with type_col1:
            for dtype, count in type_counts.items():
                st.metric(f"{dtype}", count)
        
        with type_col2:
            st.markdown("**🔍 Quick Facts:**")
            st.write(f"• **{len(df.select_dtypes(include=[np.number]).columns)}** numeric columns")
            st.write(f"• **{len(df.select_dtypes(include=['object', 'category']).columns)}** text columns")
            st.write(f"• **{len(df.select_dtypes(include=['datetime64']).columns)}** date columns")
    
    def _distribution_analysis(self, df, columns):
        """Generate distribution analysis"""
        st.subheader("📈 Distribution Analysis")
        
        if not columns['numeric']:
            st.warning("No numeric columns available for distribution analysis")
            return
        
        # Select column for analysis
        selected_col = st.selectbox(
            "**Select column for distribution analysis:**",
            columns['numeric'],
            key="dist_col"
        )
        
        if selected_col:
            col_data = df[selected_col].dropna()
            
            if len(col_data) == 0:
                st.warning(f"No data available for {selected_col}")
                return
            
            # Distribution metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Mean", f"{col_data.mean():.2f}")
            with metric_col2:
                st.metric("Median", f"{col_data.median():.2f}")
            with metric_col3:
                st.metric("Std Dev", f"{col_data.std():.2f}")
            with metric_col4:
                st.metric("Skewness", f"{col_data.skew():.2f}")
            
            # Percentiles
            st.subheader("📊 Percentile Analysis")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            perc_data = []
            
            for p in percentiles:
                perc_data.append({
                    'Percentile': f"{p}th",
                    'Value': col_data.quantile(p/100)
                })
            
            perc_df = pd.DataFrame(perc_data)
            st.dataframe(perc_df, use_container_width=True)