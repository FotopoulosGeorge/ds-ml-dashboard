# data_visualizer.py - Independent visualization module
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

class DataVisualizer:
    """
    Independent visualization class that creates charts from session state data
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
            st.error("No data available for visualization")
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
    
    def render_visualization_tab(self):
        """
        Render the complete visualization tab
        """
        st.header("📊 **Data Visualization**")
        
        # Get current data
        current_data = self.get_current_data()
        
        if current_data.empty:
            st.warning("⚠️ No data available for visualization")
            return
        
        st.markdown(f"*Creating charts from **{len(current_data):,} rows** of data*")
        
        # Get column information
        columns = self.get_column_info(current_data)
        
        if not columns['all']:
            st.warning("⚠️ No columns available for visualization")
            return
        
        # Chart type selection
        chart_col1, chart_col2 = st.columns([1, 2])
        
        with chart_col1:
            chart_type = st.selectbox(
                "**Select Chart Type:**",
                [
                    "📊 Bar Chart",
                    "📈 Line Chart", 
                    "🔍 Scatter Plot",
                    "📊 Histogram",
                    "📦 Box Plot",
                    "🔥 Heatmap",
                    "⏰ Time Series"
                ],
                key="viz_chart_type"
            )
        
        with chart_col2:
            st.metric("Data Points", len(current_data), help="Number of rows being visualized")
        
        st.markdown("---")
        
        # Create chart based on selection
        fig = None
        
        try:
            if chart_type == "📊 Bar Chart":
                fig = self._create_bar_chart(current_data, columns)
            elif chart_type == "📈 Line Chart":
                fig = self._create_line_chart(current_data, columns)
            elif chart_type == "🔍 Scatter Plot":
                fig = self._create_scatter_plot(current_data, columns)
            elif chart_type == "📊 Histogram":
                fig = self._create_histogram(current_data, columns)
            elif chart_type == "📦 Box Plot":
                fig = self._create_box_plot(current_data, columns)
            elif chart_type == "🔥 Heatmap":
                fig = self._create_heatmap(current_data, columns)
            elif chart_type == "⏰ Time Series":
                fig = self._create_time_series(current_data, columns)
            
            # Display the chart
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart info
                st.caption(f"📊 Chart displays {len(current_data):,} data points")
            
        except Exception as e:
            st.error(f"❌ Error creating chart: {str(e)}")
    
    def _create_bar_chart(self, df, columns):
        """Create bar chart"""
        if not columns['all']:
            st.warning("No columns available for bar chart")
            return None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("**X-axis:**", columns['all'], key="bar_x")
        with col2:
            y_axis = st.selectbox("**Y-axis:**", columns['numeric'] or columns['all'], key="bar_y")
        with col3:
            if (x_axis in columns['categorical'] and y_axis in columns['numeric']):
                aggregation = st.selectbox(
                    "**Aggregation:**", 
                    ["mean", "sum", "count", "median", "min", "max"],
                    index=0,  # Default to mean
                    key="bar_agg",
                    help="How to combine multiple values for each category"
                )
            else:
                aggregation = None


        if x_axis and y_axis and x_axis in df.columns and y_axis in df.columns:
            if (x_axis in columns['categorical'] and y_axis in columns['numeric'] and aggregation):
                if aggregation == "count":
                        agg_data = df.groupby(x_axis).size().reset_index(name='count')
                        y_plot = 'count'
                        title_suffix = f"(Count by {x_axis})"
                else:
                    agg_data = df.groupby(x_axis)[y_axis].agg(aggregation).reset_index()
                    y_plot = y_axis
                    title_suffix = f"({aggregation.title()} {y_axis} by {x_axis})"


                fig = px.bar(
                    agg_data, 
                    x=x_axis, 
                    y=y_plot,
                    title=f"{title_suffix} (n={len(df):,} records)"
                )

                avg_value = agg_data[y_plot].mean() if y_plot in agg_data.columns else 0
                fig.add_annotation(
                    text=f"Avg: {avg_value:.2f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                    )
            else:
                fig = px.bar(
                    df, 
                    x=x_axis, 
                    y=y_axis,
                    title=f"{y_axis} by {x_axis} (n={len(df):,})"
                )
            return fig
        
        return None
    
    def _create_line_chart(self, df, columns):
        """Create line chart"""
        if not columns['all']:
            st.warning("No columns available for line chart")
            return None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("**X-axis:**", columns['all'], key="line_x")
        with col2:
            y_axis = st.selectbox("**Y-axis:**", columns['numeric'] or columns['all'], key="line_y")
        with col3:
            if (x_axis in columns['categorical'] and y_axis in columns['numeric']):
                aggregation = st.selectbox(
                    "**Aggregation:**", 
                    ["mean", "sum", "median", "min", "max"],
                    index=0,
                    key="line_agg"
                )
            else:
                aggregation = None
        
        if x_axis and y_axis and x_axis in df.columns and y_axis in df.columns:
            if (x_axis in columns['categorical'] and y_axis in columns['numeric'] and aggregation):
                agg_data = df.groupby(x_axis)[y_axis].agg(aggregation).reset_index()
                title_suffix = f"({aggregation.title()} {y_axis} by {x_axis})"
                fig = px.line(
                    agg_data, 
                    x=x_axis, 
                    y=y_axis,
                    title=f"{title_suffix} (n={len(df):,} records)"
                )
            else:
                fig = px.line(
                    df, 
                    x=x_axis, 
                    y=y_axis,
                    title=f"{y_axis} over {x_axis} (n={len(df):,})"
                )
            return fig
        
        return None
    
    def _create_scatter_plot(self, df, columns):
        """Create scatter plot"""
        if len(columns['numeric']) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot")
            return None
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("**X-axis:**", columns['numeric'], key="scatter_x")
        with col2:
            y_axis = st.selectbox("**Y-axis:**", columns['numeric'], key="scatter_y")
        
        # Optional color and size
        color_col1, color_col2 = st.columns(2)
        with color_col1:
            color_by = st.selectbox("**Color by:**", ["None"] + columns['categorical'] + columns['numeric'][:3], key="scatter_color")
        with color_col2:
            size_by = st.selectbox("**Size by:**", ["None"] + columns['numeric'][:3], key="scatter_size")
        
        if x_axis and y_axis and x_axis in df.columns and y_axis in df.columns:
            color_col = None if color_by == "None" else color_by
            size_col = None if size_by == "None" else size_by
            
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_col if color_col in df.columns else None,
                size=size_col if size_col in df.columns else None,
                title=f"{y_axis} vs {x_axis} (n={len(df):,})",
                hover_data=columns['numeric'][:2] if len(columns['numeric']) >= 2 else None
            )
            return fig
        
        return None
    
    def _create_histogram(self, df, columns):
        """Create histogram"""
        if not columns['numeric']:
            st.warning("No numeric columns available for histogram")
            return None
        
        col1, col2 = st.columns(2)
        
        with col1:
            column = st.selectbox("**Column:**", columns['numeric'], key="hist_col")
        with col2:
            bins = st.slider("**Number of bins:**", 5, 100, 30, key="hist_bins")
        
        if column and column in df.columns:
            fig = px.histogram(
                df,
                x=column,
                nbins=bins,
                title=f"Distribution of {column} (n={len(df):,})"
            )
            return fig
        
        return None
    
    def _create_box_plot(self, df, columns):
        """Create box plot"""
        if not columns['numeric']:
            st.warning("No numeric columns available for box plot")
            return None
        
        col1, col2 = st.columns(2)
        
        with col1:
            y_axis = st.selectbox("**Y-axis (values):**", columns['numeric'], key="box_y")
        with col2:
            x_axis = st.selectbox("**Group by:**", ["None"] + columns['categorical'], key="box_x")
        
        if y_axis and y_axis in df.columns:
            x_col = None if x_axis == "None" else x_axis
            x_col = x_col if x_col in df.columns else None
            
            fig = px.box(
                df,
                x=x_col,
                y=y_axis,
                title=f"Distribution of {y_axis} (n={len(df):,})"
            )
            return fig
        
        return None
    
    def _create_heatmap(self, df, columns):
        """Create correlation heatmap"""
        if len(columns['numeric']) < 2:
            st.warning("Need at least 2 numeric columns for correlation heatmap")
            return None
        
        # Select columns for heatmap
        selected_cols = st.multiselect(
            "**Select numeric columns for correlation:**",
            columns['numeric'],
            default=columns['numeric'][:5],  # Default to first 5 numeric columns
            key="heatmap_cols"
        )
        
        if len(selected_cols) >= 2:
            corr_matrix = df[selected_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title=f"Correlation Heatmap (n={len(df):,})",
                color_continuous_scale="RdBu",
                aspect="auto",
                text_auto=True
            )
            
            fig.update_layout(
                width=700,
                height=500
            )
            
            return fig
        
        return None
    
    def _create_time_series(self, df, columns):
        """Create time series chart"""
        if not columns['datetime']:
            st.warning("No datetime columns found for time series")
            return None
        
        if not columns['numeric']:
            st.warning("No numeric columns available for time series values")
            return None
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("**Date column:**", columns['datetime'], key="ts_date")
        with col2:
            value_col = st.selectbox("**Value column:**", columns['numeric'], key="ts_value")
        
        if date_col and value_col and date_col in df.columns and value_col in df.columns:
            # Sort by date
            ts_df = df.sort_values(date_col)
            
            fig = px.line(
                ts_df,
                x=date_col,
                y=value_col,
                title=f"{value_col} over time (n={len(df):,})"
            )
            
            return fig
        
        return None