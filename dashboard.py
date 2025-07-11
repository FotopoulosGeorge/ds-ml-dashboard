import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime, date
from feature_engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="InsightStream - Advanced BI Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'joined_data' not in st.session_state:
    st.session_state.joined_data = None

# Header
st.title("📊 InsightStream")
st.markdown("**Advanced Business Intelligence Dashboard** • Upload, Join, Filter, Visualize")

# Sidebar for file upload and controls
st.sidebar.header("📁 Data Management")

# Multi-file upload
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV Files",
    type=['csv'],
    accept_multiple_files=True,
    help="Upload one or more CSV files"
)

# Load datasets
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state.datasets:
            @st.cache_data
            def load_data(file):
                df = pd.read_csv(file)
                # Auto-detect and convert date columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                        except:
                            pass
                return df
            
            st.session_state.datasets[file_name] = load_data(uploaded_file)

# Dataset management sidebar
if st.session_state.datasets:
    st.sidebar.subheader("📋 Loaded Datasets")
    
    for name, df in st.session_state.datasets.items():
        with st.sidebar.expander(f"{name}"):
            st.write(f"📏 **Shape:** {df.shape[0]:,} × {df.shape[1]}")
            st.write(f"💾 **Size:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                del st.session_state.datasets[name]
                st.rerun()

# Main content area
if st.session_state.datasets:
    
    # Data joining section (collapsible if multiple datasets)
    if len(st.session_state.datasets) > 1:
        with st.expander("🔗 **Data Joining** - Combine Multiple Datasets", expanded=False):
            st.markdown("*Join two datasets based on common columns*")
            
            join_col1, join_col2, join_col3 = st.columns(3)
            
            with join_col1:
                dataset_names = list(st.session_state.datasets.keys())
                left_dataset = st.selectbox("Left Dataset:", dataset_names, key="left_ds")
                
            with join_col2:
                right_dataset = st.selectbox(
                    "Right Dataset:", 
                    [name for name in dataset_names if name != left_dataset],
                    key="right_ds"
                )
                
            with join_col3:
                join_type = st.selectbox(
                    "Join Type:",
                    ["inner", "left", "right", "outer"],
                    help="Inner: Only matching rows | Left: All left + matches | Right: All right + matches | Outer: Everything",
                    key="join_type"
                )
            
            # Select join keys
            if left_dataset and right_dataset:
                left_df = st.session_state.datasets[left_dataset]
                right_df = st.session_state.datasets[right_dataset]
                
                join_key_col1, join_key_col2 = st.columns(2)
                
                with join_key_col1:
                    left_key = st.selectbox(
                        f"Join key from {left_dataset}:",
                        left_df.columns.tolist(),
                        key="left_key"
                    )
                    
                with join_key_col2:
                    right_key = st.selectbox(
                        f"Join key from {right_dataset}:",
                        right_df.columns.tolist(),
                        key="right_key"
                    )
                
                # Data type compatibility check
                if left_key and right_key:
                    left_dtype = str(left_df[left_key].dtype)
                    right_dtype = str(right_df[right_key].dtype)
                    
                    compatibility_col1, compatibility_col2 = st.columns(2)
                    
                    with compatibility_col1:
                        if left_dtype == right_dtype:
                            st.success(f"✅ Compatible types: `{left_dtype}`")
                            compatibility = True
                        else:
                            st.warning(f"⚠️ Type mismatch: `{left_dtype}` vs `{right_dtype}`")
                            compatibility = False
                    
                    with compatibility_col2:
                        auto_fix = st.checkbox(
                            "🔧 Auto-fix types", 
                            value=not compatibility,
                            help="Automatically convert data types to enable joining"
                        )
                    
                    # Join button
                    if st.button("🔗 **Join Datasets**", type="primary"):
                        try:
                            left_prep = left_df.copy()
                            right_prep = right_df.copy()
                            
                            # Auto-fix data types if requested
                            if auto_fix and not compatibility:
                                try:
                                    if 'object' in [left_dtype, right_dtype]:
                                        left_prep[left_key] = left_prep[left_key].astype(str)
                                        right_prep[right_key] = right_prep[right_key].astype(str)
                                    elif 'int' in left_dtype and 'float' in right_dtype:
                                        left_prep[left_key] = left_prep[left_key].astype(float)
                                    elif 'float' in left_dtype and 'int' in right_dtype:
                                        right_prep[right_key] = right_prep[right_key].astype(float)
                                except Exception as conv_error:
                                    st.warning(f"Type conversion failed: {conv_error}")
                            
                            # Perform the join
                            joined_df = pd.merge(
                                left_prep, 
                                right_prep, 
                                left_on=left_key, 
                                right_on=right_key, 
                                how=join_type,
                                suffixes=('_left', '_right')
                            )
                            
                            st.session_state.joined_data = joined_df
                            
                            # Success metrics
                            success_col1, success_col2, success_col3, success_col4 = st.columns(4)
                            with success_col1:
                                st.metric("✅ Result Rows", f"{len(joined_df):,}")
                            with success_col2:
                                st.metric("📊 Columns", len(joined_df.columns))
                            with success_col3:
                                efficiency = len(joined_df) / max(len(left_df), len(right_df)) * 100
                                st.metric("🎯 Efficiency", f"{efficiency:.1f}%")
                            with success_col4:
                                st.metric("🔗 Join Type", join_type.title())
                            
                        except Exception as e:
                            st.error(f"❌ Join failed: {str(e)}")
                            if "dtype" in str(e).lower():
                                st.info("💡 **Try**: Enable 'Auto-fix types' option")
    
    # Select working dataset
    if st.session_state.joined_data is not None:
        dataset_options = ["🔗 Joined Data"] + [f"📄 {name}" for name in st.session_state.datasets.keys()]
        selected_option = st.selectbox("**Select dataset to analyze:**", dataset_options)
        
        if selected_option.startswith("🔗"):
            df = st.session_state.joined_data
            dataset_name = "Joined Data"
        else:
            dataset_name = selected_option.replace("📄 ", "")
            df = st.session_state.datasets[dataset_name]
    else:
        if st.session_state.datasets:
            dataset_options = [f"📄 {name}" for name in st.session_state.datasets.keys()]
            selected_option = st.selectbox("**Select dataset to analyze:**", dataset_options)
            dataset_name = selected_option.replace("📄 ", "")
            df = st.session_state.datasets[dataset_name]
        else:
            df = None

    if df is not None:
        
        # Quick metrics (always visible)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("📊 **Total Rows**", f"{len(df):,}")
        with metric_col2:
            st.metric("📋 **Columns**", len(df.columns))
        with metric_col3:
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("🔢 **Numeric**", numeric_count)
        with metric_col4:
            missing_count = df.isnull().sum().sum()
            st.metric("❓ **Missing**", f"{missing_count:,}")
        
        st.markdown("---")
        
        # Data preview (collapsible)
        with st.expander("📋 **Data Preview** - Explore Your Data", expanded=False):
            preview_col1, preview_col2 = st.columns([3, 1])
            
            with preview_col1:
                st.markdown(f"*Dataset: {dataset_name}*")
            with preview_col2:
                preview_rows = st.selectbox("Rows to show:", [5, 10, 20, 50], index=1, key="preview_rows")
            
            st.dataframe(df.head(preview_rows), use_container_width=True)
            
            # Column info
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            
            with st.expander("📊 Column Details"):
                st.dataframe(col_info, use_container_width=True)
        
        # Data cleaning (collapsible)
        with st.expander("🧹 **Data Cleaning** - Improve Data Quality", expanded=False):
            st.markdown("*Automated data quality checks and cleaning options*")
            
            # Data quality metrics
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            
            duplicates = df.duplicated().sum()
            missing_cells = df.isnull().sum().sum()
            empty_rows = df.isnull().all(axis=1).sum()
            
            with quality_col1:
                st.metric("🔄 Duplicates", duplicates, delta=None, delta_color="inverse")
            with quality_col2:
                st.metric("❓ Missing Values", missing_cells, delta=None, delta_color="inverse")
            with quality_col3:
                st.metric("📭 Empty Rows", empty_rows, delta=None, delta_color="inverse")
            with quality_col4:
                quality_score = max(0, 100 - ((duplicates + missing_cells + empty_rows) / len(df) * 100))
                st.metric("🎯 Quality Score", f"{quality_score:.1f}%")
            
            # Quick cleaning actions
            if duplicates > 0 or missing_cells > 0 or empty_rows > 0:
                st.markdown("**Quick Actions:**")
                clean_col1, clean_col2, clean_col3 = st.columns(3)
                
                with clean_col1:
                    if duplicates > 0 and st.button(f"Remove {duplicates} Duplicates"):
                        df = df.drop_duplicates()
                        st.success("Duplicates removed!")
                        st.rerun()
                
                with clean_col2:
                    if missing_cells > 0 and st.button("Smart Fill Missing"):
                        # Simple smart filling
                        for col in df.columns:
                            if df[col].dtype in ['int64', 'float64']:
                                df[col] = df[col].fillna(df[col].median())
                            elif df[col].dtype == 'object':
                                mode_val = df[col].mode()
                                if len(mode_val) > 0:
                                    df[col] = df[col].fillna(mode_val[0])
                        st.success("Missing values filled!")
                        st.rerun()
                
                with clean_col3:
                    if empty_rows > 0 and st.button(f"Remove {empty_rows} Empty Rows"):
                        df = df.dropna(how='all')
                        st.success("Empty rows removed!")
                        st.rerun()
        
        # Advanced filtering (collapsible)
        with st.expander("🔍 **Advanced Filtering** - Focus Your Analysis", expanded=False):
            st.markdown("*Filter data using simple controls or advanced queries*")
            
            # Filter type selection
            filter_type = st.radio(
                "Choose filter mode:",
                ["🎛️ Simple Filters", "⚡ Query Builder"],
                horizontal=True
            )
            
            filtered_df = df.copy()
            
            if filter_type == "🎛️ Simple Filters":
                filter_columns = st.multiselect(
                    "Select columns to filter:",
                    options=df.columns.tolist(),
                    help="Choose columns to create filters for"
                )
                
                if filter_columns:
                    filter_cols = st.columns(min(2, len(filter_columns)))
                    
                    for i, col in enumerate(filter_columns):
                        with filter_cols[i % 2]:
                            st.write(f"**{col}** *({df[col].dtype})*")
                            
                            if df[col].dtype in ['object', 'category']:
                                unique_values = df[col].unique()
                                if len(unique_values) <= 50:  # Dropdown for reasonable number of values
                                    selected_values = st.multiselect(
                                        "Select values:",
                                        options=unique_values,
                                        default=unique_values,
                                        key=f"filter_{col}"
                                    )
                                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                                else:  # Text input for many values
                                    search_term = st.text_input(f"Search in {col}:", key=f"search_{col}")
                                    if search_term:
                                        filtered_df = filtered_df[filtered_df[col].str.contains(search_term, case=False, na=False)]
                            
                            elif df[col].dtype in ['int64', 'float64']:
                                min_val, max_val = float(df[col].min()), float(df[col].max())
                                selected_range = st.slider(
                                    "Value range:",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"range_{col}"
                                )
                                filtered_df = filtered_df[
                                    (filtered_df[col] >= selected_range[0]) & 
                                    (filtered_df[col] <= selected_range[1])
                                ]
                            
                            elif 'datetime' in str(df[col].dtype):
                                date_range = st.date_input(
                                    "Date range:",
                                    value=(df[col].min().date(), df[col].max().date()),
                                    key=f"date_{col}"
                                )
                                if len(date_range) == 2:
                                    start_date, end_date = date_range
                                    filtered_df = filtered_df[
                                        (filtered_df[col].dt.date >= start_date) & 
                                        (filtered_df[col].dt.date <= end_date)
                                    ]
            
            else:  # Query Builder
                query_text = st.text_area(
                    "Write a pandas query:",
                    help="Example: sales > 1000 and category.str.contains('tech')",
                    placeholder="column_name > 100 and other_column == 'value'",
                    height=100
                )
                
                if query_text:
                    try:
                        filtered_df = df.query(query_text)
                        st.success(f"✅ Query applied: {len(filtered_df):,} rows returned")
                    except Exception as e:
                        st.error(f"Query error: {str(e)}")
                        filtered_df = df.copy()
                
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
            
            # Show filter results
            if len(filtered_df) != len(df):
                st.info(f"🔍 **Filtered:** {len(filtered_df):,} rows ({(len(filtered_df)/len(df)*100):.1f}% of data)")
            
            # Update working dataframe
            df = filtered_df
        
        # FEATURE ENGINEERING (collapsible)
        with st.expander("🔧 **Feature Engineering** - Transform Your Data for ML", expanded=False):
            st.markdown("*Create new features and transform existing ones to improve model performance*")
            
            # Initialize FE class
            if 'fe_handler' not in st.session_state:
                st.session_state.fe_handler = FeatureEngineer(df)
            
            fe_handler = st.session_state.fe_handler

            column_info = fe_handler.get_column_info()
            numeric_cols = column_info['numeric']
            categorical_cols = column_info['categorical'] 
            date_cols = column_info['datetime']
            
            # UI logic stays here, but calls methods from FeatureEngineer
            fe_type = st.selectbox("Select technique:", 
                [    
                    "🔢 Numerical Transformations",
                    "📝 Text Feature Extraction", 
                    "📅 Date/Time Features",
                    "🏷️ Categorical Encoding",
                    "⚡ Advanced Features",
                    "📊 Statistical Features"
                ]
            )
            
            if fe_type == "🔢 Numerical Transformations":
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select column:", numeric_cols)
                    transform_type = st.selectbox("Transformation:", 
                                                fe_handler.get_available_numerical_transforms())
                
                with col2:
                    # Show transform info
                    info = fe_handler.get_transform_info(transform_type)
                    st.info(f"**{transform_type}:** {info['description']}")
                    
                    # Handle special parameters
                    kwargs = {}
                    if transform_type == "Quantile Binning":
                        kwargs['n_bins'] = st.slider("Number of bins:", 2, 20, 5)
                    
                    new_name = st.text_input("New column name:", 
                                            value=f"{selected_col}_{transform_type.lower().replace(' ', '_')}")
                
                if st.button("🔧 Apply Transformation"):
                    try:
                        created_col = fe_handler.apply_numerical_transform(
                            selected_col, transform_type, new_name, **kwargs
                        )
                        st.success(f"✅ Created feature: {created_col}")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
            
            # Repeat for other FE types...
            
            # Update working dataframe
            df = fe_handler.df

        # VISUALIZATIONS (Always prominent and expanded)
        st.header("📈 **Data Visualization**")
        st.markdown("*Create interactive charts and explore your data visually*")
        
        # Chart controls
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            chart_type = st.selectbox(
                "**Chart Type:**",
                ["📊 Bar Chart", "📈 Line Chart", "🔍 Scatter Plot", "📊 Histogram", "📦 Box Plot", "🔥 Heatmap", "⏰ Time Series"]
            )
        
        # Get column types for smart defaults
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Chart configuration based on type
        fig = None
        
        if chart_type == "📊 Bar Chart":
            with viz_col2:
                x_axis = st.selectbox("**X-axis:**", all_cols, key="bar_x")
            with viz_col3:
                y_axis = st.selectbox("**Y-axis:**", numeric_cols, key="bar_y")
            
            if x_axis and y_axis:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        
        elif chart_type == "📈 Line Chart":
            with viz_col2:
                x_axis = st.selectbox("**X-axis:**", all_cols, key="line_x")
            with viz_col3:
                y_axis = st.selectbox("**Y-axis:**", numeric_cols, key="line_y")
            
            if x_axis and y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        
        elif chart_type == "🔍 Scatter Plot":
            with viz_col2:
                x_axis = st.selectbox("**X-axis:**", numeric_cols, key="scatter_x")
            with viz_col3:
                y_axis = st.selectbox("**Y-axis:**", numeric_cols, key="scatter_y")
            
            # Additional scatter options
            scatter_col1, scatter_col2 = st.columns(2)
            with scatter_col1:
                color_by = st.selectbox("**Color by:**", ["None"] + categorical_cols + numeric_cols[:3])
            with scatter_col2:
                size_by = st.selectbox("**Size by:**", ["None"] + numeric_cols[:3])
            
            if x_axis and y_axis:
                color_col = None if color_by == "None" else color_by
                size_col = None if size_by == "None" else size_by
                
                fig = px.scatter(
                    df, x=x_axis, y=y_axis, 
                    color=color_col, size=size_col,
                    title=f"{y_axis} vs {x_axis}",
                    hover_data=numeric_cols[:2]
                )
        
        elif chart_type == "📊 Histogram":
            with viz_col2:
                column = st.selectbox("**Column:**", numeric_cols, key="hist_col")
            with viz_col3:
                bins = st.slider("**Bins:**", 10, 100, 30)
            
            if column:
                fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
        
        elif chart_type == "📦 Box Plot":
            with viz_col2:
                y_axis = st.selectbox("**Y-axis:**", numeric_cols, key="box_y")
            with viz_col3:
                x_axis = st.selectbox("**Group by:**", ["None"] + categorical_cols, key="box_x")
            
            if y_axis:
                x_col = None if x_axis == "None" else x_axis
                fig = px.box(df, x=x_col, y=y_axis, title=f"Distribution of {y_axis}")
        
        elif chart_type == "🔥 Heatmap":
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    text_auto=True
                )
            else:
                st.warning("⚠️ Need at least 2 numeric columns for correlation heatmap")
        
        elif chart_type == "⏰ Time Series":
            if date_cols:
                with viz_col2:
                    date_col = st.selectbox("**Date column:**", date_cols, key="ts_date")
                with viz_col3:
                    value_col = st.selectbox("**Value column:**", numeric_cols, key="ts_value")
                
                if date_col and value_col:
                    # Sort by date
                    ts_df = df.sort_values(date_col)
                    fig = px.line(ts_df, x=date_col, y=value_col, title=f"{value_col} over time")
            else:
                st.warning("⚠️ No date columns found for time series")
        
        # Display chart
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics (collapsible)
        with st.expander("📊 **Summary Statistics** - Data Insights", expanded=False):
            if numeric_cols:
                summary_type = st.selectbox(
                    "Analysis type:",
                    ["📈 Descriptive Statistics", "👥 Group Analysis", "❓ Missing Data Report"]
                )
                
                if summary_type == "📈 Descriptive Statistics":
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                elif summary_type == "👥 Group Analysis" and categorical_cols:
                    group_col1, group_col2 = st.columns(2)
                    with group_col1:
                        group_by_col = st.selectbox("Group by:", categorical_cols)
                    with group_col2:
                        metric_col = st.selectbox("Analyze:", numeric_cols)
                    
                    if group_by_col and metric_col:
                        grouped_stats = df.groupby(group_by_col)[metric_col].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(2)
                        st.dataframe(grouped_stats, use_container_width=True)
                
                elif summary_type == "❓ Missing Data Report":
                    missing_data = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Count': df.isnull().sum(),
                        'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
                        'Data Type': df.dtypes
                    })
                    missing_data = missing_data[missing_data['Missing Count'] > 0]
                    
                    if len(missing_data) > 0:
                        st.dataframe(missing_data, use_container_width=True)
                    else:
                        st.success("🎉 No missing data found!")
        
        # Export section (collapsible)
        with st.expander("💾 **Export Data** - Download Results", expanded=False):
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 **Download Filtered Data**",
                    data=csv,
                    file_name=f"{dataset_name}_filtered.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with export_col2:
                if numeric_cols:
                    summary_csv = df[numeric_cols].describe().to_csv()
                    st.download_button(
                        label="📊 **Download Statistics**",
                        data=summary_csv,
                        file_name=f"{dataset_name}_stats.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with export_col3:
                # Export info
                export_info = pd.DataFrame([{
                    'Dataset': dataset_name,
                    'Original Rows': len(df),
                    'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'Columns': len(df.columns)
                }])
                info_csv = export_info.to_csv(index=False)
                st.download_button(
                    label="ℹ️ **Download Metadata**",
                    data=info_csv,
                    file_name=f"{dataset_name}_info.csv",
                    mime="text/csv",
                    use_container_width=True
                )

else:
    # Welcome screen when no data is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h2>🚀 Welcome to InsightStream</h2>
        <p style="font-size: 18px; color: #666;">Your Advanced Business Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👆 **Get Started:** Upload one or more CSV files using the sidebar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📁 **Multi-File Management**
        - Upload multiple CSV files
        - Smart data type detection
        - Easy dataset switching
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 **Advanced Joining**
        - Join datasets with different types
        - Automatic type conversion
        - Join compatibility checking
        """)
    
    with col3:
        st.markdown("""
        ### 📊 **Smart Visualizations**
        - Interactive charts and plots
        - Time series analysis
        - Correlation heatmaps
        """)
    
    with st.expander("🎯 **Feature Overview** - What You Can Do"):
        st.markdown("""
        **🔍 Advanced Filtering:**
        - Simple point-and-click filters
        - Powerful query builder with pandas syntax
        - Real-time filter preview
        
        **🧹 Data Cleaning:**
        - Automated quality checks
        - One-click data cleaning
        - Missing value handling
        
        **📈 Rich Visualizations:**
        - Bar, line, scatter, and box plots
        - Time series with date handling
        - Interactive correlation heatmaps
        
        **📊 Deep Analytics:**
        - Descriptive statistics
        - Group-by analysis
        - Data quality reporting
        
        **💾 Export Everything:**
        - Filtered datasets
        - Summary statistics  
        - Analysis metadata
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "Built with ❤️ using Streamlit • <strong>InsightStream v2.0</strong> • Ready for ML Integration"
    "</div>", 
    unsafe_allow_html=True
)