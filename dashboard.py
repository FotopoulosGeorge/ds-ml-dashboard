import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="InsightStream",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'joined_data' not in st.session_state:
    st.session_state.joined_data = None

# Title and description
st.title("📊 InsightStream")
st.markdown("Upload multiple CSV files, join them, and create interactive visualizations with advanced filtering")

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

# Dataset management
if st.session_state.datasets:
    st.sidebar.subheader("📋 Loaded Datasets")
    
    for name, df in st.session_state.datasets.items():
        with st.sidebar.expander(f"{name} ({len(df)} rows)"):
            st.write(f"Columns: {len(df.columns)}")
            st.write(f"Size: {df.shape}")
            if st.button(f"Remove {name}", key=f"remove_{name}"):
                del st.session_state.datasets[name]
                st.rerun()

# Data joining section
if len(st.session_state.datasets) > 1:
    st.header("🔗 Data Joining")
    try: 
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
            st.subheader("🔍 Join Compatibility Check")
            
            left_dtype = str(left_df[left_key].dtype)
            right_dtype = str(right_df[right_key].dtype)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{left_dataset}[{left_key}]**")
                st.write(f"Type: `{left_dtype}`")
                st.write(f"Sample: {left_df[left_key].iloc[0]}")
            
            with col2:
                st.write("**Compatibility**")
                if left_dtype == right_dtype:
                    st.success("✅ Compatible")
                    compatibility = True
                else:
                    st.warning("⚠️ Type mismatch")
                    compatibility = False
            
            with col3:
                st.write(f"**{right_dataset}[{right_key}]**")
                st.write(f"Type: `{right_dtype}`")
                st.write(f"Sample: {right_df[right_key].iloc[0]}")

            # Auto-fix option for type mismatches
            auto_fix = False
            if not compatibility:
                auto_fix = st.checkbox(
                    "🔧 Auto-fix data types", 
                    help="Automatically convert data types to enable joining"
                )

            # Preview join results
            if st.button("👁️ Preview Join", help="See first 5 rows of join result"):
                try:
                    # Prepare data for joining
                    left_prep = left_df.copy()
                    right_prep = right_df.copy()
                    
                    # Auto-fix data types if requested
                    if auto_fix and not compatibility:
                        try:
                            # Try to convert to common type
                            if 'object' in [left_dtype, right_dtype]:
                                # Convert both to string
                                left_prep[left_key] = left_prep[left_key].astype(str)
                                right_prep[right_key] = right_prep[right_key].astype(str)
                                st.info("🔧 Converted both columns to string type")
                            elif 'int' in left_dtype and 'float' in right_dtype:
                                # Convert int to float
                                left_prep[left_key] = left_prep[left_key].astype(float)
                                st.info("🔧 Converted integer to float")
                            elif 'float' in left_dtype and 'int' in right_dtype:
                                # Convert int to float
                                right_prep[right_key] = right_prep[right_key].astype(float)
                                st.info("🔧 Converted integer to float")
                        except Exception as conv_error:
                            st.error(f"Auto-fix failed: {conv_error}")
                            left_prep, right_prep = left_df.copy(), right_df.copy()
                    
                    # Perform preview join
                    preview_join = pd.merge(
                        left_prep.head(100),  # Limit for preview
                        right_prep.head(100),
                        left_on=left_key,
                        right_on=right_key,
                        how=join_type,
                        suffixes=('_left', '_right')
                    )
                    
                    st.write("**Join Preview (first 5 rows):**")
                    st.dataframe(preview_join.head(), use_container_width=True)
                    
                    # Join statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Preview Rows", len(preview_join))
                    with col2:
                        st.metric("Total Columns", len(preview_join.columns))
                    with col3:
                        matching_keys = len(set(left_prep[left_key]) & set(right_prep[right_key]))
                        st.metric("Matching Keys", matching_keys)
                    
                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")
                    st.info("💡 Try enabling auto-fix or check your join keys")

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
            
            # Success message with details
            st.success(f"✅ Successfully joined datasets!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Result Rows", len(joined_df))
            with col2:
                st.metric("Result Columns", len(joined_df.columns))
            with col3:
                data_loss = len(left_df) + len(right_df) - len(joined_df)
                st.metric("Data Loss", data_loss)
            with col4:
                efficiency = len(joined_df) / max(len(left_df), len(right_df)) * 100
                st.metric("Join Efficiency", f"{efficiency:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Join failed: {str(e)}")
        
        # Helpful error messages
        if "can only concatenate str" in str(e):
            st.info("💡 **Solution**: Enable 'Auto-fix data types' to convert mismatched types")
        elif "incompatible dtype" in str(e):
            st.info("💡 **Solution**: Check that join columns contain compatible data")
        elif "not in index" in str(e):
            st.info("💡 **Solution**: Verify that the selected columns exist in both datasets")

# Select working dataset
st.header("📊 Data Analysis")

if st.session_state.joined_data is not None:
    dataset_options = ["Joined Data"] + list(st.session_state.datasets.keys())
    selected_dataset = st.selectbox("Select dataset to analyze:", dataset_options)
    
    if selected_dataset == "Joined Data":
        df = st.session_state.joined_data
    else:
        df = st.session_state.datasets[selected_dataset]
else:
    if st.session_state.datasets:
        dataset_options = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select dataset to analyze:", dataset_options)
        df = st.session_state.datasets[selected_dataset]
    else:
        df = None

if df is not None:
    # Data overview section
    st.subheader("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Advanced filtering section
    st.header("🔍 Advanced Filtering")
    
    # Filter type selection
    filter_type = st.radio(
        "Filter Mode:",
        ["Simple Filters", "Advanced Query Builder"],
        horizontal=True
    )
    
    filtered_df = df.copy()
    
    if filter_type == "Simple Filters":
        # Column selection for filtering
        filter_columns = st.multiselect(
            "Select columns to filter by:",
            options=df.columns.tolist(),
            help="Choose columns to create filters for"
        )
        
        if filter_columns:
            filter_col1, filter_col2 = st.columns(2)
            
            for i, col in enumerate(filter_columns):
                with filter_col1 if i % 2 == 0 else filter_col2:
                    st.write(f"**{col}** ({df[col].dtype})")
                    
                    if df[col].dtype in ['object', 'category']:
                        # Text/categorical filtering
                        filter_mode = st.radio(
                            "Filter mode:",
                            ["Select values", "Text pattern"],
                            key=f"mode_{col}",
                            horizontal=True
                        )
                        
                        if filter_mode == "Select values":
                            unique_values = df[col].unique()
                            selected_values = st.multiselect(
                                f"Select values:",
                                options=unique_values,
                                default=unique_values,
                                key=f"filter_{col}"
                            )
                            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                        
                        else:  # Text pattern
                            pattern = st.text_input(
                                "Text pattern (regex supported):",
                                key=f"pattern_{col}",
                                help="Use regex patterns like '^A' for starts with A"
                            )
                            case_sensitive = st.checkbox("Case sensitive", key=f"case_{col}")
                            
                            if pattern:
                                try:
                                    if case_sensitive:
                                        mask = filtered_df[col].str.contains(pattern, na=False, regex=True)
                                    else:
                                        mask = filtered_df[col].str.contains(pattern, na=False, case=False, regex=True)
                                    filtered_df = filtered_df[mask]
                                except Exception as e:
                                    st.error(f"Invalid pattern: {e}")
                    
                    elif df[col].dtype in ['int64', 'float64']:
                        # Numeric filtering
                        numeric_mode = st.radio(
                            "Filter type:",
                            ["Range", "Conditions"],
                            key=f"numeric_mode_{col}",
                            horizontal=True
                        )
                        
                        if numeric_mode == "Range":
                            min_val, max_val = float(df[col].min()), float(df[col].max())
                            selected_range = st.slider(
                                f"Value range:",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"range_{col}"
                            )
                            filtered_df = filtered_df[
                                (filtered_df[col] >= selected_range[0]) & 
                                (filtered_df[col] <= selected_range[1])
                            ]
                        
                        else:  # Conditions
                            condition = st.selectbox(
                                "Condition:",
                                ["equals", "greater than", "less than", "not equal"],
                                key=f"condition_{col}"
                            )
                            threshold = st.number_input(
                                "Value:",
                                value=float(df[col].mean()),
                                key=f"threshold_{col}"
                            )
                            
                            if condition == "equals":
                                filtered_df = filtered_df[filtered_df[col] == threshold]
                            elif condition == "greater than":
                                filtered_df = filtered_df[filtered_df[col] > threshold]
                            elif condition == "less than":
                                filtered_df = filtered_df[filtered_df[col] < threshold]
                            elif condition == "not equal":
                                filtered_df = filtered_df[filtered_df[col] != threshold]
                    
                    elif 'datetime' in str(df[col].dtype):
                        # Date filtering
                        st.write("Date range filter:")
                        min_date = df[col].min().date() if pd.notna(df[col].min()) else date.today()
                        max_date = df[col].max().date() if pd.notna(df[col].max()) else date.today()
                        
                        date_range = st.date_input(
                            "Select date range:",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key=f"date_{col}"
                        )
                        
                        if len(date_range) == 2:
                            start_date, end_date = date_range
                            filtered_df = filtered_df[
                                (filtered_df[col].dt.date >= start_date) & 
                                (filtered_df[col].dt.date <= end_date)
                            ]
                    
                    st.write("---")
    
    else:  # Advanced Query Builder
        st.subheader("Query Builder")
        
        query_text = st.text_area(
            "Write a pandas query:",
            help="Example: column1 > 100 and column2.str.contains('text') and column3.between(10, 50)",
            placeholder="column_name > 100 and other_column == 'value'",
            height=100
        )
        
        if query_text:
            try:
                filtered_df = df.query(query_text)
                st.success(f"Query executed successfully! {len(filtered_df)} rows returned.")
            except Exception as e:
                st.error(f"Query error: {str(e)}")
                filtered_df = df.copy()
        
        # Query examples
        with st.expander("Query Examples"):
            st.code("""
# Numeric conditions
sales > 1000
price.between(10, 100)
quantity >= 5

# Text conditions
category == 'Electronics'
name.str.contains('phone', case=False)
region.str.startswith('North')

# Date conditions
date >= '2023-01-01'
date.dt.year == 2023
date.dt.month.isin([1, 2, 3])

# Combined conditions
sales > 1000 and category == 'Electronics'
(price > 50) or (quantity > 10)
region.isin(['North', 'South']) and sales > 500
            """)
    
    # Show filtered data stats
    if len(filtered_df) != len(df):
        st.info(f"📊 Filtered data: **{len(filtered_df):,}** rows (from {len(df):,} total) - **{(len(filtered_df)/len(df)*100):.1f}%** of data")
    
    # Visualization section
    st.header("📈 Data Visualization")
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap", "Time Series"]
    )
    
    # Get column types
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = filtered_df.select_dtypes(include=['datetime64']).columns.tolist()
    all_cols = filtered_df.columns.tolist()
    
    # Chart configuration
    viz_col1, viz_col2 = st.columns(2)
    
    if chart_type in ["Bar Chart", "Line Chart"]:
        with viz_col1:
            x_axis = st.selectbox("X-axis:", all_cols, key="x_axis")
        with viz_col2:
            y_axis = st.selectbox("Y-axis:", numeric_cols, key="y_axis")
        
        # Additional options
        color_by = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key="color_by")
        color_col = None if color_by == "None" else color_by
        
        if chart_type == "Bar Chart":
            fig = px.bar(filtered_df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}")
        else:
            fig = px.line(filtered_df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} over {x_axis}")
    
    elif chart_type == "Scatter Plot":
        with viz_col1:
            x_axis = st.selectbox("X-axis:", numeric_cols, key="x_axis")
        with viz_col2:
            y_axis = st.selectbox("Y-axis:", numeric_cols, key="y_axis")
        
        color_by = st.selectbox("Color by (optional):", ["None"] + categorical_cols + numeric_cols)
        size_by = st.selectbox("Size by (optional):", ["None"] + numeric_cols)
        
        color_col = None if color_by == "None" else color_by
        size_col = None if size_by == "None" else size_by
        
        fig = px.scatter(
            filtered_df, 
            x=x_axis, 
            y=y_axis, 
            color=color_col,
            size=size_col,
            title=f"{y_axis} vs {x_axis}",
            hover_data=numeric_cols[:3]  # Show additional data on hover
        )
    
    elif chart_type == "Time Series":
        if date_cols:
            with viz_col1:
                date_col = st.selectbox("Date column:", date_cols, key="date_col")
            with viz_col2:
                value_col = st.selectbox("Value column:", numeric_cols, key="value_col")
            
            # Time series aggregation
            agg_level = st.selectbox("Aggregation:", ["No aggregation", "Daily", "Weekly", "Monthly"])
            
            if agg_level == "No aggregation":
                ts_df = filtered_df.sort_values(date_col)
            else:
                if agg_level == "Daily":
                    freq = 'D'
                elif agg_level == "Weekly":
                    freq = 'W'
                else:  # Monthly
                    freq = 'M'
                
                ts_df = filtered_df.set_index(date_col)[value_col].resample(freq).mean().reset_index()
            
            fig = px.line(ts_df, x=date_col, y=value_col, title=f"{value_col} over time")
        else:
            st.warning("No date columns found for time series chart")
            fig = None
    
    elif chart_type == "Histogram":
        with viz_col1:
            column = st.selectbox("Column:", numeric_cols, key="hist_col")
        with viz_col2:
            bins = st.slider("Number of bins:", 10, 100, 30)
        
        fig = px.histogram(filtered_df, x=column, nbins=bins, title=f"Distribution of {column}")
    
    elif chart_type == "Box Plot":
        with viz_col1:
            y_axis = st.selectbox("Y-axis (numeric):", numeric_cols, key="box_y")
        with viz_col2:
            x_axis = st.selectbox("X-axis (categorical, optional):", ["None"] + categorical_cols, key="box_x")
        
        x_col = None if x_axis == "None" else x_axis
        fig = px.box(filtered_df, x=x_col, y=y_axis, title=f"Box Plot of {y_axis}")
    
    elif chart_type == "Heatmap":
        if len(numeric_cols) > 1:
            corr_matrix = filtered_df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto",
                text_auto=True
            )
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap")
            fig = None
    
    # Display the chart
    if 'fig' in locals() and fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.header("📊 Summary Statistics")
    
    if numeric_cols:
        summary_option = st.selectbox(
            "Select summary type:",
            ["Descriptive Statistics", "Group By Analysis", "Missing Data Analysis"]
        )
        
        if summary_option == "Descriptive Statistics":
            st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)
        
        elif summary_option == "Group By Analysis" and categorical_cols:
            group_by_col = st.selectbox("Group by:", categorical_cols)
            metric_cols = st.multiselect("Analyze columns:", numeric_cols, default=numeric_cols[:3])
            
            if metric_cols:
                agg_functions = st.multiselect(
                    "Aggregation functions:",
                    ['count', 'mean', 'median', 'std', 'min', 'max', 'sum'],
                    default=['count', 'mean']
                )
                
                grouped_stats = filtered_df.groupby(group_by_col)[metric_cols].agg(agg_functions).round(2)
                st.dataframe(grouped_stats, use_container_width=True)
        
        elif summary_option == "Missing Data Analysis":
            missing_data = pd.DataFrame({
                'Column': filtered_df.columns,
                'Missing Count': filtered_df.isnull().sum(),
                'Missing Percentage': (filtered_df.isnull().sum() / len(filtered_df) * 100).round(2)
            })
            missing_data = missing_data[missing_data['Missing Count'] > 0]
            
            if len(missing_data) > 0:
                st.dataframe(missing_data, use_container_width=True)
            else:
                st.success("No missing data found!")
    
    # Export section
    st.header("💾 Export Data")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
    
    with export_col2:
        if numeric_cols:
            summary_csv = filtered_df[numeric_cols].describe().to_csv()
            st.download_button(
                label="📊 Download Summary Stats",
                data=summary_csv,
                file_name="summary_stats.csv",
                mime="text/csv"
            )
    
    with export_col3:
        # Export filter settings
        filter_info = {
            'dataset': selected_dataset if 'selected_dataset' in locals() else 'Unknown',
            'original_rows': len(df),
            'filtered_rows': len(filtered_df),
            'filter_percentage': f"{(len(filtered_df)/len(df)*100):.1f}%"
        }
        filter_csv = pd.DataFrame([filter_info]).to_csv(index=False)
        st.download_button(
            label="⚙️ Download Filter Info",
            data=filter_csv,
            file_name="filter_info.csv",
            mime="text/csv"
        )

else:
    # Instructions when no files are uploaded
    st.info("👆 Upload CSV file(s) using the sidebar to get started!")
    
    st.markdown("""
    ### 🚀 Advanced Features:
    
    #### 📁 Multi-File Management
    - Upload multiple CSV files simultaneously
    - Preview and manage each dataset individually
    - Remove datasets you no longer need
    
    #### 🔗 Data Joining
    - Join multiple datasets with different join types (inner, left, right, outer)
    - Select custom join keys from any column
    - Preview join results before analysis
    
    #### 🔍 Advanced Filtering
    - **Simple Filters**: Point-and-click filtering with intuitive controls
    - **Advanced Query Builder**: Write pandas queries for complex conditions
    - **Multiple Data Types**: Handle text, numeric, and date columns
    - **Pattern Matching**: Use regex for text filtering
    - **Date Ranges**: Filter time-series data by date ranges
    
    #### 📈 Enhanced Visualizations
    - Time series charts with aggregation options
    - Interactive scatter plots with size and color mapping
    - Correlation heatmaps with values displayed
    - Hover data for additional context
    
    #### 📊 Advanced Analytics
    - Multi-column group-by analysis
    - Missing data analysis and reporting
    - Customizable aggregation functions
    - Statistical summaries with multiple metrics
    
    ### 🎯 Getting Started:
    1. Upload one or more CSV files
    2. Join datasets if needed
    3. Apply advanced filters to focus your analysis
    4. Create interactive visualizations
    5. Export your results and insights
    
    **Coming Next**: Machine learning integration for predictive analytics!
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Advanced BI Features • ML-Ready Architecture")