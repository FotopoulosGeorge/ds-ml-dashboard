# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import all our modules
from data_filter import DataFilter
from data_visualizer import DataVisualizer  
from data_statistics import DataStatistics
from data_exporter import DataExporter
from data_preview import DataPreview
from feature_engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="InsightStream - BI Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

# Header
st.title("📊 InsightStream")
st.markdown("** Business Intelligence Dashboard** • Upload, Filter, Analyze, Visualize")

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
    
    # Dataset selection
    if len(st.session_state.datasets) > 1:
        dataset_options = [f"📄 {name}" for name in st.session_state.datasets.keys()]
        selected_option = st.selectbox("**Select dataset to analyze:**", dataset_options)
        dataset_name = selected_option.replace("📄 ", "")
        df = st.session_state.datasets[dataset_name]
    else:
        dataset_name = list(st.session_state.datasets.keys())[0]
        df = st.session_state.datasets[dataset_name]
    
    # Initialize session state variables for current dataset
    if 'base_df' not in st.session_state:
        st.session_state.base_df = df.copy()
    if 'working_df' not in st.session_state:
        st.session_state.working_df = df.copy()
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = []
    
    # Check if dataset changed (reset state for new dataset)
    if (len(st.session_state.base_df) != len(df) or 
        list(st.session_state.base_df.columns) != list(df.columns)):
        st.session_state.base_df = df.copy()
        st.session_state.working_df = df.copy()
        st.session_state.active_filters = []
    
    # Quick metrics header (always visible)
    current_rows = len(st.session_state.working_df)
    original_rows = len(st.session_state.base_df)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        if current_rows != original_rows:
            delta = f"{current_rows - original_rows:,} filtered"
            st.metric("📊 **Total Rows**", f"{current_rows:,}", delta=delta)
        else:
            st.metric("📊 **Total Rows**", f"{current_rows:,}")
    
    with metric_col2:
        st.metric("📋 **Columns**", len(st.session_state.working_df.columns))
    
    with metric_col3:
        numeric_count = len(st.session_state.working_df.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 **Numeric**", numeric_count)
    
    with metric_col4:
        missing_count = st.session_state.working_df.isnull().sum().sum()
        st.metric("❓ **Missing**", f"{missing_count:,}")
    
    # Filter status bar
    if len(st.session_state.active_filters) > 0:
        filter_col1, filter_col2 = st.columns([4, 1])
        
        with filter_col1:
            reduction_pct = ((original_rows - current_rows) / original_rows * 100)
            st.markdown(f"""
            🔍 **Active Filters:** Showing **{current_rows:,}** of **{original_rows:,}** rows 
            ({reduction_pct:.1f}% filtered)
            """)
        
        with filter_col2:
            if st.button("❌ Clear Filters", key="clear_all_filters"):
                st.session_state.working_df = st.session_state.base_df.copy()
                st.session_state.active_filters = []
                st.rerun()
    
    st.markdown("---")
    
    # =================== CREATE TABS ===================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Filter Data", 
        "📊 Visualize", 
        "📈 Statistics", 
        "🔧 Feature Engineering",
        "🧹 Preview",
        "💾 Export"
    ])
    
    # =================== TAB 1: FILTERING ===================
    with tab1:
        # Initialize the filter module
        if 'data_filter' not in st.session_state:
            st.session_state.data_filter = DataFilter(st.session_state.base_df)
        
        # Check if we need to reinitialize for new dataset
        if (len(st.session_state.data_filter.original_df) != len(st.session_state.base_df) or
            list(st.session_state.data_filter.original_df.columns) != list(st.session_state.base_df.columns)):
            st.session_state.data_filter = DataFilter(st.session_state.base_df)
        
        # Render the filtering UI
        filtered_data, filters_applied = st.session_state.data_filter.render_filter_ui()
        
        # Update session state based on filtering results
        if filters_applied:
            st.session_state.working_df = filtered_data.copy()
            filter_info = st.session_state.data_filter.get_filter_info()
            st.session_state.active_filters = [filter_info] if filter_info['active'] else []
        else:
            # Check if we should clear filters
            if len(st.session_state.active_filters) > 0:
                st.session_state.working_df = st.session_state.base_df.copy()
                st.session_state.active_filters = []
    
    # =================== TAB 2: VISUALIZATION ===================
    with tab2:
        # Initialize the visualizer module
        visualizer = DataVisualizer()
        
        # Render the visualization tab
        visualizer.render_visualization_tab()
    
    # =================== TAB 3: STATISTICS ===================
    with tab3:
        # Initialize the statistics module
        statistics = DataStatistics()
        
        # Render the statistics tab
        statistics.render_statistics_tab()


    # =================== TAB 4: FEATURE ENGINEERING ===================
    with tab4:
        st.header("🔧 **Feature Engineering**")
        
        # Get current data
        current_data = st.session_state.working_df.copy()
        
        # Initialize FeatureEngineer directly
        if 'fe_handler' not in st.session_state:
            st.session_state.fe_handler = FeatureEngineer(current_data)
        
        # Check if data changed
        if not st.session_state.fe_handler.df.equals(current_data):
            st.session_state.fe_handler = FeatureEngineer(current_data)
        
        fe_handler = st.session_state.fe_handler
        
        # UI code directly here - no wrapper needed
        column_info = fe_handler.get_column_info()
        numeric_cols = column_info['numeric']
        
        if numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_col = st.selectbox("Select column:", numeric_cols)
                transform_type = st.selectbox("Transformation:", 
                                            fe_handler.get_available_numerical_transforms())
            
            with col2:
                info = fe_handler.get_transform_info(transform_type)
                st.info(f"**{transform_type}:** {info['description']}")
                
                new_name = st.text_input("New column name:", 
                                        value=f"{selected_col}_{transform_type.lower().replace(' ', '_')}")
            
            if st.button("🔧 Apply Transformation"):
                try:
                    created_col = fe_handler.apply_numerical_transform(
                        selected_col, transform_type, new_name
                    )
                    st.success(f"✅ Created feature: {created_col}")
                    # Update working data
                    st.session_state.working_df = fe_handler.df.copy()
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
    
    # =================== TAB 4: PREVIEW ===================
    with tab4:
        # Initialize the preview module
        preview = DataPreview()
        
        # Render the preview tab
        preview.render_preview_tab()
    
    # =================== TAB 5: EXPORT ===================
    with tab5:
        # Initialize the exporter module
        exporter = DataExporter()
        
        # Render the export tab
        exporter.render_export_tab()

else:
    # Welcome screen when no data is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h2>🚀 Welcome to InsightStream</h2>
        <p style="font-size: 18px; color: #666;">Your Modular Business Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👆 **Get Started:** Upload one or more CSV files using the sidebar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 **Smart Filtering**
        - Simple point-and-click filters
        - Advanced query builder
        - Real-time data updates
        """)
    
    with col2:
        st.markdown("""
        ### 📊 **Interactive Visualizations**
        - Multiple chart types
        - Dynamic data binding
        - Export-ready graphics
        """)
    
    with col3:
        st.markdown("""
        ### 📈 **Deep Analytics**
        - Statistical summaries
        - Data quality reports
        - Comprehensive exports
        """)
    
    with st.expander("🏗️ **Architecture Overview** - Modular Design"):
        st.markdown("""
        **🧩 Modular Components:**
        
        - **`data_filter.py`** - Independent filtering logic
        - **`data_visualizer.py`** - Chart generation and display  
        - **`data_statistics.py`** - Statistical analysis and insights
        - **`data_preview.py`** - Data exploration and quality checks
        - **`data_exporter.py`** - Export functionality for all formats
        
        **✅ Benefits:**
        - Clean separation of concerns
        - Easy to maintain and extend
        - No variable scope issues
        - Independent testing possible
        - Consistent data flow through session state
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "Built with ❤️ using Streamlit • <strong>InsightStream v0.3 - Modular</strong>"
    "</div>", 
    unsafe_allow_html=True
)