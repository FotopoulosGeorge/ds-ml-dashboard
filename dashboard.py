# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import all our modules
from src.processing.data_filter import DataFilter
from src.analysis.data_visualizer import DataVisualizer  
from src.analysis.data_statistics import DataStatistics
from src.analysis.data_exporter import DataExporter
from src.analysis.data_preview import DataPreview
from src.processing.feature_engineering import FeatureEngineer
from src.ml.ml_trainer import MLTrainer
from src.processing.excel_handler import ExcelHandler
from src.demo.demo_datasets import DemoDatasets

# Page configuration
st.set_page_config(
    page_title="Data Science & Machine Learning Dashboard",
    page_icon="📊",
    layout="wide"
)



# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'model_chains' not in st.session_state:
    st.session_state.model_chains = {}

if 'ensemble_models' not in st.session_state:
    st.session_state.ensemble_models = {}

if 'stacking_ensembles' not in st.session_state:
    st.session_state.stacking_ensembles = {}

if 'ml_pipelines' not in st.session_state:
    st.session_state.ml_pipelines = {}

# Header
st.title("📊 Data Science & Machine Learning Dashboard")
st.markdown("• Analyze, Visualize, Train, Predict")

# Sidebar for file upload and controls


def render_data_input_section():
    """
    Render data input section - either file upload (local) or dataset selection (deployed)
    """
    st.sidebar.header("📁 Data Management")
    
    # Check if app is deployed
    is_deployed = DemoDatasets.is_deployed()
    
    if is_deployed:
        # DEPLOYED MODE: Use demo datasets only
        st.sidebar.info("🌐 **Demo Mode**: Using curated test datasets for privacy. Download app for full version")
        st.sidebar.markdown("*Your data stays safe - no uploads to cloud*")
        
        # Dataset selection
        datasets_info = DemoDatasets.get_available_datasets()
        dataset_names = list(datasets_info.keys())
        
        selected_dataset = st.sidebar.selectbox(
            "📊 Select Demo Dataset:",
            [""] + dataset_names,
            help="Choose a dataset to explore all features safely"
        )
        
        if selected_dataset:
            # Show dataset info
            info = datasets_info[selected_dataset]
            with st.sidebar.expander(f"ℹ️ About {selected_dataset}"):
                st.write(f"**Type:** {info['type']}")
                st.write(f"**Samples:** {info['samples']:,}")
                st.write(f"**Features:** {info['features']}")
                st.write(f"**Target:** {info['target']}")
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Use Cases:** {', '.join(info['use_cases'])}")
            
            if st.sidebar.button(f"📥 Load {selected_dataset}", type="primary"):
                try:
                    df = DemoDatasets.load_dataset(selected_dataset)
                    dataset_key = selected_dataset.replace(" ", "_").replace("🌸", "").replace("🍷", "").replace("🏠", "").replace("💊", "").replace("📈", "").replace("🛒", "").strip()
                    
                    # Store in session state
                    if 'datasets' not in st.session_state:
                        st.session_state.datasets = {}
                    
                    st.session_state.datasets[dataset_key] = df
                    st.sidebar.success(f"✅ Loaded {selected_dataset}")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to load dataset: {str(e)}")
        
        # Show available features in demo mode
        with st.sidebar.expander("🎯 What you can try"):
            st.markdown("""
            **🔬 Machine Learning:**
            - Classification & Regression
            - AutoML experiments  
            - Clustering analysis
            - Time series forecasting
            
            **🔧 Data Processing:**
            - Feature engineering
            - Data filtering & joining
            - Statistical analysis
            - Data visualization
            
            **🤖 Advanced Features:**
            - Pattern mining
            - Anomaly detection
            - Model comparison
            - Model persistence
            """)
    
    else:
        # LOCAL MODE: Allow file uploads
        st.sidebar.success("💻 **Local Mode**: Upload your own data safely")
        
        # Original file upload code
        uploaded_files = st.sidebar.file_uploader(
            "Upload Files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload CSV, Excel (.xlsx), or legacy Excel (.xls) files"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_extension = file_name.split('.')[-1].lower()

                if file_name not in st.session_state.datasets:
                    @st.cache_data
                    def load_data(file, file_extension):
                        try:
                            if file_extension == 'csv':
                                df = pd.read_csv(file)
                            elif file_extension in ['xlsx', 'xls']:
                                # Handle Excel files
                                excel_file = pd.ExcelFile(file)
                                                        
                                if len(excel_file.sheet_names) > 1:
                                    
                                    df = pd.read_excel(file, sheet_name=0)
                                    # Store sheet info for later use
                                    df.attrs['excel_sheets'] = excel_file.sheet_names
                                    df.attrs['selected_sheet'] = excel_file.sheet_names[0]
                                else:
                                    df = pd.read_excel(file)

                            # Auto-detect and convert date columns
                            for col in df.columns:
                                if df[col].dtype == 'object':
                                    try:
                                        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                                    except:
                                        pass
                            
                            if file_extension in ['xlsx', 'xls']:
                                df, sheet_name = ExcelHandler.handle_excel_upload(uploaded_file)
                                if df is not None:
                                    df = ExcelHandler.auto_detect_data_types(df)
                                    st.session_state.datasets[f"{file_name}_{sheet_name}"] = df

                            return df
                        
                        except Exception as e:
                            st.error(f"❌ Error loading {file.name}: {str(e)}")
                            return None
                    
                    

                    loaded_df = load_data(uploaded_file, file_extension)
                    if loaded_df is not None:
                        st.session_state.datasets[file_name] = loaded_df
                    
                    st.session_state.datasets[file_name] = load_data(uploaded_file, file_extension)
        
        # Show local mode benefits
        with st.sidebar.expander("🔒 Local Mode Benefits"):
            st.markdown("""
            **🛡️ Complete Privacy:**
            - Data never leaves your computer
            - No cloud storage or processing
            - Full GDPR compliance
            
            **📁 File Support:**
            - CSV files
            - Excel (.xlsx, .xls)  
            - Multiple sheets
            - Automatic type detection
            
            **🚀 Full Features:**
            - All ML algorithms
            - Custom data uploads
            - Model saving/loading
            - No usage limits
            """)
render_data_input_section()
# Dataset management sidebar
if st.session_state.datasets:
    st.sidebar.subheader("📋 Loaded Datasets")
    
    for name, df in st.session_state.datasets.items():
        with st.sidebar.expander(f"{name}"):
            st.write(f"📏 **Shape:** {df.shape[0]:,} × {df.shape[1]}")
            st.write(f"💾 **Size:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Show Excel sheet info if available
            if hasattr(df, 'attrs') and 'excel_sheets' in df.attrs:
                st.write(f"📑 **Sheet:** {df.attrs['selected_sheet']}")
                if len(df.attrs['excel_sheets']) > 1:
                    st.caption(f"Other sheets: {', '.join([s for s in df.attrs['excel_sheets'] if s != df.attrs['selected_sheet']])}")

            if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                del st.session_state.datasets[name]
                st.rerun()

        # from src.ml.performance_utils import SessionStateBackup
        # SessionStateBackup.show_backup_options()
        

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 Filter Data", 
        "📊 Visualize", 
        "📈 Statistics", 
        "🔧 Feature Engineering",
        "🧹 Preview",
        "💾 Export",
        "🤖 Machine Learning" 
    ])
    
    # =================== TAB 1: FILTERING ===================
    with tab1:
        # Intitialize or update DataFilter
        current_working_data = st.session_state.working_df.copy() 

        # Check if we need to reinitialize the filter (new columns added)
        if ('data_filter' not in st.session_state or 
            len(st.session_state.data_filter.original_df.columns) != len(current_working_data.columns) or
            list(st.session_state.data_filter.original_df.columns) != list(current_working_data.columns)):
            st.session_state.data_filter = DataFilter(current_working_data)

        # Render the filtering UI
        filtered_data, filters_applied = st.session_state.data_filter.render_filter_ui()
        
        # Update session state based on filtering results
        if filters_applied:
            st.session_state.working_df = filtered_data.copy()
            filter_info = st.session_state.data_filter.get_filter_info()
            st.session_state.active_filters = [filter_info] if filter_info['active'] else []
            
            # Don't force rerun for joins - let user navigate naturally
            join_applied = any("Joined" in detail for detail in filter_info.get('filters_applied', []))
            if not join_applied:
                # Only show navigation hint for regular filters
                if len(filtered_data) != len(st.session_state.base_df):
                    st.info("💡 **Filters applied!** Switch to other tabs to see filtered results")
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
        # Get current data
        current_data = st.session_state.working_df.copy()
        
        # Initialize or update FeatureEngineer
        if ('fe_handler' not in st.session_state or 
            len(st.session_state.fe_handler.df) != len(current_data)):
            st.session_state.fe_handler = FeatureEngineer(current_data)    
        st.session_state.fe_handler.render_feature_engineering_tab()
    
    # =================== TAB 5: PREVIEW ===================
    with tab5:
        # Initialize the preview module
        preview = DataPreview()
        
        # Render the preview tab
        preview.render_preview_tab()
    
    # =================== TAB 6: EXPORT ===================
    with tab6:
        # Initialize the exporter module
        exporter = DataExporter()
        
        # Render the export tab
        exporter.render_export_tab()

    # =================== TAB 7: MACHINE LEARNING ===================
    with tab7:
        # Initialize the machine learning module
        ml_trainer = MLTrainer()
        
        # Render the machine learning tab
        ml_trainer.render_ml_tab()

else:
    # Welcome screen when no data is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h2>🚀 Data Science & Machine Learning Dashboard</h2>
        <p style="font-size: 18px; color: #666;">Your Modular DS & ML Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        ### 🔧 **Feature Engineering**
        - Transform numerical data
        - Create ML-ready features  
        - Statistical transformations
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
        
        - **`data_filter.py`** - Independent filtering logic with dataset joining
        - **`data_visualizer.py`** - Chart generation and display  
        - **`data_statistics.py`** - Statistical analysis and insights
        - **`feature_engineering_tab.py`** - ML feature transformation pipeline
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
    "Built with ❤️ using Streamlit • <strong>Data Science & Machine Learning Dashboard v3.1 - Enhanced</strong>"
    "</div>", 
    unsafe_allow_html=True
)