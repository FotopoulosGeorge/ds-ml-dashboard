# 📊 InsightStream - Modular BI Dashboard

A powerful, modular Business Intelligence dashboard built with Streamlit that enables users to upload, join, filter, transform, and visualize data with enterprise-level features.

## 🚀 Key Features

### 🏗️ **Modular Architecture**
- **Independent modules** for each major function
- **Clean data flow** through session state management
- **No scope issues** - each module works independently
- **Easy to extend** - add new features without breaking existing ones

### 📁 **Multi-File Data Management**
- Upload multiple CSV files simultaneously
- Smart data type detection with automatic date parsing
- **Dataset joining** with flexible join types and automatic type conversion
- Easy dataset switching and management

### 🔍 **Advanced Filtering & Joining**
- **Simple Mode**: Point-and-click filtering with intuitive controls
- **Advanced Mode**: Pandas query builder for complex conditions
- **Dataset Joining**: Merge multiple datasets with compatibility checking
- Real-time filter results with persistent state

### 🔧 **Feature Engineering Pipeline**
- **Numerical transformations**: Log, square root, standardization, scaling
- **ML-ready features**: Quantile binning, outlier handling
- **Interactive transformation** with preview capabilities
- Seamless integration with analysis pipeline

### 📊 **Interactive Visualizations**
- Bar charts, line charts, scatter plots with color/size mapping
- Time series analysis and correlation heatmaps
- **Always reflects filtered data** - no disconnects
- Interactive hover data and export-ready graphics

### 📈 **Deep Analytics**
- Descriptive statistics and distribution analysis
- Group-by analysis and missing data reporting
- Data quality checks with actionable insights
- Comprehensive data preview and exploration

### 💾 **Comprehensive Export System**
- Filtered datasets in multiple formats
- Statistical summaries and data reports
- Filter information and metadata export
- All analytics combined for complete documentation

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/FotopoulosGeorge/insightstream-bi
cd insightstream

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 📁 **Project Structure**

```
insightstream/
├── dashboard.py           # Main orchestrator
├── data_filter.py             # Filtering & joining logic
├── data_visualizer.py         # Chart generation
├── data_statistics.py         # Statistical analysis
├── data_preview.py            # Data exploration
├── data_exporter.py           # Export functionality
├── feature_engineering.py     # Core feature engineering 
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🎯 **Quick Start Guide**

1. **Upload Data**: Use sidebar to upload CSV files
2. **Filter Tab**: Apply filters or join multiple datasets
3. **Feature Engineering Tab**: Transform data for ML analysis
4. **Visualize Tab**: Create interactive charts from processed data
5. **Statistics Tab**: Generate insights and summaries
6. **Preview Tab**: Explore data quality and structure
7. **Export Tab**: Download results in various formats

## 🔧 **Technical Highlights**

- **Session State Management**: Persistent data across all modules
- **Independent Module Design**: No variable scope conflicts
- **Real-time Data Flow**: Filtering → Feature Engineering → Visualization → Export
- **Error Handling**: Robust validation and user feedback
- **Performance Optimized**: Efficient data processing for large datasets

## 🔮 **Machine Learning Ready**

InsightStream is designed as a complete **ML preprocessing pipeline**:
- Data cleaning and quality checks
- Feature engineering and transformation
- Statistical analysis and insights
- Export ML-ready datasets

Perfect foundation for data science projects and model development.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️**