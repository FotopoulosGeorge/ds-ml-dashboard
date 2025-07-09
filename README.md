# 📊 InsightStream - Advanced BI Dashboard

A powerful, interactive Business Intelligence dashboard built with Streamlit that enables users to upload, join, filter, and visualize data with enterprise-level features.

## 🚀 Key Features

### 📁 Multi-File Data Management
- Upload multiple CSV files simultaneously
- Preview and manage datasets individually
- Smart data type detection with automatic date parsing

### 🔗 Advanced Data Joining
- Join multiple datasets with flexible join types (inner, left, right, outer)
- Custom join key selection from any column
- Automatic handling of column conflicts with suffixes

### 🔍 Intelligent Filtering System
- **Simple Mode**: Point-and-click filtering with intuitive controls
- **Advanced Mode**: Pandas query builder for complex conditions
- **Multi-Type Support**: Text (with regex), numeric, and date filtering
- **Real-time Results**: Instant feedback on data filtering

### 📈 Interactive Visualizations
- Bar charts, line charts, scatter plots with color/size mapping
- Time series analysis with aggregation options
- Statistical distributions and correlation heatmaps
- Interactive hover data and drill-down capabilities

### 📊 Advanced Analytics
- Descriptive statistics and group-by analysis
- Missing data analysis and data quality reporting
- Multi-column aggregations with custom functions
- Export capabilities for filtered data and insights

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/goergefotopoulos/insightstream
cd insightstream

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 📋 Usage Guide

### Getting Started
1. **Upload Data**: Use the sidebar to upload one or more CSV files
2. **Join Data** (optional): If multiple files, join them using the data joining interface
3. **Filter Data**: Apply filters using either Simple or Advanced mode
4. **Visualize**: Select chart types and configure visualizations
5. **Analyze**: Review summary statistics and insights
6. **Export**: Download filtered data and analysis results

### Advanced Features

#### Multi-File Joining
- Upload multiple related datasets
- Select join keys and join types
- Preview results before committing

#### Advanced Filtering
**Simple Mode:**
- Categorical: Multi-select or regex patterns
- Numeric: Range sliders or conditional operators
- Dates: Calendar-based date range selection

**Advanced Mode:**
- Write pandas query expressions
- Use complex boolean logic
- Reference examples for common patterns

#### Query Examples
```python
# Numeric conditions
sales > 1000 and quantity >= 5

# Text patterns
category.str.contains('electronics', case=False)

# Date ranges
date >= '2023-01-01' and date.dt.month.isin([1,2,3])

# Combined conditions
(revenue > 10000) or (region == 'North' and sales > 5000)
```

## 🏗️ Technical Architecture

### Core Technologies
- **Frontend**: Streamlit for interactive web interface
- **Data Processing**: Pandas for data manipulation and analysis
- **Visualization**: Plotly for interactive charts and graphs
- **Performance**: PyArrow for optimized CSV processing

### Key Components
- **Session State Management**: Persistent data across user interactions
- **Data Caching**: Optimized performance for large datasets
- **Error Handling**: Robust validation and user feedback
- **Responsive Design**: Clean, intuitive user interface

## 📊 Performance Specifications

- **File Size**: Supports CSV files up to 200MB
- **Data Processing**: Handles datasets with 100K+ rows efficiently
- **Memory Management**: Optimized caching and data structures
- **Response Time**: Sub-second filtering and visualization updates

## 🔮 Future Enhancements

### Machine Learning Integration
- Predictive analytics and forecasting models
- Customer segmentation and clustering analysis
- Anomaly detection and outlier identification
- Automated feature engineering and insights

### Enhanced Capabilities
- Database connectivity (SQL, MongoDB)
- Real-time data streaming
- Custom dashboard templates
- Collaborative features and sharing

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Development Notes

### Code Structure
```
insightstream/
├── dashboard.py          # Main application file
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

### Testing
```bash
# Run basic functionality test
streamlit run test_streamlit.py

# Test with sample data
# Upload sample CSV files to verify all features
```

## 🐛 Troubleshooting

### Common Issues
- **White Screen**: Check terminal for import errors, verify Python version 3.8+
- **Port Conflicts**: Use `streamlit run dashboard.py --server.port 8502`
- **Package Issues**: Try `pip install --upgrade streamlit`

### Browser Compatibility
- Recommended: Chrome, Firefox, Safari (latest versions)
- JavaScript must be enabled
- Cookies and local storage should be allowed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️**