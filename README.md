# 📊 InsightStream - Complete Data Science Platform

Transform raw data into trained ML models with zero coding. Upload CSV files, engineer features, train algorithms, and make predictions - all through an intuitive web interface.

## 🎯 Core Workflow

**1. Upload Data** → **2. Process & Clean** → **3. Train Models** → **4. Make Predictions**

- **📁 Data Input**: Multi-file CSV upload with automatic type detection
- **🔧 Processing**: Filter, join, and engineer features for ML
- **🤖 Training**: 15+ algorithms including time series and anomaly detection  
- **🔮 Predictions**: Real-time predictions and batch processing

## 🚀 Quick Start

```bash
git clone https://github.com/FotopoulosGeorge/insightstream-bi
cd insightstream-bi
pip install -r requirements.txt
streamlit run dashboard.py
```

## 🛠️ What's Inside

### **Data Processing**
- Smart filtering with UI controls and pandas queries
- Dataset joining with automatic type compatibility
- Feature engineering: scaling, encoding, text analysis, date features
- Data quality assessment and cleaning recommendations

### **Machine Learning**
- **Supervised**: Classification & regression (6 algorithms)
- **Unsupervised**: K-means clustering with 3D visualization
- **Time Series**: Prophet forecasting with seasonality analysis
- **Anomaly Detection**: Isolation Forest and Local Outlier Factor
- **Model Management**: Save, load, and compare trained models

### **Analysis & Export**
- Interactive visualizations with Plotly
- Comprehensive model evaluation and metrics
- Statistical analysis and data exploration
- Export models, predictions, and reports

## 📁 Architecture

```
src/
├── processing/          # Data filtering and feature engineering
├── analysis/           # Visualization, statistics, preview, export  
└── ml/                # Machine learning and pretrained models
```

## 🎯 Use Cases

**Business Analytics**: Sales forecasting, customer segmentation, anomaly detection  
**Data Science**: Rapid prototyping, model comparison, feature engineering  
**Education**: Complete ML workflow demonstration and learning  
**Teams**: Shared data analysis platform without coding requirements

## 📈 Supported Models

**Classification/Regression**: Logistic Regression, Random Forest, SVM, Decision Tree, KNN, Naive Bayes  
**Time Series**: Prophet forecasting with trend and seasonality  
**Anomaly Detection**: Isolation Forest, Local Outlier Factor  
**Clustering**: K-Means with interactive visualization

## 🔧 Requirements

- Python 3.8+
- 8GB RAM recommended for large datasets
- Modern web browser

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

**Built with ❤️**