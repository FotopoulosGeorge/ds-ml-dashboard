# 📊 InsightStream - Full-Stack Data Science Platform

A comprehensive data science platform built with Streamlit that takes you from raw data to trained ML models. Upload, clean, engineer features, and build machine learning models - all in one unified interface.

## 🚀 Key Features

### 🏗️ **Modular Architecture**
- **Clean separation** of data processing, feature engineering, and ML components
- **Session state management** for seamless data flow across all modules
- **Easy to extend** - add new algorithms or features without breaking existing functionality

### 📁 **Complete Data Pipeline**
- **Multi-file upload** with automatic data type detection
- **Smart dataset joining** with type compatibility checking
- **Advanced filtering** with both UI controls and pandas query builder
- **Data quality assessment** with actionable insights

### 🔧 **Comprehensive Feature Engineering**
- **Numerical transformations**: scaling, normalization, log transforms, binning
- **Text feature extraction**: length, word count, character analysis
- **Date/time features**: seasonality, trends, time-based indicators  
- **Categorical encoding**: one-hot, label, frequency, target encoding
- **Advanced features**: interactions, ratios, outlier detection

### 🤖 **Machine Learning Suite**
- **Supervised learning**: classification and regression with 6+ algorithms
- **Unsupervised learning**: K-means clustering with interactive visualization
- **Model evaluation**: comprehensive metrics, ROC curves, feature importance
- **Model comparison**: side-by-side performance analysis
- **Prediction interface**: single predictions and batch processing
- **Model persistence**: save and load trained models

### 📊 **Interactive Analytics**
- **Rich visualizations**: scatter plots, heatmaps, time series, cluster plots
- **Statistical analysis**: distributions, correlations, group-by operations
- **Export capabilities**: processed datasets, models, and analysis reports

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/insightstream
cd insightstream

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies  
pip install -r requirements.txt

# Run the application
streamlit run dashboard.py
```

## 📁 Project Structure

```
insightstream/
├── dashboard.py                    # Main application orchestrator
├── src/
│   ├── data_filter.py             # Data filtering and joining
│   ├── data_visualizer.py         # Interactive visualizations
│   ├── data_statistics.py         # Statistical analysis
│   ├── data_preview.py            # Data quality and exploration
│   ├── data_exporter.py           # Export functionality
│   ├── feature_engineering.py     # Feature transformation pipeline
│   └── ml/                        # Machine Learning module
│       ├── ml_trainer.py          # ML training interface
│       ├── ml_evaluator.py        # Model evaluation and metrics
│       ├── ml_utils.py            # ML utilities and helpers
│       └── models/                # Saved models (auto-created)
├── requirements.txt               # Dependencies
└── README.md
```

## 🎯 Workflow

1. **📁 Upload** - Load CSV files with automatic data type detection
2. **🔍 Filter** - Clean and join datasets using intuitive controls
3. **🔧 Engineer** - Create ML-ready features with transformations
4. **🤖 Train** - Build and compare machine learning models
5. **📊 Analyze** - Evaluate performance with comprehensive metrics
6. **🔮 Predict** - Make predictions on new data
7. **💾 Export** - Download models, predictions, and analysis reports

## 🧠 Supported ML Algorithms

**Classification**: Logistic Regression, Random Forest, SVM, Decision Tree, KNN, Naive Bayes  
**Regression**: Linear Regression, Random Forest, SVR, Decision Tree, KNN  
**Clustering**: K-Means with interactive 2D/3D visualization

## 🔮 Advanced Features

- **Cross-validation** for robust model evaluation
- **Feature importance** analysis and visualization  
- **Automated problem type detection** (classification vs regression)
- **Hyperparameter recommendations** with sensible defaults
- **Model persistence** with metadata tracking
- **Batch predictions** via CSV upload
- **Interactive cluster visualization** in 2D and 3D

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