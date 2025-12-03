# FraudScope Dashboard Quick Start Guide

## Prerequisites

Before running the dashboard, ensure you have:

1. **Trained the model**: Run `notebooks/modeling_evaluation.ipynb` to train and save the XGBoost model
2. **Generated results**: The notebook should create all CSV files in the `results/` directory
3. **Test data available**: `data/raw/test_transaction.csv` should exist

## Installation

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Dashboard

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The dashboard will automatically open in your browser at `http://localhost:8501`

## Dashboard Sections

### 1. Overview
- Quick status of model and data availability
- Key performance metrics at a glance
- Available analyses summary

### 2. Model Performance
- Detailed model comparison table
- ROC-AUC and F1-Score visualizations
- Best model (XGBoost) metrics highlight

### 3. Fraud Risk Analysis
Interactive exploration of fraud patterns:

- **Region**: Geographic risk mapping by address combinations
  - Top 20 high-risk regions
  - Regional statistics and insights

- **Time Patterns**: Temporal fraud trends
  - Monthly fraud risk trends
  - Day-of-week patterns

- **Card Type**: Card vulnerability analysis
  - Card network (card4) risk
  - Card type (card6) risk
  - Combined card network + type analysis

- **Transaction Amount**: Amount-based risk analysis
  - Fraud probability by amount ranges
  - Key insights on high-risk amounts

### 4. Feature Importance
- Top N most important features from XGBoost
- Interactive bar chart visualization
- Feature importance statistics

## Troubleshooting

### Model Not Found
If you see "Model not found" warning:
- Run the modeling notebook to train and save the model
- Ensure the model is saved at `models/xgboost_fraud_model.pkl`

### Data Not Available
If analysis data is missing:
- Run the complete modeling notebook
- Check that CSV files exist in the `results/` directory

### Port Already in Use
If port 8501 is already in use:
- Stop other Streamlit apps
- Or specify a different port: `streamlit run app.py --server.port 8502`

## Features

- ✅ Interactive Plotly visualizations
- ✅ Real-time data filtering and exploration
- ✅ Responsive design for different screen sizes
- ✅ Cached data loading for performance
- ✅ Error handling for missing data

## Tips

- Use the sidebar to navigate between sections
- Adjust sliders and dropdowns to explore different views
- Hover over charts for detailed information
- Export data tables by using Streamlit's built-in download options







