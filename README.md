# FraudScope

**AI-Driven Fraud Detection and Global Risk Mapping Using Ensemble Learning**

FraudScope is a comprehensive fraud detection system that leverages ensemble learning methods to identify fraudulent transactions and analyze fraud patterns across multiple dimensions including geography, time, card types, and transaction amounts.

## Overview

This project implements and evaluates multiple ensemble learning algorithms for credit card fraud detection, providing detailed analysis of fraud patterns and risk factors. The system uses advanced machine learning techniques to achieve high-precision fraud detection while maintaining interpretability through feature importance analysis.

## Features

- **Multiple Ensemble Models**: Implements Random Forest, AdaBoost, XGBoost, and Gradient Boosting classifiers
- **Comprehensive Evaluation**: ROC-AUC, Precision-Recall curves, F1-Score, and Confusion Matrix analysis
- **Fraud Pattern Analysis**: 
  - Geographic risk mapping by region
  - Temporal analysis (monthly, weekly, daily, hourly trends)
  - Card type vulnerability assessment
  - Transaction amount risk analysis
- **Feature Importance**: Identifies top predictive features for fraud detection
- **Visualizations**: Generates comprehensive charts and heatmaps for analysis

## Dataset

The project uses transaction and identity data:
- **Training Data**: `train_transaction.csv`, `train_identity.csv`
- **Test Data**: `test_transaction.csv`, `test_identity.csv`
- **Processed Data**: `clean_train.parquet` (preprocessed training data)
- **Dataset Size**: 590,540 transactions with 435 features
- **Fraud Rate**: 3.50% (20,663 fraudulent transactions)

## Model Performance

### Best Model: XGBoost

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.9213 |
| **Average Precision** | 0.6483 |
| **F1-Score** | 0.5571 |
| **Precision** | 0.8941 |
| **Recall** | 0.4045 |

### Model Comparison

| Model | Rank | ROC-AUC | Average Precision | F1-Score | Precision | Recall |
|-------|------|---------|-------------------|----------|-----------|--------|
| XGBoost | 1 | 0.9213 | 0.6483 | 0.5571 | 0.8941 | 0.4045 |
| Gradient Boosting | 2 | 0.9007 | 0.6225 | 0.5571 | 0.9029 | 0.4029 |
| Random Forest | 3 | 0.8717 | 0.4815 | 0.2589 | 0.1579 | 0.7193 |
| AdaBoost | 4 | 0.8411 | 0.3405 | 0.0370 | 0.9070 | 0.0189 |

## Key Findings

### 1. Geographic Risk Patterns
- Highest fraud risk regions identified through address combinations (addr1 + addr2)
- Top risk region: addr1=296.0, addr2=65.0 (42.68% fraud probability)
- Regional heatmap visualization available

### 2. Temporal Patterns
- **Month**: February shows highest fraud risk (4.09%)
- **Day of Week**: Sunday has highest fraud probability (3.71%)
- **Hour**: Specific hours show elevated risk patterns

### 3. Card Type Vulnerability
- **Card Network**: Discover cards show highest risk (6.94%)
- **Card Type**: Credit cards are more vulnerable (6.55%) than debit cards (2.47%)
- **Combined**: Discover credit cards have highest risk (7.19%)

### 4. Transaction Amount Analysis
- Small transactions ($0-10) show highest fraud risk (8.51%)
- Very large transactions ($5K+) also show elevated risk (5.91%)
- Medium-range transactions ($100-250) have lower risk (3.08%)

### 5. Top Predictive Features
The most important features for fraud detection (XGBoost):
- V258 (17.6% importance)
- V201 (5.9% importance)
- V149 (5.3% importance)
- V70, V91, V147, V172, V294, V225, C14

## File Structure

```
FraudScope/
├── README.md                          # Project documentation
├── modeling_evaluation.ipynb          # Main analysis notebook
├── clean_train.parquet                # Preprocessed training data
├── train_transaction.csv              # Training transaction data
├── train_identity.csv                 # Training identity data
├── test_transaction.csv               # Test transaction data
├── test_identity.csv                  # Test identity data
│
├── model_comparison.csv               # Model performance comparison
│
├── roc_curves.png                     # ROC curve comparison
├── precision_recall_curves.png        # Precision-Recall curves
├── confusion_matrices.png             # Confusion matrices for all models
├── feature_importance.png             # Top 20 feature importance
│
├── fraud_risk_by_region.csv           # Regional fraud risk analysis
├── fraud_risk_by_region_heatmap.png   # Regional risk heatmap
├── fraud_risk_by_month.csv            # Monthly fraud patterns
├── fraud_risk_by_day_of_week.csv      # Day-of-week patterns
├── fraud_risk_over_time.png           # Temporal trend visualizations
│
├── fraud_risk_by_card4.csv            # Card network analysis
├── fraud_risk_by_card6.csv            # Card type analysis
├── fraud_risk_by_card_combined.csv    # Combined card analysis
├── fraud_risk_by_card.png             # Card risk visualizations
│
└── fraud_risk_by_amount.csv           # Transaction amount analysis
    fraud_risk_by_amount.png           # Amount risk visualizations
```

## Usage

### Running the Analysis

1. **Prerequisites**: Install required Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
   ```

2. **Execute the Notebook**:
   - Open `modeling_evaluation.ipynb` in Jupyter Notebook or JupyterLab
   - Run all cells to:
     - Load and preprocess data
     - Train ensemble models
     - Evaluate model performance
     - Generate fraud pattern analyses
     - Create visualizations

3. **View Results**:
   - Model comparison: `model_comparison.csv`
   - Visualizations: PNG files in the project directory
   - Analysis tables: CSV files for each analysis dimension

### Data Preprocessing

The notebook handles:
- Missing value imputation (median for numeric, mode for categorical)
- Categorical variable encoding (one-hot encoding)
- Feature engineering and selection
- Train-test split (80-20) with stratification

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- scipy

## Results Summary

The XGBoost ensemble model achieved the best performance with:
- **92.13% ROC-AUC**: Excellent discrimination between fraud and legitimate transactions
- **89.41% Precision**: Low false positive rate
- **40.45% Recall**: Identifies a significant portion of fraudulent transactions
- **55.71% F1-Score**: Balanced precision-recall performance

The ensemble methods demonstrate superior performance by combining multiple weak learners, with XGBoost providing the optimal balance of precision and recall for fraud detection.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
