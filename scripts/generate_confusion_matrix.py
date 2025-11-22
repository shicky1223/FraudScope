"""
Script to generate confusion matrix CSV for XGBoost model
Run this after training the model in the notebook
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

def main():
    # Load model
    model_path = MODELS_DIR / "xgboost_fraud_model.pkl"
    if not model_path.exists():
        print("Error: Model file not found. Please train the model first.")
        return
    
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    
    # Load test data
    # Note: This assumes test data has been preprocessed the same way as training data
    # You may need to adjust this based on your preprocessing pipeline
    test_path = DATA_DIR / "raw" / "test_transaction.csv"
    if not test_path.exists():
        print("Error: Test data not found.")
        return
    
    print("Loading test data...")
    test_data = pd.read_csv(test_path)
    print(f"Test data shape: {test_data.shape}")
    
    # Note: You'll need to preprocess the test data the same way as training
    # This is a placeholder - adjust based on your actual preprocessing
    print("\n⚠️  Note: This script requires test data to be preprocessed the same way as training data.")
    print("Please ensure you have:")
    print("1. Preprocessed test data with the same features as training")
    print("2. True labels (isFraud column) for the test set")
    print("\nAlternatively, you can save the confusion matrix directly from the notebook.")
    
    # If you have preprocessed test data and labels, uncomment below:
    # X_test = test_data.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
    # y_test = test_data['isFraud']
    # 
    # y_pred = model.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    # 
    # # Save confusion matrix
    # cm_df = pd.DataFrame(
    #     cm,
    #     index=['Not Fraud', 'Fraud'],
    #     columns=['Not Fraud', 'Fraud']
    # )
    # cm_df.to_csv(RESULTS_DIR / "confusion_matrix_xgboost.csv")
    # print(f"\nConfusion matrix saved to {RESULTS_DIR / 'confusion_matrix_xgboost.csv'}")

if __name__ == "__main__":
    main()

