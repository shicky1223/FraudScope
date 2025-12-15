"""
Script to generate confusion matrix CSV for XGBoost model
Run this after training the model in the notebook
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

def main():
    model_path = MODELS_DIR / "xgboost_fraud_model.pkl"
    if not model_path.exists():
        print("Error: Model file not found. Please train the model first.")
        return
    
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    
    test_path = DATA_DIR / "raw" / "test_transaction.csv"
    if not test_path.exists():
        print("Error: Test data not found.")
        return
    
    print("Loading test data...")
    test_data = pd.read_csv(test_path)
    print(f"Test data shape: {test_data.shape}")
    
    print("\n⚠️  Note: This script requires test data to be preprocessed the same way as training data.")
    print("Please ensure you have:")
    print("1. Preprocessed test data with the same features as training")
    print("2. True labels (isFraud column) for the test set")
    print("\nAlternatively, you can save the confusion matrix directly from the notebook.")

if __name__ == "__main__":
    main()

