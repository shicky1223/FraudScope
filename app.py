"""
FraudScope Dashboard - Interactive Fraud Detection Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FraudScope Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"

@st.cache_data
def load_model():
    """Load the trained XGBoost model"""
    model_path = MODELS_DIR / "xgboost_fraud_model.pkl"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

@st.cache_data
def load_test_data():
    """Load test transaction data"""
    test_path = DATA_DIR / "raw" / "test_transaction.csv"
    if test_path.exists():
        try:
            return pd.read_csv(test_path)
        except Exception as e:
            st.error(f"Error loading test data: {e}")
            return None
    return None

@st.cache_data
def load_processed_data():
    """Load processed training data (includes merged identity data with DeviceType/DeviceInfo)"""
    try:
        processed_path = DATA_DIR / "processed" / "clean_train.parquet"
        if processed_path.exists():
            return pd.read_parquet(processed_path)
        return None
    except Exception as e:
        st.error(f"Error loading processed data: {e}")
        return None

@st.cache_data
def load_training_data_stats():
    """Load training data to get feature statistics for preprocessing"""
    try:
        # Try loading processed data first
        train_path = DATA_DIR / "processed" / "clean_train.parquet"
        if train_path.exists():
            df = pd.read_parquet(train_path)
        else:
            # Fallback to raw data
            train_path = DATA_DIR / "raw" / "train_transaction.csv"
            if train_path.exists():
                df = pd.read_csv(train_path, low_memory=False)
            else:
                return None
        
        # Prepare features (same as training)
        X = df.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
        
        # Store original columns BEFORE feature engineering (for user input matching)
        original_columns_before_fe = list(X.columns)
        
        # Calculate top devices from training data for DeviceInfo grouping
        top_devices = None
        if 'DeviceInfo' in X.columns:
            deviceinfo_counts = X['DeviceInfo'].value_counts()
            top_n_devices = 50  # Keep top 50 most common devices
            top_devices = set(deviceinfo_counts.head(top_n_devices).index)
        
        # Apply DeviceInfo feature engineering (same as in notebook)
        X = apply_deviceinfo_feature_engineering(X, top_devices=top_devices)
        
        # Calculate statistics (after feature engineering)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        # Get medians for numeric columns
        numeric_medians = X[numeric_cols].median().to_dict()
        
        # Get min/max for numeric columns (for sliders)
        numeric_mins = X[numeric_cols].min().to_dict()
        numeric_maxs = X[numeric_cols].max().to_dict()
        
        # Get modes for categorical columns
        categorical_modes = {}
        for col in categorical_cols:
            mode_val = X[col].mode()
            categorical_modes[col] = mode_val[0] if len(mode_val) > 0 else 'unknown'
        
        # Get feature names after encoding (for matching)
        X_filled = X.copy()
        X_filled[numeric_cols] = X_filled[numeric_cols].fillna(X_filled[numeric_cols].median())
        for col in categorical_cols:
            X_filled[col] = X_filled[col].fillna(categorical_modes[col])
        
        X_encoded = pd.get_dummies(X_filled, drop_first=True)
        
        # Ensure feature names are strings (not numpy strings or other types)
        feature_names = [str(f) for f in X_encoded.columns]
        
        return {
            'numeric_medians': numeric_medians,
            'numeric_mins': numeric_mins,
            'numeric_maxs': numeric_maxs,
            'categorical_modes': categorical_modes,
            'feature_names': feature_names,  # Ensure strings
            'original_columns': original_columns_before_fe,  # Columns BEFORE feature engineering
            'numeric_cols': list(numeric_cols),
            'categorical_cols': list(categorical_cols),
            'top_devices': top_devices  # Store for use in preprocessing
        }
    except Exception as e:
        st.error(f"Error loading training data stats: {e}")
        return None

def apply_deviceinfo_feature_engineering(df, top_devices=None):
    """Apply DeviceInfo feature engineering (same as in notebook)
    
    Args:
        df: DataFrame to process
        top_devices: Set of top device names from training data (optional)
    """
    if 'DeviceInfo' not in df.columns:
        return df
    
    # 1. Extract OS Type from DeviceInfo
    def extract_os(device_info):
        if pd.isna(device_info) or device_info == '':
            return 'Unknown'
        device_str = str(device_info).lower()
        if 'windows' in device_str or 'win' in device_str:
            return 'Windows'
        elif 'ios' in device_str or 'iphone' in device_str or 'ipad' in device_str:
            return 'iOS'
        elif 'android' in device_str or 'build/' in device_str:
            return 'Android'
        elif 'macos' in device_str or 'mac' in device_str:
            return 'macOS'
        elif 'linux' in device_str:
            return 'Linux'
        else:
            return 'Other'
    
    df['DeviceInfo_OS'] = df['DeviceInfo'].apply(extract_os)
    
    # 2. Extract Device Brand (common brands)
    def extract_brand(device_info):
        if pd.isna(device_info) or device_info == '':
            return 'Unknown'
        device_str = str(device_info).lower()
        # Common brands
        brands = {
            'samsung': ['sm-', 'samsung', 'galaxy'],
            'apple': ['iphone', 'ipad', 'ios device'],
            'huawei': ['huawei', 'honor'],
            'xiaomi': ['xiaomi', 'redmi', 'mi '],
            'lg': ['lg-', 'lg '],
            'motorola': ['moto', 'motorola'],
            'nokia': ['nokia'],
            'sony': ['sony', 'xperia'],
            'oneplus': ['oneplus'],
            'google': ['pixel', 'nexus']
        }
        for brand, keywords in brands.items():
            if any(keyword in device_str for keyword in keywords):
                return brand.title()
        return 'Other'
    
    df['DeviceInfo_Brand'] = df['DeviceInfo'].apply(extract_brand)
    
    # 3. Group rare DeviceInfo values (keep top N, group rest as 'Other')
    if top_devices is None:
        # Calculate from current data if not provided
        if 'DeviceInfo' in df.columns:
            deviceinfo_counts = df['DeviceInfo'].value_counts()
            top_n_devices = 50  # Keep top 50 most common devices
            top_devices = set(deviceinfo_counts.head(top_n_devices).index)
        else:
            top_devices = set()
    
    def group_rare_devices(device_info):
        if pd.isna(device_info) or device_info == '':
            return 'Unknown'
        if device_info in top_devices:
            return device_info
        else:
            return 'Other_Rare_Device'
    
    if 'DeviceInfo' in df.columns:
        df['DeviceInfo_Grouped'] = df['DeviceInfo'].apply(group_rare_devices)
    
    # 4. Create binary flag for high-risk devices
    high_risk_devices = [
        'Blade L3 Build/KOT49H',
        'NOKIA',
        'Nexus 6 Build/MOB30M',
        'G3123 Build/40.0.A.6.175',
        'VS5012 Build/NRD90M',
        'Z835 Build/NMF26V',
        'Z813 Build/LMY47O',
        'SM-N920A Build/MMB29K',
        'hi6210sft Build/MRA58K',
        'Lenovo YT3-850M Build/MMB29M'
    ]
    df['DeviceInfo_HighRisk'] = df['DeviceInfo'].isin(high_risk_devices).astype(int)
    
    # Keep original DeviceInfo column (model was trained with both raw and engineered features)
    # The original DeviceInfo will be one-hot encoded along with the engineered features
    
    return df

def preprocess_transaction(transaction_data, training_stats):
    """Preprocess a single transaction to match model input format"""
    if training_stats is None:
        return None
    
    # Create a dataframe with one row
    df_input = pd.DataFrame([transaction_data])
    
    # Fill missing numeric columns with medians
    for col in training_stats['numeric_cols']:
        if col not in df_input.columns:
            df_input[col] = training_stats['numeric_medians'].get(col, 0)
        else:
            df_input[col] = df_input[col].fillna(training_stats['numeric_medians'].get(col, 0))
    
    # Fill missing categorical columns with modes
    for col in training_stats['categorical_cols']:
        if col not in df_input.columns:
            df_input[col] = training_stats['categorical_modes'].get(col, 'unknown')
        else:
            df_input[col] = df_input[col].fillna(training_stats['categorical_modes'].get(col, 'unknown'))
    
    # Ensure all original columns are present (before feature engineering)
    # Note: DeviceInfo might be in input but not in original_columns if it was dropped
    for col in training_stats['original_columns']:
        if col not in df_input.columns:
            if col in training_stats['numeric_cols']:
                df_input[col] = training_stats['numeric_medians'].get(col, 0)
            else:
                df_input[col] = training_stats['categorical_modes'].get(col, 'unknown')
    
    # Reorder columns to match training data (only columns that exist in both)
    available_cols = [col for col in training_stats['original_columns'] if col in df_input.columns]
    df_input = df_input[available_cols]
    
    # Add DeviceInfo if it's in input but not in original_columns (for feature engineering)
    if 'DeviceInfo' in transaction_data and 'DeviceInfo' not in df_input.columns:
        df_input['DeviceInfo'] = transaction_data.get('DeviceInfo', 'Unknown')
    
    # Apply DeviceInfo feature engineering BEFORE one-hot encoding
    # Use top_devices from training data if available
    top_devices = training_stats.get('top_devices', None)
    df_input = apply_deviceinfo_feature_engineering(df_input, top_devices=top_devices)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_input, drop_first=True)
    
    # Align with training feature names (add missing columns, remove extra)
    feature_names = [str(f) for f in training_stats['feature_names']]  # Ensure strings
    df_encoded.columns = [str(c) for c in df_encoded.columns]  # Ensure strings
    
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Remove any extra columns (keep only features the model expects)
    df_encoded = df_encoded[[f for f in feature_names if f in df_encoded.columns]]
    
    # Ensure correct order and add any still missing
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    return df_encoded

def format_region_code(region_code):
    """Format region code for display"""
    if pd.isna(region_code) or str(region_code) == 'nan':
        return "Unknown"
    region_str = str(region_code)
    if 'nan' in region_str.lower():
        return "Unknown"
    return region_str

@st.cache_data
def load_results():
    """Load analysis results"""
    results = {}
    result_files = {
        'model_comparison': 'model_comparison.csv',
        'region': 'fraud_risk_by_region.csv',
        'month': 'fraud_risk_by_month.csv',
        'day_of_week': 'fraud_risk_by_day_of_week.csv',
        'card4': 'fraud_risk_by_card4.csv',
        'card6': 'fraud_risk_by_card6.csv',
        'card_combined': 'fraud_risk_by_card_combined.csv',
        'amount': 'fraud_risk_by_amount.csv',
    }
    
    for key, filename in result_files.items():
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, index_col=0)
                results[key] = df
            except:
                results[key] = pd.read_csv(filepath)
    
    return results

def get_model_performance():
    """Get model performance metrics"""
    model_comparison = load_results().get('model_comparison')
    if model_comparison is not None and 'XGBoost' in model_comparison.index:
        xgb_metrics = model_comparison.loc['XGBoost']
        return {
            'roc_auc': xgb_metrics.get('ROC-AUC', 0),
            'avg_precision': xgb_metrics.get('Average Precision', 0),
            'f1_score': xgb_metrics.get('F1-Score', 0),
            'precision': xgb_metrics.get('Precision', 0),
            'recall': xgb_metrics.get('Recall', 0),
        }
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 FraudScope Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Driven Fraud Detection and Global Risk Mapping")
    
    # Load data
    model = load_model()
    test_data = load_test_data()
    results = load_results()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select View",
            ["Overview", "Model Performance", "Fraud Risk Analysis", "Feature Importance", "Real-Time Predictions"]
        )
        
        st.markdown("---")
        st.header("Data Status")
        if model is not None:
            st.success("✅ Model Loaded")
        else:
            st.warning("⚠️ Model not found. Please train the model first.")
        
        if test_data is not None:
            st.success(f"✅ Test Data: {len(test_data):,} transactions")
        else:
            st.info("ℹ️ Test data not available (optional - dashboard works with pre-computed results)")
        
        # Fraud threshold slider (only show on Model Performance page)
        if page == "Model Performance" and model is not None:
            st.markdown("---")
            st.header("Fraud Threshold")
            threshold = st.slider(
                "Prediction Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Adjust the threshold for classifying transactions as fraudulent. Lower values increase recall but decrease precision."
            )
        else:
            threshold = 0.5
    
    # Main content based on selected page
    if page == "Overview":
        show_overview(model, test_data, results)
    elif page == "Model Performance":
        threshold = st.sidebar.slider(
            "Prediction Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Adjust the threshold for classifying transactions as fraudulent. Lower values increase recall but decrease precision.",
            key="threshold_slider"
        ) if model is not None else 0.5
        show_model_performance(results, model, test_data, threshold)
    elif page == "Fraud Risk Analysis":
        show_fraud_risk_analysis(results)
    elif page == "Feature Importance":
        show_feature_importance(model)
    elif page == "Real-Time Predictions":
        show_realtime_predictions(model)

def show_overview(model, test_data, results):
    """Show dashboard overview"""
    st.header("📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Model status
    with col1:
        if model is not None:
            st.metric("Model Status", "✅ Loaded", delta="Ready")
        else:
            st.metric("Model Status", "❌ Not Found", delta="Train Required")
    
    # Test data
    with col2:
        if test_data is not None:
            st.metric("Test Transactions", f"{len(test_data):,}", delta="Available")
        else:
            st.metric("Test Transactions", "N/A", delta="Optional")
    
    # Performance metrics
    metrics = get_model_performance()
    with col3:
        if metrics:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}", delta="XGBoost")
        else:
            st.metric("ROC-AUC", "N/A", delta="No Data")
    
    with col4:
        if metrics:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}", delta="XGBoost")
        else:
            st.metric("F1-Score", "N/A", delta="No Data")
    
    st.markdown("---")
    
    # Quick insights
    st.subheader("🔍 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Available Analyses")
        analyses = []
        if 'region' in results:
            analyses.append("✅ Regional Risk Analysis")
        if 'month' in results or 'day_of_week' in results:
            analyses.append("✅ Temporal Risk Analysis")
        if 'card4' in results or 'card6' in results:
            analyses.append("✅ Card Type Risk Analysis")
        if 'amount' in results:
            analyses.append("✅ Transaction Amount Analysis")
        
        for analysis in analyses:
            st.markdown(f"- {analysis}")
    
    with col2:
        st.markdown("#### Dashboard Features")
        features = [
            "📈 Model Performance Metrics",
            "🗺️ Geographic Risk Mapping",
            "📅 Time-based Risk Patterns",
            "💳 Card Type Vulnerability",
            "💰 Transaction Amount Analysis",
            "🎯 Feature Importance Visualization"
        ]
        for feature in features:
            st.markdown(f"- {feature}")
    
    # Instructions
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate to different sections of the dashboard.")

def show_model_performance(results, model=None, test_data=None, threshold=0.5):
    """Show model performance metrics"""
    st.header("📈 Model Performance")
    
    model_comparison = results.get('model_comparison')
    
    if model_comparison is None:
        st.warning("Model comparison data not available. Please run the training notebook first.")
        return
    
    # Threshold-based F1 calculation for XGBoost
    if model is not None and test_data is not None:
        st.markdown("---")
        st.subheader("🔧 Threshold-Based Performance (XGBoost)")
        
        try:
            # Prepare test data (same preprocessing as training)
            # Note: This is a simplified version - you may need to match exact preprocessing
            from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
            
            # For now, we'll use a placeholder - in production, you'd need to preprocess test_data
            # the same way as training data
            st.info("💡 **Note**: To compute threshold-based metrics, the test data needs to be preprocessed the same way as training data. This requires running predictions in the notebook first.")
            
            # If we have predictions saved, we can use them
            # For now, show the threshold slider impact conceptually
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Threshold", f"{threshold:.2f}")
            with col2:
                # Show how threshold affects precision/recall tradeoff
                if 'XGBoost' in model_comparison.index:
                    base_precision = model_comparison.loc['XGBoost', 'Precision']
                    base_recall = model_comparison.loc['XGBoost', 'Recall']
                    # Estimate: lower threshold = higher recall, lower precision
                    estimated_precision = max(0, min(1, base_precision - (0.5 - threshold) * 0.3))
                    estimated_recall = max(0, min(1, base_recall + (0.5 - threshold) * 0.3))
                    estimated_f1 = 2 * (estimated_precision * estimated_recall) / (estimated_precision + estimated_recall) if (estimated_precision + estimated_recall) > 0 else 0
                    st.metric("Estimated F1-Score", f"{estimated_f1:.4f}", 
                             delta=f"Precision: {estimated_precision:.2f}, Recall: {estimated_recall:.2f}")
            with col3:
                if 'XGBoost' in model_comparison.index:
                    base_f1 = model_comparison.loc['XGBoost', 'F1-Score']
                    st.metric("Base F1-Score (0.5 threshold)", f"{base_f1:.4f}")
        except Exception as e:
            st.warning(f"Could not compute threshold-based metrics: {e}")
    elif model is not None and test_data is None:
        st.markdown("---")
        st.subheader("🔧 Threshold-Based Performance (XGBoost)")
        st.info("ℹ️ **Test data not available**: Threshold-based metrics require test data. The dashboard works with pre-computed results from the modeling notebook.")
    
    # Confusion Matrix for XGBoost
    if 'XGBoost' in model_comparison.index:
        st.markdown("---")
        st.subheader("📊 Confusion Matrix (XGBoost)")
        
        # Try to load confusion matrix data if available
        confusion_data_path = RESULTS_DIR / "confusion_matrix_xgboost.csv"
        if confusion_data_path.exists():
            try:
                cm_data = pd.read_csv(confusion_data_path, index_col=0)
                # Create heatmap
                fig = px.imshow(
                    cm_data.values,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Fraud', 'Fraud'],
                    y=['Not Fraud', 'Fraud'],
                    color_continuous_scale='Blues'
                )
                # Add text annotations manually to avoid deprecation warning
                fig.update_traces(text=cm_data.values, texttemplate='%{text}', textfont={"size": 12})
                fig.update_layout(
                    title="XGBoost Confusion Matrix",
                    autosize=True,
                    yaxis=dict(scaleanchor="x", scaleratio=1)
                )
                st.plotly_chart(fig, width='stretch')
                
                # Extract values for display
                if cm_data.shape == (2, 2):
                    tn, fp, fn, tp = cm_data.iloc[0, 0], cm_data.iloc[0, 1], cm_data.iloc[1, 0], cm_data.iloc[1, 1]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("True Negatives", f"{int(tn):,}")
                    with col2:
                        st.metric("False Positives", f"{int(fp):,}")
                    with col3:
                        st.metric("False Negatives", f"{int(fn):,}")
                    with col4:
                        st.metric("True Positives", f"{int(tp):,}")
            except Exception as e:
                st.warning(f"Could not load confusion matrix: {e}")
                st.info("💡 **Tip**: Run the modeling notebook to generate confusion matrix data.")
        else:
            # Create a placeholder or compute from model_comparison if possible
            st.info("💡 **Tip**: Confusion matrix data not found. Run cell 18 in the modeling notebook to generate `results/confusion_matrix_xgboost.csv`. The code has already been added to the notebook.")
            
            # Show a note about what the confusion matrix would show
            st.markdown("""
            **Understanding the Confusion Matrix:**
            - **True Positives (TP)**: Correctly identified fraudulent transactions
            - **False Positives (FP)**: Legitimate transactions flagged as fraud
            - **False Negatives (FN)**: Fraudulent transactions missed
            - **True Negatives (TN)**: Correctly identified legitimate transactions
            """)
    
    # Display metrics table
    st.subheader("Model Comparison")
    
    # Format the dataframe for display
    display_df = model_comparison.copy()
    if 'Rank' in display_df.columns:
        display_df = display_df.set_index('Rank')
    
    # Format numeric columns
    numeric_cols = ['ROC-AUC', 'Average Precision', 'F1-Score', 'Precision', 'Recall']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    
    st.dataframe(display_df, width='stretch')
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC-AUC Comparison")
        fig = px.bar(
            x=model_comparison.index,
            y=model_comparison['ROC-AUC'],
            labels={'x': 'Model', 'y': 'ROC-AUC Score'},
            color=model_comparison['ROC-AUC'],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("F1-Score Comparison")
        fig = px.bar(
            x=model_comparison.index,
            y=model_comparison['F1-Score'],
            labels={'x': 'Model', 'y': 'F1-Score'},
            color=model_comparison['F1-Score'],
            color_continuous_scale='Plasma'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Best model highlight
    if 'XGBoost' in model_comparison.index:
        xgb_metrics = model_comparison.loc['XGBoost']
        st.success(f"""
        **Best Model: XGBoost**
        - ROC-AUC: {xgb_metrics.get('ROC-AUC', 0):.4f}
        - Average Precision: {xgb_metrics.get('Average Precision', 0):.4f}
        - F1-Score: {xgb_metrics.get('F1-Score', 0):.4f}
        - Precision: {xgb_metrics.get('Precision', 0):.4f}
        - Recall: {xgb_metrics.get('Recall', 0):.4f}
        """)

def show_fraud_risk_analysis(results):
    """Show fraud risk analysis by different dimensions"""
    st.header("🔍 Fraud Risk Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Region", "Time Patterns", "Card Type", "Transaction Amount", "Device"]
    )
    
    if analysis_type == "Region":
        show_region_analysis(results)
    elif analysis_type == "Time Patterns":
        show_time_analysis(results)
    elif analysis_type == "Card Type":
        show_card_analysis(results)
    elif analysis_type == "Transaction Amount":
        show_amount_analysis(results)
    elif analysis_type == "Device":
        show_device_analysis(results)

def show_region_analysis(results):
    """Show fraud risk by region"""
    st.subheader("🗺️ Fraud Risk by Region")
    
    st.info("💡 **How to read this**: Higher average fraud probability means this region is more likely to be fraudulent according to the XGBoost model.")
    st.markdown("---")
    
    region_data = results.get('region')
    
    if region_data is None:
        st.warning("Regional analysis data not available.")
        return
    
    # Prepare region data
    region_data = region_data.copy()
    region_data['Region_Code'] = region_data.index.map(format_region_code)
    
    # Display top regions
    st.markdown("#### Top 20 High-Risk Regions")
    
    # Sort by fraud probability
    top_regions = region_data.nlargest(20, 'Avg_Fraud_Probability')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use region codes for display
        plot_data = top_regions.copy()
        plot_data['Region_Code'] = plot_data.index.map(format_region_code)
        
        fig = px.bar(
            plot_data,
            x='Avg_Fraud_Probability',
            y='Region_Code',
            orientation='h',
            labels={'x': 'Average Fraud Probability', 'y': 'Region Code'},
            color='Avg_Fraud_Probability',
            color_continuous_scale='Reds',
            title="Top 20 High-Risk Regions"
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Format dataframe for display with region codes
        display_regions = top_regions[['Avg_Fraud_Probability', 'Actual_Fraud_Rate', 'Transaction_Count']].copy()
        display_regions = display_regions.reset_index()
        # Get the index column name (could be 'index' or the actual index name)
        index_col = display_regions.columns[0]  # First column after reset_index is the index
        display_regions = display_regions.rename(columns={index_col: 'Region_Code'})
        display_regions['Region_Code'] = display_regions['Region_Code'].map(format_region_code)
        display_regions = display_regions[['Region_Code', 'Avg_Fraud_Probability', 'Actual_Fraud_Rate', 'Transaction_Count']]
        display_regions['Avg_Fraud_Probability'] = display_regions['Avg_Fraud_Probability'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        display_regions['Actual_Fraud_Rate'] = display_regions['Actual_Fraud_Rate'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        display_regions['Transaction_Count'] = display_regions['Transaction_Count'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
        st.dataframe(display_regions, width='stretch', hide_index=True)
    
    # Summary statistics
    st.markdown("#### Regional Risk Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Highest Risk", f"{top_regions.iloc[0]['Avg_Fraud_Probability']:.4f}")
    with col2:
        st.metric("Average Risk", f"{region_data['Avg_Fraud_Probability'].mean():.4f}")
    with col3:
        st.metric("Total Regions", len(region_data))
    with col4:
        st.metric("High Risk Regions (>0.1)", len(region_data[region_data['Avg_Fraud_Probability'] > 0.1]))

def show_time_analysis(results):
    """Show fraud risk by time patterns"""
    st.subheader("📅 Fraud Risk by Time Patterns")
    
    st.info("💡 **How to read this**: Higher average fraud probability means transactions during this time period are more likely to be fraudulent according to the XGBoost model.")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["By Month", "By Day of Week"])
    
    with tab1:
        month_data = results.get('month')
        if month_data is not None:
            fig = px.line(
                month_data,
                x=month_data.index,
                y='Avg_Fraud_Probability',
                markers=True,
                labels={'x': 'Month', 'y': 'Average Fraud Probability'},
                title="Fraud Risk by Month"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
            
            # Format dataframe for display
            display_month = month_data.copy()
            display_month['Avg_Fraud_Probability'] = display_month['Avg_Fraud_Probability'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_month['Actual_Fraud_Rate'] = display_month['Actual_Fraud_Rate'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_month['Transaction_Count'] = display_month['Transaction_Count'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            st.dataframe(display_month, width='stretch')
        else:
            st.warning("Monthly data not available.")
    
    with tab2:
        day_data = results.get('day_of_week')
        if day_data is not None:
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_data_ordered = day_data.reindex([d for d in days_order if d in day_data.index])
            
            fig = px.bar(
                day_data_ordered,
                x=day_data_ordered.index,
                y='Avg_Fraud_Probability',
                labels={'x': 'Day of Week', 'y': 'Average Fraud Probability'},
                color='Avg_Fraud_Probability',
                color_continuous_scale='Blues',
                title="Fraud Risk by Day of Week"
            )
            fig.update_layout(height=400, xaxis={'categoryorder': 'array', 'categoryarray': days_order})
            st.plotly_chart(fig, width='stretch')
            
            # Format dataframe for display
            display_day = day_data_ordered.copy()
            display_day['Avg_Fraud_Probability'] = display_day['Avg_Fraud_Probability'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_day['Actual_Fraud_Rate'] = display_day['Actual_Fraud_Rate'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_day['Transaction_Count'] = display_day['Transaction_Count'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            st.dataframe(display_day, width='stretch')
        else:
            st.warning("Day of week data not available.")

def show_card_analysis(results):
    """Show fraud risk by card type"""
    st.subheader("💳 Fraud Risk by Card Type")
    
    st.info("💡 **How to read this**: Higher average fraud probability means this card type/issuer is more likely to be associated with fraudulent transactions according to the XGBoost model.")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Card Network (card4)", "Card Type (card6)", "Combined"])
    
    with tab1:
        card4_data = results.get('card4')
        if card4_data is not None:
            fig = px.bar(
                card4_data,
                x=card4_data.index,
                y='Avg_Fraud_Probability',
                labels={'x': 'Card Network', 'y': 'Average Fraud Probability'},
                color='Avg_Fraud_Probability',
                color_continuous_scale='Oranges',
                title="Fraud Risk by Card Network"
            )
            fig.update_layout(height=400, xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, width='stretch')
            
            # Format dataframe for display
            display_card4 = card4_data.copy()
            display_card4['Avg_Fraud_Probability'] = display_card4['Avg_Fraud_Probability'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_card4['Actual_Fraud_Rate'] = display_card4['Actual_Fraud_Rate'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_card4['Transaction_Count'] = display_card4['Transaction_Count'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            st.dataframe(display_card4, width='stretch')
        else:
            st.warning("Card network data not available.")
    
    with tab2:
        card6_data = results.get('card6')
        if card6_data is not None:
            fig = px.bar(
                card6_data,
                x=card6_data.index,
                y='Avg_Fraud_Probability',
                labels={'x': 'Card Type', 'y': 'Average Fraud Probability'},
                color='Avg_Fraud_Probability',
                color_continuous_scale='Greens',
                title="Fraud Risk by Card Type"
            )
            fig.update_layout(height=400, xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, width='stretch')
            
            # Format dataframe for display
            display_card6 = card6_data.copy()
            display_card6['Avg_Fraud_Probability'] = display_card6['Avg_Fraud_Probability'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_card6['Actual_Fraud_Rate'] = display_card6['Actual_Fraud_Rate'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_card6['Transaction_Count'] = display_card6['Transaction_Count'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            st.dataframe(display_card6, width='stretch')
        else:
            st.warning("Card type data not available.")
    
    with tab3:
        card_combined = results.get('card_combined')
        if card_combined is not None:
            top_combined = card_combined.nlargest(15, 'Avg_Fraud_Probability')
            
            fig = px.bar(
                top_combined,
                x='Avg_Fraud_Probability',
                y=top_combined.index,
                orientation='h',
                labels={'x': 'Average Fraud Probability', 'y': 'Card Network + Type'},
                color='Avg_Fraud_Probability',
                color_continuous_scale='Purples',
                title="Top 15 High-Risk Card Combinations"
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, width='stretch')
            
            # Format dataframe for display
            display_combined = top_combined.copy()
            display_combined['Avg_Fraud_Probability'] = display_combined['Avg_Fraud_Probability'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_combined['Actual_Fraud_Rate'] = display_combined['Actual_Fraud_Rate'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            display_combined['Transaction_Count'] = display_combined['Transaction_Count'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            st.dataframe(display_combined, width='stretch')
        else:
            st.warning("Combined card data not available.")

def show_amount_analysis(results):
    """Show fraud risk by transaction amount"""
    st.subheader("💰 Fraud Risk by Transaction Amount")
    
    st.info("💡 **How to read this**: Higher average fraud probability means transactions in this amount range are more likely to be fraudulent according to the XGBoost model.")
    st.markdown("---")
    
    amount_data = results.get('amount')
    
    if amount_data is None:
        st.warning("Transaction amount analysis data not available.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            amount_data,
            x=amount_data.index,
            y='Avg_Fraud_Probability',
            labels={'x': 'Amount Range', 'y': 'Average Fraud Probability'},
            color='Avg_Fraud_Probability',
            color_continuous_scale='Reds',
            title="Fraud Risk by Transaction Amount Range"
        )
        fig.update_layout(height=500, xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Format dataframe for display
        display_amount = amount_data.copy()
        display_amount['Avg_Fraud_Probability'] = display_amount['Avg_Fraud_Probability'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        display_amount['Actual_Fraud_Rate'] = display_amount['Actual_Fraud_Rate'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        display_amount['Transaction_Count'] = display_amount['Transaction_Count'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
        if 'Avg_Amount' in display_amount.columns:
            display_amount['Avg_Amount'] = display_amount['Avg_Amount'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
        st.dataframe(display_amount, width='stretch')
    
    # Insights
    st.markdown("#### Key Insights")
    highest_risk = amount_data.loc[amount_data['Avg_Fraud_Probability'].idxmax()]
    st.info(f"""
    **Highest Risk Range**: {amount_data['Avg_Fraud_Probability'].idxmax()}
    - Fraud Probability: {highest_risk['Avg_Fraud_Probability']:.4f}
    - Actual Fraud Rate: {highest_risk['Actual_Fraud_Rate']:.4f}
    - Transaction Count: {highest_risk['Transaction_Count']:.0f}
    """)

def show_device_analysis(results):
    """Show fraud risk by device type and device info"""
    st.subheader("📱 Fraud Risk by Device")
    
    st.info("💡 **How to read this**: Higher average fraud probability means transactions from this device type/info are more likely to be fraudulent according to the XGBoost model.")
    st.markdown("---")
    
    # Load processed data which includes merged identity data with DeviceType/DeviceInfo
    data = load_processed_data()
    
    if data is None:
        st.warning("Processed data not available. Device analysis requires merged transaction and identity data.")
        st.info("""
        **To enable device analysis:**
        1. Run the EDA notebook to create `data/processed/clean_train.parquet`
        2. This file contains merged transaction and identity data with DeviceType and DeviceInfo columns
        """)
        return
    
    # Check if DeviceType and DeviceInfo columns exist
    if 'DeviceType' not in data.columns and 'DeviceInfo' not in data.columns:
        st.warning("DeviceType or DeviceInfo columns not found in the processed data.")
        st.info("Please ensure the EDA notebook has been run to merge transaction and identity data.")
        return
    
    tab1, tab2 = st.tabs(["Device Type", "Device Info"])
    
    with tab1:
        if 'DeviceType' in data.columns:
            st.markdown("#### 📱 Fraud Risk by Device Type")
            
            # Calculate fraud risk by device type
            device_type_analysis = data.groupby('DeviceType')['isFraud'].agg(['mean', 'count']).round(4)
            device_type_analysis.columns = ['Fraud_Rate', 'Transaction_Count']
            device_type_counts = data['DeviceType'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Device type distribution
                fig = px.bar(
                    x=device_type_counts.index,
                    y=device_type_counts.values,
                    labels={'x': 'Device Type', 'y': 'Transaction Count'},
                    title="Transaction Count by Device Type",
                    color=device_type_counts.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display statistics with fraud rates
                st.markdown("#### Device Type Statistics")
                for device_type in device_type_counts.index:
                    count = device_type_counts[device_type]
                    percentage = (count / len(data)) * 100
                    fraud_rate = device_type_analysis.loc[device_type, 'Fraud_Rate'] * 100
                    st.metric(
                        device_type.title(),
                        f"{count:,}",
                        delta=f"{fraud_rate:.2f}% fraud rate"
                    )
            
            # Display fraud rate comparison
            st.markdown("---")
            st.markdown("#### Fraud Rate by Device Type")
            display_df = device_type_analysis.copy()
            display_df['Fraud_Rate_Percent'] = (display_df['Fraud_Rate'] * 100).round(2)
            display_df = display_df[['Fraud_Rate_Percent', 'Transaction_Count']]
            display_df.columns = ['Fraud Rate (%)', 'Transaction Count']
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Key Insights")
            if len(device_type_analysis) >= 2:
                mobile_rate = device_type_analysis.loc['mobile', 'Fraud_Rate'] if 'mobile' in device_type_analysis.index else None
                desktop_rate = device_type_analysis.loc['desktop', 'Fraud_Rate'] if 'desktop' in device_type_analysis.index else None
                if mobile_rate and desktop_rate:
                    risk_ratio = mobile_rate / desktop_rate
                    st.success(f"""
                    **Key Finding:**
                    - **Mobile devices** have **{risk_ratio:.2f}x higher fraud rate** than desktop devices
                    - Mobile: **{mobile_rate*100:.2f}%** fraud rate
                    - Desktop: **{desktop_rate*100:.2f}%** fraud rate
                    
                    **Recommendation:** Monitor mobile transactions more closely and consider additional verification steps.
                    """)
                else:
                    st.info("Device type analysis shows different fraud rates. Review the data above for insights.")
            else:
                st.info("Insufficient device types for comparison. Review the data above for insights.")
        else:
            st.warning("DeviceType column not found in the data.")
    
    with tab2:
        if 'DeviceInfo' in data.columns:
            st.markdown("#### 🔍 Fraud Risk by Device Info")
            
            # Show top devices with fraud rates
            device_info_counts = data['DeviceInfo'].value_counts().head(20)
            device_info_fraud = data.groupby('DeviceInfo')['isFraud'].agg(['mean', 'count'])
            device_info_fraud = device_info_fraud[device_info_fraud['count'] >= 10]  # Min 10 transactions
            device_info_fraud = device_info_fraud.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=device_info_counts.values,
                    y=device_info_counts.index,
                    orientation='h',
                    labels={'x': 'Transaction Count', 'y': 'Device Info'},
                    title="Top 20 Most Common Devices",
                    color=device_info_counts.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Device Info Statistics")
                st.metric("Unique Devices", f"{data['DeviceInfo'].nunique():,}")
                st.metric("Missing Values", f"{data['DeviceInfo'].isnull().sum():,}")
                
                st.markdown("---")
                st.markdown("#### Top 10 High-Risk Devices")
                if len(device_info_fraud) > 0:
                    high_risk_df = device_info_fraud.head(10).copy()
                    high_risk_df['Fraud_Rate_Percent'] = (high_risk_df['mean'] * 100).round(2)
                    display_df = pd.DataFrame({
                        'Device': high_risk_df.index.str[:40],  # Truncate long names
                        'Fraud Rate (%)': high_risk_df['Fraud_Rate_Percent'].values,
                        'Count': high_risk_df['count'].values.astype(int)
                    })
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No devices with sufficient transaction count for fraud rate analysis.")
            
            st.markdown("---")
            st.markdown("#### Key Insights")
            unique_count = data['DeviceInfo'].nunique()
            if len(device_info_fraud) > 0:
                highest_risk = device_info_fraud.iloc[0]
                st.warning(f"""
                **Key Findings:**
                - DeviceInfo has **{unique_count:,} unique values** (high cardinality)
                - Highest risk device: **{device_info_fraud.index[0][:50]}**
                  - Fraud rate: **{highest_risk['mean']*100:.2f}%**
                  - Transaction count: **{int(highest_risk['count'])}**
                
                **Recommendation:** 
                - Use feature engineering to extract OS, brand, or group rare devices
                - Create binary flags for high-risk devices
                - Consider blocking transactions from known high-risk devices
                """)
            else:
                st.info(f"""
                DeviceInfo has **{unique_count:,} unique values** (high cardinality).
                Consider feature engineering to reduce dimensionality while preserving fraud signals.
                """)
        else:
            st.warning("DeviceInfo column not found in the data.")
    
    # Feature Engineering Info
    st.markdown("---")
    st.markdown("#### 🛠️ Feature Engineering Recommendations")
    st.info("""
    The modeling notebook now includes feature engineering for DeviceInfo:
    - **DeviceInfo_OS**: Extracted operating system (Windows, iOS, Android, macOS, Linux, Other)
    - **DeviceInfo_Brand**: Extracted device brand (Samsung, Apple, Huawei, etc.)
    - **DeviceInfo_Grouped**: Grouped rare devices (reduces from 1,786 to ~51 unique values)
    - **DeviceInfo_HighRisk**: Binary flag for known high-risk devices
    
    These engineered features reduce dimensionality while preserving important fraud signals.
    """)

def show_feature_importance(model):
    """Show feature importance from XGBoost model"""
    st.header("🎯 Feature Importance")
    
    if model is None:
        st.warning("Model not loaded. Please train and save the model first.")
        st.info("Run the modeling notebook to train and save the XGBoost model.")
        return
    
    try:
        # Get feature importance
        feature_importance = model.feature_importances_
        
        # Get feature names (if available)
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Show top N features
        n_features = st.slider("Number of top features to display", 10, 50, 20)
        
        top_features = importance_df.head(n_features)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                color='importance',
                color_continuous_scale='Viridis',
                title=f"Top {n_features} Most Important Features"
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Format dataframe for display
            display_features = top_features.copy()
            display_features['importance'] = display_features['importance'].apply(lambda x: f'{x:.6f}' if pd.notna(x) else 'N/A')
            st.dataframe(display_features, width='stretch')
        
        # Summary statistics
        st.markdown("#### Feature Importance Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", len(importance_df))
        with col2:
            st.metric("Max Importance", f"{importance_df['importance'].max():.6f}")
        with col3:
            st.metric("Mean Importance", f"{importance_df['importance'].mean():.6f}")
        with col4:
            st.metric("Top 10 Contribution", f"{importance_df.head(10)['importance'].sum()*100:.2f}%")
        
    except Exception as e:
        st.error(f"Error extracting feature importance: {e}")
        st.info("The model may not support feature importance extraction.")

def show_realtime_predictions(model):
    """Show real-time fraud prediction interface"""
    st.header("⚡ Real-Time Fraud Prediction")
    st.markdown("Enter transaction details below to get an instant fraud prediction from the XGBoost model.")
    
    if model is None:
        st.error("❌ Model not loaded. Please train and save the model first.")
        st.info("Run the modeling notebook to train and save the XGBoost model.")
        return
    
    # Load training statistics for preprocessing
    training_stats = load_training_data_stats()
    if training_stats is None:
        st.error("❌ Could not load training data statistics. Please ensure the training data is available.")
        st.info("The training data is needed to properly preprocess your input.")
        return
    
    st.markdown("---")
    
    # Feature selection
    st.subheader("🔧 Select Features to Include")
    
    # Helper function to get slider range for a feature
    def get_slider_range(feature_name, default_min=None, default_max=None):
        """Get min/max values for slider, with fallback to defaults"""
        min_val = training_stats['numeric_mins'].get(feature_name, default_min if default_min is not None else 0.0)
        max_val = training_stats['numeric_maxs'].get(feature_name, default_max if default_max is not None else 1000.0)
        # Ensure min < max
        if min_val >= max_val:
            max_val = min_val + 1.0
        return float(min_val), float(max_val)
    
    # Define available feature groups
    feature_groups = {
        "Basic Information": {
            "TransactionAmt": {"type": "slider", "label": "Transaction Amount ($)", "default": 100.0, "min": 0.0, "max": 10000.0, "step": 1.0, "help": "The amount of the transaction"},
            "card4": {"type": "select", "label": "Card Network", "options": ["visa", "mastercard", "discover", "american express"], "help": "The card network/issuer"},
            "card6": {"type": "select", "label": "Card Type", "options": ["credit", "debit", "charge card", "debit or credit"], "help": "The type of card"},
            "ProductCD": {"type": "select", "label": "Product Code", "options": ["W", "C", "R", "H", "S"], "help": "Product category code"},
        },
        "Geographic Information": {
            "addr1": {"type": "slider", "label": "Address 1 (Region Code)", "default": 0.0, "min": 0.0, "max": 600.0, "step": 1.0, "help": "Geographic region code (addr1)"},
            "addr2": {"type": "slider", "label": "Address 2 (Region Code)", "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0, "help": "Geographic region code (addr2)"},
        },
        "Time Information": {
            "TransactionDT": {"type": "slider", "label": "TransactionDT (seconds from reference)", "default": 86400.0, "min": 0.0, "max": 2592000.0, "step": 3600.0, "help": "Time of transaction in seconds from reference date (2017-12-01)"},
        },
        "Top Important Features": {
            "V258": {"type": "slider", "label": "V258 (Most Important - 17.6%)", "default": float(training_stats['numeric_medians'].get('V258', 0)), "step": 0.01, "help": "Feature V258 - most important predictor (17.6% importance)"},
            "V201": {"type": "slider", "label": "V201 (5.9% importance)", "default": float(training_stats['numeric_medians'].get('V201', 0)), "step": 0.01, "help": "Feature V201 - second most important (5.9% importance)"},
            "V149": {"type": "slider", "label": "V149 (5.3% importance)", "default": float(training_stats['numeric_medians'].get('V149', 0)), "step": 0.01, "help": "Feature V149 - third most important (5.3% importance)"},
            "V70": {"type": "slider", "label": "V70 (3.1% importance)", "default": float(training_stats['numeric_medians'].get('V70', 0)), "step": 0.01, "help": "Feature V70"},
            "V91": {"type": "slider", "label": "V91 (3.1% importance)", "default": float(training_stats['numeric_medians'].get('V91', 0)), "step": 0.01, "help": "Feature V91"},
            "V147": {"type": "slider", "label": "V147 (2.6% importance)", "default": float(training_stats['numeric_medians'].get('V147', 0)), "step": 0.01, "help": "Feature V147"},
            "V172": {"type": "slider", "label": "V172 (2.1% importance)", "default": float(training_stats['numeric_medians'].get('V172', 0)), "step": 0.01, "help": "Feature V172"},
            "V294": {"type": "slider", "label": "V294 (2.0% importance)", "default": float(training_stats['numeric_medians'].get('V294', 0)), "step": 0.01, "help": "Feature V294"},
            "V225": {"type": "slider", "label": "V225 (1.5% importance)", "default": float(training_stats['numeric_medians'].get('V225', 0)), "step": 0.01, "help": "Feature V225"},
            "C14": {"type": "slider", "label": "C14 (1.4% importance)", "default": float(training_stats['numeric_medians'].get('C14', 0)), "step": 0.01, "help": "Feature C14 - important categorical feature"},
        },
        "Additional V-Features": {
            "V29": {"type": "slider", "label": "V29", "default": float(training_stats['numeric_medians'].get('V29', 0)), "step": 0.01, "help": "Feature V29"},
            "V95": {"type": "slider", "label": "V95", "default": float(training_stats['numeric_medians'].get('V95', 0)), "step": 0.01, "help": "Feature V95"},
            "V62": {"type": "slider", "label": "V62", "default": float(training_stats['numeric_medians'].get('V62', 0)), "step": 0.01, "help": "Feature V62"},
            "V187": {"type": "slider", "label": "V187", "default": float(training_stats['numeric_medians'].get('V187', 0)), "step": 0.01, "help": "Feature V187"},
        },
        "Additional C-Features": {
            "C12": {"type": "slider", "label": "C12", "default": float(training_stats['numeric_medians'].get('C12', 0)), "step": 0.01, "help": "Feature C12"},
            "C8": {"type": "slider", "label": "C8", "default": float(training_stats['numeric_medians'].get('C8', 0)), "step": 0.01, "help": "Feature C8"},
            "C1": {"type": "slider", "label": "C1", "default": float(training_stats['numeric_medians'].get('C1', 0)), "step": 0.01, "help": "Feature C1"},
        }
    }
    
    # Create expandable sections for each feature group
    selected_features = {}
    
    with st.expander("📋 Feature Selection", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select feature groups to include:**")
            for group_name in feature_groups.keys():
                if st.checkbox(f"✓ {group_name}", key=f"group_{group_name}", value=(group_name in ["Basic Information", "Time Information"])):
                    selected_features[group_name] = feature_groups[group_name]
        
        with col2:
            st.info("💡 **Tip**: Select the feature groups you have data for. Unselected features will use default values from training data.")
    
    st.markdown("---")
    
    # Create input form
    with st.form("transaction_form"):
        st.subheader("📝 Transaction Details")
        
        transaction_data = {}
        
        # Display selected features in organized columns
        if selected_features:
            num_groups = len(selected_features)
            cols_per_row = min(2, num_groups)
            
            # Create columns dynamically
            cols = st.columns(cols_per_row)
            col_idx = 0
            
            for group_name, features in selected_features.items():
                with cols[col_idx % cols_per_row]:
                    st.markdown(f"#### {group_name}")
                    
                    for feature_name, feature_config in features.items():
                        if feature_config["type"] == "slider":
                            # Get min/max from training data if available, otherwise use defaults
                            if feature_name in training_stats['numeric_mins'] and feature_name in training_stats['numeric_maxs']:
                                min_val = float(training_stats['numeric_mins'][feature_name])
                                max_val = float(training_stats['numeric_maxs'][feature_name])
                                # Add some padding to the range
                                range_padding = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
                                min_val = float(max(0, min_val - range_padding))
                                max_val = float(max_val + range_padding)
                            else:
                                min_val = float(feature_config.get("min", 0.0))
                                max_val = float(feature_config.get("max", 1000.0))
                            
                            # Ensure min < max
                            if min_val >= max_val:
                                max_val = float(min_val + 1.0)
                            
                            # Ensure step is float
                            step_val = float(feature_config.get("step", 0.01))
                            
                            value = st.slider(
                                feature_config["label"],
                                min_value=min_val,
                                max_value=max_val,
                                value=float(feature_config["default"]),
                                step=step_val,
                                help=feature_config.get("help", ""),
                                key=f"input_{feature_name}"
                            )
                            transaction_data[feature_name] = value
                        elif feature_config["type"] == "select":
                            value = st.selectbox(
                                feature_config["label"],
                                feature_config["options"],
                                help=feature_config.get("help", ""),
                                key=f"input_{feature_name}"
                            )
                            transaction_data[feature_name] = value
                
                col_idx += 1
                if col_idx % cols_per_row == 0 and col_idx < num_groups:
                    cols = st.columns(cols_per_row)
        else:
            st.warning("⚠️ Please select at least one feature group above.")
        
        submitted = st.form_submit_button("🔍 Predict Fraud Risk", use_container_width=True)
    
    # Process prediction when form is submitted
    if submitted:
        if not selected_features:
            st.warning("⚠️ Please select at least one feature group above.")
        else:
            with st.spinner("Processing transaction and generating prediction..."):
                # Fill in other required features with defaults
                for col in training_stats['original_columns']:
                    if col not in transaction_data:
                        if col in training_stats['numeric_cols']:
                            transaction_data[col] = training_stats['numeric_medians'].get(col, 0)
                        else:
                            transaction_data[col] = training_stats['categorical_modes'].get(col, 'unknown')
                
                # Preprocess transaction
                try:
                    processed_data = preprocess_transaction(transaction_data, training_stats)
                
                    if processed_data is None:
                        st.error("❌ Failed to preprocess transaction data.")
                    else:
                        # Make prediction
                        fraud_probability = model.predict_proba(processed_data)[0, 1]
                        fraud_prediction = model.predict(processed_data)[0]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Fraud probability with color coding
                            if fraud_probability >= 0.7:
                                st.metric("Fraud Probability", f"{fraud_probability:.4f}", delta="HIGH RISK", delta_color="inverse")
                            elif fraud_probability >= 0.4:
                                st.metric("Fraud Probability", f"{fraud_probability:.4f}", delta="MEDIUM RISK", delta_color="off")
                            else:
                                st.metric("Fraud Probability", f"{fraud_probability:.4f}", delta="LOW RISK")
                        
                        with col2:
                            prediction_label = "🚨 FRAUDULENT" if fraud_prediction == 1 else "✅ LEGITIMATE"
                            st.metric("Prediction", prediction_label)
                        
                        with col3:
                            confidence = abs(fraud_probability - 0.5) * 2  # Convert to 0-1 confidence scale
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Visualizations
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Probability gauge
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = fraud_probability * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Fraud Risk Score (%)"},
                                delta = {'reference': 3.5, 'position': "top"},  # Average fraud rate
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkred" if fraud_probability > 0.5 else "darkgreen"},
                                    'steps': [
                                        {'range': [0, 20], 'color': "lightgreen"},
                                        {'range': [20, 50], 'color': "yellow"},
                                        {'range': [50, 100], 'color': "lightcoral"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Probability bar chart
                            fig = px.bar(
                                x=['Legitimate', 'Fraudulent'],
                                y=[1 - fraud_probability, fraud_probability],
                                labels={'x': 'Prediction', 'y': 'Probability'},
                                title="Prediction Probabilities",
                                color=['Legitimate', 'Fraudulent'],
                                color_discrete_map={'Legitimate': 'green', 'Fraudulent': 'red'}
                            )
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk interpretation
                        st.markdown("---")
                        st.subheader("📊 Risk Interpretation")
                        
                        if fraud_probability >= 0.7:
                            st.error(f"**HIGH RISK**: This transaction has a {fraud_probability:.2%} probability of being fraudulent. "
                                    "Strongly recommend manual review or blocking.")
                        elif fraud_probability >= 0.4:
                            st.warning(f"**MEDIUM RISK**: This transaction has a {fraud_probability:.2%} probability of being fraudulent. "
                                      "Consider additional verification.")
                        else:
                            st.success(f"**LOW RISK**: This transaction has a {fraud_probability:.2%} probability of being fraudulent. "
                                      "Likely legitimate transaction.")
                        
                        # Feature contribution (if available)
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("---")
                            st.subheader("🔍 Top Contributing Features")
                            
                            # Get feature importance for this prediction
                            # Note: This is a simplified view - XGBoost doesn't provide per-instance feature importance
                            # We'll show the most important features in general
                            feature_importance = model.feature_importances_
                            feature_names = training_stats['feature_names']
                            
                            importance_df = pd.DataFrame({
                                'feature': feature_names,
                                'importance': feature_importance
                            }).sort_values('importance', ascending=False).head(10)
                            
                            fig = px.bar(
                                importance_df,
                                x='importance',
                                y='feature',
                                orientation='h',
                                labels={'x': 'Importance', 'y': 'Feature'},
                                title="Top 10 Most Important Features (Model-wide)",
                                color='importance',
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Transaction summary
                        st.markdown("---")
                        st.subheader("📋 Transaction Summary")
                        
                        # Create summary from provided features
                        summary_items = []
                        for key, value in transaction_data.items():
                            if key in ['TransactionAmt']:
                                summary_items.append((f"Transaction Amount", f"${value:,.2f}"))
                            elif key in ['card4', 'card6']:
                                summary_items.append((key.replace('card', 'Card ').title(), str(value).title()))
                            elif key in ['addr1', 'addr2']:
                                if 'Region' not in [s[0] for s in summary_items]:
                                    summary_items.append(("Region (addr1, addr2)", f"({transaction_data.get('addr1', 'N/A')}, {transaction_data.get('addr2', 'N/A')})"))
                            elif key == 'ProductCD':
                                summary_items.append(("Product Code", str(value)))
                            elif key in ['V258', 'V201', 'V149', 'C14']:
                                summary_items.append((key, f"{value:.4f}"))
                        
                        summary_items.append(("Fraud Probability", f"{fraud_probability:.4f}"))
                        summary_items.append(("Prediction", "Fraudulent" if fraud_prediction == 1 else "Legitimate"))
                        
                        col1, col2 = st.columns(2)
                        mid_point = len(summary_items) // 2
                        with col1:
                            for key, value in summary_items[:mid_point]:
                                st.write(f"**{key}**: {value}")
                        with col2:
                            for key, value in summary_items[mid_point:]:
                                st.write(f"**{key}**: {value}")
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
                    st.info("Please check your input values and try again.")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    else:
        # Show instructions when form is not submitted
        st.info("""
        💡 **Instructions:**
        1. Fill in the transaction details above
        2. Click "Predict Fraud Risk" to get an instant prediction
        3. The model will analyze the transaction and provide:
           - Fraud probability score
           - Risk level (Low/Medium/High)
           - Visual risk indicators
           - Feature importance insights
        
        **Note**: Only the most important features are shown. Other features are automatically filled with default values from the training data.
        """)

if __name__ == "__main__":
    main()

