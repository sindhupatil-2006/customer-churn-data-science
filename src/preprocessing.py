"""
Preprocessing and Feature Engineering Module
Author: Sindhu Patil (Data Science Intern)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_telco_data(df, target_col="Churn", drop_id=True):
    """
    Preprocesses the cleaned Telco Customer Churn dataframe:
    - Encodes target variable ('Yes': 1, 'No': 0)
    - Encodes binary features
    - Applies One-Hot Encoding to categorical features
    - Scales numerical features with StandardScaler
    Returns X (features DataFrame), y (target Series), feature_names, scaler
    """
    df_prep = df.copy()

    # Drop customerID if present
    if drop_id and "customerID" in df_prep.columns:
        customer_ids = df_prep["customerID"]
        df_prep = df_prep.drop(columns=["customerID"])
    else:
        customer_ids = None

    # Binary Target encoding
    if target_col in df_prep.columns:
        if df_prep[target_col].dtype in [object, str] or 'str' in str(df_prep[target_col].dtype):
            df_prep[target_col] = df_prep[target_col].map({"Yes": 1, "No": 0}).astype(int)
        y = df_prep[target_col].astype(int)
        df_prep = df_prep.drop(columns=[target_col])
    else:
        y = None

    # Binary categorical features mapping
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df_prep.columns:
            if col == "gender":
                df_prep[col] = df_prep[col].map({"Male": 1, "Female": 0})
            else:
                df_prep[col] = df_prep[col].map({"Yes": 1, "No": 0})

    # Multi-class categorical features to One-Hot Encode
    cat_cols = df_prep.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        df_prep = pd.get_dummies(df_prep, columns=cat_cols, drop_first=True, dtype=int)

    # Separate numerical features for scaling
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    num_cols = [c for c in num_cols if c in df_prep.columns]

    scaler = StandardScaler()
    df_scaled = df_prep.copy()
    if num_cols:
        df_scaled[num_cols] = scaler.fit_transform(df_prep[num_cols])

    return df_scaled, y, list(df_scaled.columns), scaler, customer_ids


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """Performs stratified train/test split on features and target."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from src.data_cleaning import load_raw_data, clean_telco_data
    df_raw = load_raw_data()
    df_clean = clean_telco_data(df_raw)
    X, y, feature_names, scaler, ids = preprocess_telco_data(df_clean)
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts(normalize=True))
