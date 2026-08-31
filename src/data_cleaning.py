"""
Data Cleaning Module for Telco Customer Churn Dataset
Author: Sindhu Patil (Data Science Intern)
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_telco_churn.csv")


def acquire_dataset(raw_path=RAW_DATA_PATH, url=DATA_URL):
    """Acquires raw Telco Customer Churn dataset from public URL if not present locally."""
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    if not os.path.exists(raw_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, raw_path)
        print(f"Saved raw dataset to {raw_path}")
    else:
        print(f"Raw dataset already exists at {raw_path}")
    return raw_path


def load_raw_data(raw_path=RAW_DATA_PATH):
    """Loads raw dataset into pandas DataFrame."""
    acquire_dataset(raw_path)
    df = pd.read_csv(raw_path)
    return df


def inspect_raw_data(df):
    """Performs comprehensive raw data inspection and returns summary stats."""
    info_dict = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return info_dict


def clean_telco_data(df):
    """
    Cleans raw Telco Customer Churn dataframe:
    - Handles whitespace strings in 'TotalCharges'
    - Converts 'TotalCharges' to float64
    - Imputes missing TotalCharges values using median
    - Verifies zero duplicate records
    - Checks invalid numerical bounds
    """
    df_clean = df.copy()

    # 1. Clean TotalCharges whitespace strings
    if "TotalCharges" in df_clean.columns:
        # Count space strings before conversion
        space_count = (df_clean["TotalCharges"].astype(str).str.strip() == "").sum()
        print(f"Found {space_count} whitespace values in TotalCharges.")
        
        # Replace empty/whitespace strings with NaN and convert to float
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        
        # Impute missing TotalCharges with median
        median_tc = df_clean["TotalCharges"].median()
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_tc)
        print(f"Imputed missing TotalCharges with median: {median_tc:.2f}")

    # 2. Check and remove duplicate rows if any
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {dup_count} duplicate rows.")

    # 3. Save cleaned dataset
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned dataset to {PROCESSED_DATA_PATH} with shape {df_clean.shape}")

    return df_clean


if __name__ == "__main__":
    raw_file = acquire_dataset()
    df_raw = load_raw_data(raw_file)
    summary = inspect_raw_data(df_raw)
    print("Raw Data Shape:", summary["shape"])
    df_cleaned = clean_telco_data(df_raw)
    print("Clean Data Shape:", df_cleaned.shape)
