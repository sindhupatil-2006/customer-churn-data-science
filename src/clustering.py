"""
Unsupervised Learning and Customer Segmentation Module
Author: Sindhu Patil (Data Science Intern)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

MODELS_DIR = os.path.join("outputs", "models")
RESULTS_DIR = os.path.join("outputs", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def prepare_clustering_data(df):
    """
    Prepares features for clustering:
    - Excludes 'Churn' target variable to prevent data leakage
    - Excludes 'customerID' identifier
    - Encodes categorical variables
    - Scales features with StandardScaler
    Returns X_scaled, df_encoded, scaler, feature_names
    """
    feature_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Filter features existing in dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_cluster = df[feature_cols].copy()

    # One-hot encode categorical features
    cat_cols = df_cluster.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    df_encoded = pd.get_dummies(df_cluster, columns=cat_cols, drop_first=True, dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    return X_scaled, df_encoded, scaler, list(df_encoded.columns)


def calculate_elbow_scores(X_scaled, k_range=range(2, 11), random_state=42):
    """Calculates Inertia values for K=2 through K=10."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return list(k_range), inertias


def calculate_silhouette_scores(X_scaled, k_range=range(2, 11), random_state=42):
    """Calculates Silhouette Scores for K=2 through K=10."""
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    return list(k_range), scores


def train_kmeans(X_scaled, n_clusters=3, random_state=42, model_filename="kmeans_model.joblib"):
    """Trains final K-Means model and saves trained artifact to outputs/models/."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(kmeans, model_path)
    print(f"Saved K-Means trained model artifact ({n_clusters} clusters) to {model_path}")

    return kmeans, labels, model_path


def profile_clusters(df_original, cluster_labels):
    """
    Calculates detailed cluster statistics (numerical means/medians, categorical breakdowns,
    and churn rates) and assigns evidence-backed descriptive names:
    - Cluster 0: High-Value Loyal Customers
    - Cluster 1: New High-Charge At-Risk Customers
    - Cluster 2: Long-Term Budget Customers
    """
    df_analysis = df_original.copy()
    df_analysis['Cluster'] = cluster_labels

    # Churn numeric conversion
    if 'Churn' in df_analysis.columns:
        if df_analysis['Churn'].dtype == object or 'str' in str(df_analysis['Churn'].dtype):
            df_analysis['Churn_Numeric'] = (df_analysis['Churn'] == 'Yes').astype(float)
        else:
            df_analysis['Churn_Numeric'] = df_analysis['Churn'].astype(float)
    else:
        df_analysis['Churn_Numeric'] = 0.0

    # Numerical Summary
    summary = df_analysis.groupby('Cluster').agg(
        Customer_Count=('tenure', 'count'),
        Percentage=('tenure', lambda x: len(x) / len(df_analysis) * 100),
        Avg_Tenure=('tenure', 'mean'),
        Median_Tenure=('tenure', 'median'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
        Median_Monthly_Charges=('MonthlyCharges', 'median'),
        Avg_Total_Charges=('TotalCharges', 'mean'),
        Median_Total_Charges=('TotalCharges', 'median'),
        Churned_Count=('Churn_Numeric', 'sum'),
        Churn_Rate=('Churn_Numeric', lambda x: x.mean() * 100),
        Non_Churn_Rate=('Churn_Numeric', lambda x: (1 - x.mean()) * 100)
    ).reset_index()

    # Evidence-backed Descriptive Segment Naming
    def assign_segment_name(row):
        tenure = row['Avg_Tenure']
        monthly = row['Avg_Monthly_Charges']
        churn = row['Churn_Rate']

        if tenure > 40 and monthly > 65:
            return "High-Value Loyal Customers"
        elif tenure < 25 and monthly > 50 and churn > 30:
            return "New High-Charge At-Risk Customers"
        elif monthly < 30:
            return "Long-Term Budget Customers"
        else:
            return f"Segment {int(row['Cluster'])}"

    summary['Segment_Name'] = summary.apply(assign_segment_name, axis=1)

    # Map segment names back to DataFrame
    cluster_map = dict(zip(summary['Cluster'], summary['Segment_Name']))
    df_analysis['Segment_Name'] = df_analysis['Cluster'].map(cluster_map)

    # Save summary CSV
    summary_path = os.path.join(RESULTS_DIR, "cluster_profiles_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved Cluster Profiles Summary CSV to {summary_path}")

    return summary, df_analysis, cluster_map


def apply_pca(X_scaled, n_components=2, random_state=42):
    """
    Applies PCA dimensionality reduction to 2 components strictly for visual cluster scatter plotting.
    Does NOT modify the feature space used during K-Means fitting.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA 2D explained variance ratio: PC1 = {explained_variance[0]*100:.2f}%, PC2 = {explained_variance[1]*100:.2f}% (Total = {explained_variance.sum()*100:.2f}%)")
    return X_pca, pca, explained_variance


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))
    from src.data_cleaning import load_raw_data, clean_telco_data

    df_raw = load_raw_data()
    df_clean = clean_telco_data(df_raw)
    X_scaled, df_enc, scaler, feats = prepare_clustering_data(df_clean)
    k_range, inertias = calculate_elbow_scores(X_scaled)
    _, sil_scores = calculate_silhouette_scores(X_scaled)
    kmeans, labels, path = train_kmeans(X_scaled, n_clusters=3)
    summary, df_clustered, cmap = profile_clusters(df_clean, labels)
    print("Cluster Summary:\n", summary[['Cluster', 'Segment_Name', 'Customer_Count', 'Avg_Tenure', 'Avg_Monthly_Charges', 'Churn_Rate']])
