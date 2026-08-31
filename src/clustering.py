"""
Unsupervised Learning and Customer Segmentation Module
Author: Sindhu Patil (Data Science Intern)
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

MODELS_DIR = os.path.join("outputs", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def find_optimal_k(X_scaled, k_range=range(2, 9), random_state=42):
    """
    Computes Inertia and Silhouette scores across a range of k values
    to find the optimal number of customer clusters.
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        
    return list(k_range), inertias, silhouette_scores


def train_kmeans(X_scaled, n_clusters=3, random_state=42, model_filename="kmeans_model.joblib"):
    """Trains K-Means model and saves trained artifact."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(kmeans, model_path)
    print(f"Saved trained K-Means model ({n_clusters} clusters) to {model_path}")
    
    return kmeans, labels, model_path


def profile_clusters(df_original, cluster_labels):
    """
    Calculates detailed cluster statistics and assigns evidence-backed descriptive segment names:
    - High-Value Loyal Customers
    - New / High-Charge At-Risk Customers
    - Budget Long-Term Customers
    """
    df_analysis = df_original.copy()
    df_analysis['Cluster'] = cluster_labels
    
    # Calculate Churn Rate per cluster
    if df_analysis['Churn'].dtype == object or 'str' in str(df_analysis['Churn'].dtype):
        df_analysis['Churn_Numeric'] = df_analysis['Churn'].map({'Yes': 1, 'No': 0}).astype(float)
    else:
        df_analysis['Churn_Numeric'] = df_analysis['Churn'].astype(float)

    profiles = df_analysis.groupby('Cluster').agg(
        Customer_Count=('tenure', 'count'),
        Percentage=('tenure', lambda x: len(x) / len(df_analysis) * 100),
        Avg_Tenure=('tenure', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
        Avg_Total_Charges=('TotalCharges', 'mean'),
        Churn_Rate=('Churn_Numeric', lambda x: x.mean() * 100)
    ).reset_index()

    # Evidence-backed naming function
    def name_cluster(row):
        tenure = row['Avg_Tenure']
        monthly = row['Avg_Monthly_Charges']
        churn = row['Churn_Rate']
        
        if tenure > 40 and monthly > 65:
            return "High-Value Loyal Customers"
        elif tenure < 25 and monthly > 65 and churn > 30:
            return "New High-Charge At-Risk Customers"
        elif tenure > 35 and monthly <= 65:
            return "Long-Term Budget Customers"
        elif tenure < 25 and monthly <= 65:
            return "Short-Term Standard Customers"
        else:
            return f"Segment {row['Cluster']} Profile"

    profiles['Segment_Name'] = profiles.apply(name_cluster, axis=1)
    
    # Assign names back to DataFrame
    cluster_name_map = dict(zip(profiles['Cluster'], profiles['Segment_Name']))
    df_analysis['Segment_Name'] = df_analysis['Cluster'].map(cluster_name_map)

    return profiles, df_analysis, cluster_name_map


def apply_pca(X_scaled, n_components=2, random_state=42):
    """Applies PCA dimensionality reduction for visual cluster analysis."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    print(f"PCA Explains {explained_var.sum()*100:.2f}% of feature variance.")
    return X_pca, pca
