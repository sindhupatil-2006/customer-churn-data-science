"""
Data Visualization Module for Customer Churn & Segmentation
Author: Sindhu Patil (Data Science Intern)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Professional styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("husl")

FIGURES_WEEK1_DIR = os.path.join("outputs", "figures", "week1")
FIGURES_WEEK2_DIR = os.path.join("outputs", "figures", "week2")
FIGURES_WEEK3_DIR = os.path.join("outputs", "figures", "week3")

os.makedirs(FIGURES_WEEK1_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK2_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK3_DIR, exist_ok=True)

COLOR_NO_CHURN = "#2ecc71"
COLOR_CHURN = "#e74c3c"
CLUSTER_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]


def save_figure(fig, filename, folder_path=FIGURES_WEEK3_DIR):
    """Saves figure to specified folder path at 300 DPI."""
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization figure: {filepath}")
    return filepath


# =========================================================
# WEEK 3 CLUSTERING VISUALIZATION FUNCTIONS
# =========================================================

def plot_elbow_curve(k_range, inertias, filename="elbow_method.png"):
    """Plots K-Means Elbow Curve (Inertia vs K)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8, color='#2980b9')
    ax.set_title("K-Means Clustering: Elbow Method (Inertia vs K)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax.set_xticks(k_range)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight elbow point at K=3
    if 3 in k_range:
        idx = list(k_range).index(3)
        ax.plot(3, inertias[idx], 'ro', markersize=12, label='Elbow Point (K=3)')
        ax.legend(fontsize=11)

    return save_figure(fig, filename)


def plot_silhouette_scores(k_range, silhouette_scores, filename="silhouette_scores.png"):
    """Plots Silhouette Scores vs K."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, silhouette_scores, 'rs--', linewidth=2, markersize=8, color='#e67e22')
    ax.set_title("K-Means Clustering: Silhouette Score vs K", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_xticks(k_range)
    ax.grid(True, linestyle='--', alpha=0.6)

    return save_figure(fig, filename)


def plot_cluster_sizes(summary_df, filename="cluster_sizes.png"):
    """Plots cluster customer counts and proportions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_df, x='Segment_Name', y='Customer_Count', palette=CLUSTER_PALETTE[:len(summary_df)], ax=ax)
    ax.set_title("Customer Count per Segment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_xticklabels(summary_df['Segment_Name'], rotation=15, ha='right')

    for p in ax.patches:
        height = int(p.get_height())
        pct = (height / summary_df['Customer_Count'].sum()) * 100
        ax.annotate(f'{height:,}\n({pct:.1f}%)', (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    return save_figure(fig, filename)


def plot_cluster_pca_2d(X_pca, labels, centroids_pca, cluster_names, filename="cluster_pca_2d.png"):
    """Plots 2D PCA visual scatter projection of K-Means clusters."""
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        points = X_pca[labels == label]
        ax.scatter(points[:, 0], points[:, 1], label=f"Cluster {label}: {cluster_names[i]}",
                   alpha=0.5, s=30, color=CLUSTER_PALETTE[i])
        
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=250, c='black', marker='X', label='Cluster Centroids', zorder=10)
    ax.set_title("Customer Segments Visualization (2D PCA Projection)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=12)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    return save_figure(fig, filename)


def plot_avg_tenure_by_cluster(summary_df, filename="avg_tenure_by_cluster.png"):
    """Plots Average Tenure across segments."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_df, x='Segment_Name', y='Avg_Tenure', palette=CLUSTER_PALETTE[:len(summary_df)], ax=ax)
    ax.set_title("Average Tenure (Months) by Customer Segment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Average Tenure (Months)", fontsize=12)
    ax.set_xticklabels(summary_df['Segment_Name'], rotation=15, ha='right')

    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f'{val:.1f} mos', (p.get_x() + p.get_width() / 2., val + 1),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    return save_figure(fig, filename)


def plot_avg_monthly_charges_by_cluster(summary_df, filename="avg_monthly_charges_by_cluster.png"):
    """Plots Average Monthly Charges across segments."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_df, x='Segment_Name', y='Avg_Monthly_Charges', palette=CLUSTER_PALETTE[:len(summary_df)], ax=ax)
    ax.set_title("Average Monthly Charges ($) by Customer Segment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Average Monthly Charges ($)", fontsize=12)
    ax.set_xticklabels(summary_df['Segment_Name'], rotation=15, ha='right')

    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f'${val:.2f}', (p.get_x() + p.get_width() / 2., val + 1.5),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    return save_figure(fig, filename)


def plot_churn_rate_by_cluster(summary_df, filename="churn_rate_by_cluster.png"):
    """Plots Observed Churn Rate (%) across segments."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_df, x='Segment_Name', y='Churn_Rate', palette=[COLOR_NO_CHURN, COLOR_CHURN, "#34495e"], ax=ax)
    ax.set_title("Observed Churn Rate (%) across Customer Segments", fontsize=14, fontweight='bold')
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_xticklabels(summary_df['Segment_Name'], rotation=15, ha='right')

    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f'{val:.2f}%', (p.get_x() + p.get_width() / 2., val + 1),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    return save_figure(fig, filename)


def plot_contract_distribution_by_cluster(df, cluster_map, filename="contract_distribution_by_cluster.png"):
    """Plots Contract distribution by segment."""
    df_plot = df.copy()
    df_plot['Segment_Name'] = df_plot['Cluster'].map(cluster_map)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df_plot, x='Segment_Name', hue='Contract', palette="Set2", ax=ax)
    ax.set_title("Contract Type Distribution across Customer Segments", fontsize=14, fontweight='bold')
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.set_xticklabels(df_plot['Segment_Name'].unique(), rotation=15, ha='right')
    ax.legend(title="Contract Type")

    return save_figure(fig, filename)
