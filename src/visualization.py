"""
Data Visualization Module for Customer Churn Analysis & Model Evaluation
Author: Sindhu Patil (Data Science Intern)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# High-aesthetic styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("husl")
FIGURES_DIR = os.path.join("outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_churn_distribution(df, filename="week1_churn_distribution.png"):
    """Plots overall customer churn target distribution."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df['Churn'].value_counts()
    
    # Donut chart
    ax[0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90,
              colors=['#2ecc71', '#e74c3c'], explode=(0, 0.08), wedgeprops=dict(width=0.4))
    ax[0].set_title("Customer Churn Proportions", fontsize=14, fontweight='bold')
    
    # Count plot
    sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'], ax=ax[1])
    ax[1].set_title("Customer Churn Count", fontsize=14, fontweight='bold')
    ax[1].set_xlabel("Churn Status")
    ax[1].set_ylabel("Number of Customers")
    
    for p in ax[1].patches:
        ax[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_demographics_vs_churn(df, filename="week2_demographics_churn.png"):
    """Plots demographic features against customer churn."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    titles = ['Gender Distribution by Churn', 'Senior Citizen Status by Churn',
              'Partner Status by Churn', 'Dependents Status by Churn']

    for i, col in enumerate(demo_cols):
        ax = axes[i // 2, i % 2]
        sns.countplot(data=df, x=col, hue='Churn', palette=['#3498db', '#e74c3c'], ax=ax)
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Customer Count")
        ax.legend(title="Churn", labels=["No", "Yes"])

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_services_and_contract_vs_churn(df, filename="week2_contract_services_churn.png"):
    """Plots Contract, InternetService, PaymentMethod against churn."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Contract
    sns.countplot(data=df, x='Contract', hue='Churn', palette=['#2ecc71', '#e74c3c'], ax=axes[0])
    axes[0].set_title("Churn by Contract Type", fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15)
    
    # Internet Service
    sns.countplot(data=df, x='InternetService', hue='Churn', palette=['#34495e', '#e74c3c'], ax=axes[1])
    axes[1].set_title("Churn by Internet Service", fontsize=12, fontweight='bold')
    
    # Payment Method
    sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette=['#9b59b6', '#e74c3c'], ax=axes[2])
    axes[2].set_title("Churn by Payment Method", fontsize=12, fontweight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_financial_distributions(df, filename="week2_financial_distributions.png"):
    """Plots distributions of tenure, MonthlyCharges, TotalCharges by Churn."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, palette=['#2ecc71', '#e74c3c'], ax=axes[0])
    axes[0].set_title("Tenure (Months) Distribution", fontsize=12, fontweight='bold')
    
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, palette=['#3498db', '#e74c3c'], ax=axes[1])
    axes[1].set_title("Monthly Charges ($) Distribution", fontsize=12, fontweight='bold')
    
    sns.histplot(data=df, x='TotalCharges', hue='Churn', kde=True, palette=['#9b59b6', '#e74c3c'], ax=axes[2])
    axes[2].set_title("Total Charges ($) Distribution", fontsize=12, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_correlation_heatmap(df_numeric, filename="week2_correlation_heatmap.png"):
    """Plots correlation matrix heatmap for numeric features and Churn."""
    plt.figure(figsize=(10, 8))
    corr = df_numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_elbow_and_silhouette(k_range, inertias, silhouette_scores, filename="week3_elbow_silhouette.png"):
    """Plots K-Means Elbow curve (Inertia) and Silhouette Scores."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = '#2980b9'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontsize=12)
    ax1.plot(k_range, inertias, 'o-', color=color, linewidth=2, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#e67e22'
    ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
    ax2.plot(k_range, silhouette_scores, 's--', color=color, linewidth=2, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("K-Means Clustering: Elbow Method & Silhouette Analysis", fontsize=14, fontweight='bold')
    fig.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_cluster_pca(X_pca, labels, centroids_pca, cluster_names, filename="week3_cluster_pca.png"):
    """Plots 2D PCA visual representation of clusters."""
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("bright", len(np.unique(labels)))
    
    for i, name in enumerate(cluster_names):
        points = X_pca[labels == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}: {name}", alpha=0.6, s=30, color=palette[i])
        
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=250, c='black', marker='X', label='Centroids')
    plt.title("Customer Segments Visualization (PCA 2D Projection)", fontsize=14, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_roc_curves(model_results, filename="week4_roc_curves.png"):
    """Plots ROC curves for evaluated ML & DL models."""
    plt.figure(figsize=(9, 7))
    
    for model_name, info in model_results.items():
        y_test = info['y_test']
        y_prob = info['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_confusion_matrices(matrices_dict, filename="week4_confusion_matrices.png"):
    """Plots confusion matrices side-by-side for compared models."""
    n_models = len(matrices_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]
        
    for i, (name, cm) in enumerate(matrices_dict.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        axes[i].set_title(f"{name}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_feature_importance(feature_names, importances, top_n=15, filename="week4_feature_importance.png"):
    """Plots top N feature importances from ML model."""
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='crest')
    plt.title(f"Top {top_n} Predictive Features for Customer Churn", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Feature Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_dl_training_history(train_losses, val_losses, train_accs, val_accs, filename="week5_dl_history.png"):
    """Plots PyTorch / Keras Deep Learning training history (loss & accuracy)."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r--s', label='Validation Loss', linewidth=2)
    axes[0].set_title("Neural Network Loss Over Epochs", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r--s', label='Validation Accuracy', linewidth=2)
    axes[1].set_title("Neural Network Accuracy Over Epochs", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy Rate")
    axes[1].legend()

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_master_model_comparison(comp_df, filename="week6_master_model_comparison.png"):
    """Plots master model performance comparison bar chart across all metrics."""
    df_melted = pd.melt(comp_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette='magma')
    plt.title("End-to-End Model Performance Benchmark (ML vs DL)", fontsize=14, fontweight='bold')
    plt.ylim(0.0, 1.05)
    plt.ylabel("Performance Score (0.0 to 1.0)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath
