"""
Data Visualization Module for Customer Churn & Segmentation
Author: Sindhu Patil (Data Science Intern)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve

# Professional styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("husl")

FIGURES_WEEK1_DIR = os.path.join("outputs", "figures", "week1")
FIGURES_WEEK2_DIR = os.path.join("outputs", "figures", "week2")
FIGURES_WEEK3_DIR = os.path.join("outputs", "figures", "week3")
FIGURES_WEEK4_DIR = os.path.join("outputs", "figures", "week4")
FIGURES_WEEK5_DIR = os.path.join("outputs", "figures", "week5")
FIGURES_WEEK6_DIR = os.path.join("outputs", "figures", "week6")

os.makedirs(FIGURES_WEEK1_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK2_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK3_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK4_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK5_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK6_DIR, exist_ok=True)

COLOR_NO_CHURN = "#2ecc71"
COLOR_CHURN = "#e74c3c"
CLUSTER_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]


def save_figure(fig, filename, folder_path=FIGURES_WEEK6_DIR):
    """Saves figure to specified folder path at 300 DPI."""
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization figure: {filepath}")
    return filepath


# =========================================================
# WEEK 6 FINAL CAPSTONE VISUALIZATION FUNCTIONS
# =========================================================

def draw_capstone_pipeline_diagram(filename="capstone_pipeline.png"):
    """Generates an end-to-end data science architecture diagram saved to outputs/figures/week6/."""
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.axis('off')

    stages = [
        ("1. IBM Telco Dataset\n(7,043 Rows × 21 Columns)", "#34495e"),
        ("2. Data Cleaning & Preprocessing\n(0 Nulls, Median Imputation, Scaling)", "#2980b9"),
        ("3. Exploratory Data Analysis\n(Demographics, Contracts, Billing Patterns)", "#16a085"),
        ("4. Customer Segmentation\n(K-Means Clustering, K=3 Cohorts)", "#8e44ad"),
        ("5. Supervised Learning\n(Logistic Regression Baseline, 80.55% Acc)", "#d35400"),
        ("6. Deep Learning Application\n(Keras Sequential NN, 75.36% Recall)", "#c0392b"),
        ("7. Model Comparison & Evaluation\n(Metrics: Recall, F1, ROC-AUC, PR)", "#27ae60"),
        ("8. Customer Risk Analysis & Recommendations\n(Targeted Retention Campaigns)", "#2980b9")
    ]

    y_pos = np.linspace(0.92, 0.08, len(stages))

    for i, (text, color) in enumerate(stages):
        ax.text(0.5, y_pos[i], text, ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.6', facecolor=color, edgecolor='none', alpha=0.9))
        if i < len(stages) - 1:
            ax.annotate('', xy=(0.5, y_pos[i+1] + 0.035), xytext=(0.5, y_pos[i] - 0.035),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_title("End-to-End Customer Churn Analytics Pipeline Architecture", fontsize=14, fontweight='bold', pad=20)
    
    filepath = save_figure(fig, filename, folder_path=FIGURES_WEEK6_DIR)
    return filepath


def plot_final_churn_distribution(df, filename="final_churn_distribution.png"):
    """Plots final overall target churn distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=ax)
    ax.set_title("IBM Telco Churn Target Class Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Customer Churn Status", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.set_xticklabels(['Retained (No)', 'Churned (Yes)'])

    total = len(df)
    for p in ax.patches:
        height = int(p.get_height())
        pct = (height / total) * 100
        ax.annotate(f'{height:,}\n({pct:.2f}%)', (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    return save_figure(fig, filename)


def plot_customer_risk_distribution(y_prob, filename="customer_risk_distribution.png"):
    """Plots customer risk tier distribution (High Risk > 0.6, Medium Risk 0.3-0.6, Low Risk < 0.3)."""
    risk_labels = []
    for p in y_prob:
        if p > 0.6:
            risk_labels.append("High Risk (> 0.60)")
        elif p >= 0.3:
            risk_labels.append("Medium Risk (0.30 - 0.60)")
        else:
            risk_labels.append("Low Risk (< 0.30)")

    df_risk = pd.DataFrame({'Risk_Tier': risk_labels})
    tier_order = ["Low Risk (< 0.30)", "Medium Risk (0.30 - 0.60)", "High Risk (> 0.60)"]
    palette = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_risk, x='Risk_Tier', order=tier_order, palette=palette, ax=ax)
    ax.set_title("Customer Churn Risk Tier Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Risk Tier", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)

    total = len(y_prob)
    for p in ax.patches:
        height = int(p.get_height())
        pct = (height / total) * 100
        ax.annotate(f'{height:,}\n({pct:.1f}%)', (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    return save_figure(fig, filename)


def plot_final_model_comparison(comparison_df, filename="final_model_comparison.png"):
    """Plots master comparison bar chart across Logistic Regression, Random Forest, and Deep Neural Network."""
    metrics_melted = pd.melt(comparison_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'],
                             var_name='Metric', value_name='Value')

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=metrics_melted, x='Metric', y='Value', hue='Model', palette='Set2', ax=ax)
    ax.set_title("Master Machine Learning & Deep Learning Model Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Evaluation Metric", fontsize=12)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", fontsize=10)

    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f'{val:.4f}', (p.get_x() + p.get_width() / 2., val + 0.01),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    return save_figure(fig, filename)


# =========================================================
# WEEK 3 CLUSTERING VISUALIZATIONS (RETAINED)
# =========================================================

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

    return save_figure(fig, filename, folder_path=FIGURES_WEEK6_DIR)


# =========================================================
# WEEK 4 & 5 VISUALIZATIONS (RETAINED)
# =========================================================

def plot_training_history_loss(history, filename="training_history_loss.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.history['loss']) + 1)
    ax.plot(epochs, history.history['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax.plot(epochs, history.history['val_loss'], 'r--s', label='Validation Loss', linewidth=2)
    ax.set_title("Neural Network Loss Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, linestyle='--', alpha=0.6)
    return save_figure(fig, filename, folder_path=FIGURES_WEEK5_DIR)


def plot_training_history_accuracy(history, filename="training_history_accuracy.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax.plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax.plot(epochs, history.history['val_accuracy'], 'g--s', label='Validation Accuracy', linewidth=2)
    ax.set_title("Neural Network Accuracy Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, linestyle='--', alpha=0.6)
    return save_figure(fig, filename, folder_path=FIGURES_WEEK5_DIR)


def plot_confusion_matrix(cm, display_labels=['Retained (No)', 'Churned (Yes)'], filename="confusion_matrix.png", folder_path=FIGURES_WEEK6_DIR):
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax.grid(False)
    return save_figure(fig, filename, folder_path=folder_path)


def plot_roc_curve(y_test, y_prob_dict, filename="roc_curve.png", folder_path=FIGURES_WEEK6_DIR):
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, (y_prob, auc_score) in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (ROC-AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Random Baseline (AUC = 0.5000)')
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc='lower right', fontsize=11); ax.grid(True, linestyle='--', alpha=0.6)
    return save_figure(fig, filename, folder_path=folder_path)


def plot_precision_recall_curve(y_test, y_prob_dict, filename="precision_recall_curve.png", folder_path=FIGURES_WEEK6_DIR):
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, (y_prob, avg_prec) in y_prob_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, linewidth=2, label=f"{model_name} (Avg Prec = {avg_prec:.4f})")
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'No-Skill Baseline ({baseline:.4f})')
    ax.set_title("Precision-Recall (PR) Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.legend(loc='lower left', fontsize=11); ax.grid(True, linestyle='--', alpha=0.6)
    return save_figure(fig, filename, folder_path=folder_path)


def plot_threshold_analysis(thresholds, precisions, recalls, f1s, filename="threshold_analysis.png", folder_path=FIGURES_WEEK6_DIR):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, 'g--', label='F1-Score', linewidth=2)
    ax.axvline(x=0.5, color='black', linestyle=':', label='Standard Threshold (0.5)')
    ax.set_title("Classification Threshold Trade-off Analysis", fontsize=14, fontweight='bold')
    ax.set_xlabel("Probability Threshold", fontsize=12); ax.set_ylabel("Metric Value", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, linestyle='--', alpha=0.6)
    return save_figure(fig, filename, folder_path=folder_path)


def plot_week4_vs_week5_comparison(comparison_df, filename="week4_vs_week5_comparison.png"):
    return plot_final_model_comparison(comparison_df, filename=filename)


def plot_top_coefficients(df_coef, top_n=12, filename="top_logistic_coefficients.png", folder_path=FIGURES_WEEK4_DIR):
    df_top = df_coef.head(top_n).copy()
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in df_top['Coefficient']]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_top, x='Coefficient', y='Feature', hue='Feature', palette=colors, legend=False, ax=ax)
    ax.set_title(f"Top {top_n} Logistic Regression Feature Coefficients", fontsize=14, fontweight='bold')
    ax.set_xlabel("Coefficient Value", fontsize=11); ax.set_ylabel("Feature Name", fontsize=11)
    for p in ax.patches:
        val = p.get_width()
        offset = 0.02 if val >= 0 else -0.02
        ha = 'left' if val >= 0 else 'right'
        ax.annotate(f'{val:+.4f}', (val + offset, p.get_y() + p.get_height() / 2.),
                    ha=ha, va='center', fontsize=9, fontweight='bold')
    return save_figure(fig, filename, folder_path=folder_path)
