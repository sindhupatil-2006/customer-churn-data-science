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

os.makedirs(FIGURES_WEEK1_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK2_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK3_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK4_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK5_DIR, exist_ok=True)

COLOR_NO_CHURN = "#2ecc71"
COLOR_CHURN = "#e74c3c"
CLUSTER_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]


def save_figure(fig, filename, folder_path=FIGURES_WEEK5_DIR):
    """Saves figure to specified folder path at 300 DPI."""
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization figure: {filepath}")
    return filepath


# =========================================================
# WEEK 5 DEEP LEARNING VISUALIZATION FUNCTIONS
# =========================================================

def plot_training_history_loss(history, filename="training_history_loss.png"):
    """Plots Training Loss vs Validation Loss across epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.history['loss']) + 1)
    ax.plot(epochs, history.history['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax.plot(epochs, history.history['val_loss'], 'r--s', label='Validation Loss', linewidth=2)
    
    ax.set_title("Neural Network Loss Curve (Binary Cross-Entropy)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    return save_figure(fig, filename)


def plot_training_history_accuracy(history, filename="training_history_accuracy.png"):
    """Plots Training Accuracy vs Validation Accuracy across epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax.plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax.plot(epochs, history.history['val_accuracy'], 'g--s', label='Validation Accuracy', linewidth=2)
    
    ax.set_title("Neural Network Accuracy Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (0.0 to 1.0)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    return save_figure(fig, filename)


def plot_confusion_matrix(cm, display_labels=['Retained (No)', 'Churned (Yes)'], filename="confusion_matrix.png", folder_path=FIGURES_WEEK5_DIR):
    """Plots formatted Confusion Matrix display."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax.grid(False)

    return save_figure(fig, filename, folder_path=folder_path)


def plot_roc_curve(y_test, y_prob_dict, filename="roc_curve.png", folder_path=FIGURES_WEEK5_DIR):
    """Plots ROC Curve comparing model probabilities against random baseline."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, (y_prob, auc_score) in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (ROC-AUC = {auc_score:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', label='Random Baseline (AUC = 0.5000)')
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    return save_figure(fig, filename, folder_path=folder_path)


def plot_precision_recall_curve(y_test, y_prob_dict, filename="precision_recall_curve.png", folder_path=FIGURES_WEEK5_DIR):
    """Plots Precision-Recall Curve comparing model probabilities."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, (y_prob, avg_prec) in y_prob_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, linewidth=2, label=f"{model_name} (Avg Prec = {avg_prec:.4f})")

    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'No-Skill Baseline ({baseline:.4f})')
    ax.set_title("Precision-Recall (PR) Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    return save_figure(fig, filename, folder_path=folder_path)


def plot_threshold_analysis(thresholds, precisions, recalls, f1s, filename="threshold_analysis.png", folder_path=FIGURES_WEEK5_DIR):
    """Plots Precision, Recall, and F1-Score trade-offs across classification thresholds."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, 'g--', label='F1-Score', linewidth=2)
    
    ax.axvline(x=0.5, color='black', linestyle=':', label='Standard Threshold (0.5)')
    
    ax.set_title("Classification Threshold Trade-off Analysis", fontsize=14, fontweight='bold')
    ax.set_xlabel("Classification Probability Threshold", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    return save_figure(fig, filename, folder_path=folder_path)


def plot_week4_vs_week5_comparison(comparison_df, filename="week4_vs_week5_comparison.png"):
    """Plots Week 4 vs Week 5 model comparison metrics bar chart."""
    metrics_melted = pd.melt(comparison_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'],
                             var_name='Metric', value_name='Value')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrics_melted, x='Metric', y='Value', hue='Model', palette=['#3498db', '#e74c3c'], ax=ax)
    ax.set_title("Model Comparison: Week 4 Logistic Regression vs Week 5 Neural Network", fontsize=14, fontweight='bold')
    ax.set_xlabel("Evaluation Metric", fontsize=12)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model Framework", fontsize=10)

    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f'{val:.4f}', (p.get_x() + p.get_width() / 2., val + 0.01),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    return save_figure(fig, filename)


def plot_top_coefficients(df_coef, top_n=12, filename="top_logistic_coefficients.png", folder_path=FIGURES_WEEK4_DIR):
    """Plots top positive and negative Logistic Regression feature coefficients."""
    df_top = df_coef.head(top_n).copy()
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in df_top['Coefficient']]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_top, x='Coefficient', y='Feature', hue='Feature', palette=colors, legend=False, ax=ax)
    ax.set_title(f"Top {top_n} Logistic Regression Feature Coefficients", fontsize=14, fontweight='bold')
    ax.set_xlabel("Coefficient Value (Red = Increases Churn Risk, Green = Reduces Churn Risk)", fontsize=11)
    ax.set_ylabel("Feature Name", fontsize=11)

    for p in ax.patches:
        val = p.get_width()
        offset = 0.02 if val >= 0 else -0.02
        ha = 'left' if val >= 0 else 'right'
        ax.annotate(f'{val:+.4f}', (val + offset, p.get_y() + p.get_height() / 2.),
                    ha=ha, va='center', fontsize=9, fontweight='bold')

    return save_figure(fig, filename, folder_path=folder_path)
