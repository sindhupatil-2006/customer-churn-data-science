"""
Data Visualization Module for Customer Churn Analysis
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
os.makedirs(FIGURES_WEEK1_DIR, exist_ok=True)
os.makedirs(FIGURES_WEEK2_DIR, exist_ok=True)

COLOR_NO_CHURN = "#2ecc71"
COLOR_CHURN = "#e74c3c"


def save_figure(fig, filename, folder_path=FIGURES_WEEK2_DIR):
    """Saves figure to specified folder path at 300 DPI."""
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization figure: {filepath}")
    return filepath


def plot_churn_target_distribution(df, filename="churn_distribution.png"):
    """Plots overall customer churn target count & proportion."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df['Churn'].value_counts()
    churn_pcts = df['Churn'].value_counts(normalize=True) * 100

    # Donut chart
    ax[0].pie(churn_counts, labels=[f"No ({churn_pcts['No']:.1f}%)", f"Yes ({churn_pcts['Yes']:.1f}%)"],
              startangle=90, colors=[COLOR_NO_CHURN, COLOR_CHURN], explode=(0, 0.08), wedgeprops=dict(width=0.4))
    ax[0].set_title("Customer Churn Proportions", fontsize=14, fontweight='bold', fontfamily='sans-serif')

    # Count plot
    sns.countplot(data=df, x='Churn', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=ax[1], legend=False)
    ax[1].set_title("Customer Churn Counts", fontsize=14, fontweight='bold')
    ax[1].set_xlabel("Churn Status", fontsize=12)
    ax[1].set_ylabel("Number of Customers", fontsize=12)

    for p in ax[1].patches:
        height = int(p.get_height())
        pct = (height / len(df)) * 100
        ax[1].annotate(f'{height:,}\n({pct:.1f}%)', (p.get_x() + p.get_width() / 2., height / 2),
                       ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    return save_figure(fig, filename)


def plot_churn_by_contract(df, filename="churn_by_contract.png"):
    """Plots churn breakdown by Contract type."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=ax)
    ax.set_title("Customer Churn Distribution by Contract Type", fontsize=14, fontweight='bold')
    ax.set_xlabel("Contract Type", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.legend(title="Churn Status", labels=["No Churn", "Churned"])

    # Annotate percentages
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    for i, p in enumerate(ax.patches):
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height:,}', (p.get_x() + p.get_width() / 2., height + 30),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    return save_figure(fig, filename)


def plot_churn_by_internet_service(df, filename="churn_by_internet_service.png"):
    """Plots churn breakdown by Internet Service provider."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, x='InternetService', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=ax)
    ax.set_title("Customer Churn by Internet Service Type", fontsize=14, fontweight='bold')
    ax.set_xlabel("Internet Service Type", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.legend(title="Churn Status", labels=["No Churn", "Churned"])

    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height:,}', (p.get_x() + p.get_width() / 2., height + 25),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    return save_figure(fig, filename)


def plot_churn_by_payment_method(df, filename="churn_by_payment_method.png"):
    """Plots churn breakdown by Payment Method."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=ax)
    ax.set_title("Customer Churn Rate by Payment Method", fontsize=14, fontweight='bold')
    ax.set_xlabel("Payment Method", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.set_xticks(range(len(df['PaymentMethod'].unique())))
    ax.set_xticklabels(df['PaymentMethod'].value_counts().index, rotation=15, ha='right')
    ax.legend(title="Churn Status", labels=["No Churn", "Churned"])

    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            ax.annotate(f'{height:,}', (p.get_x() + p.get_width() / 2., height + 20),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    return save_figure(fig, filename)


def plot_tenure_distribution(df, filename="tenure_distribution.png"):
    """Plots tenure distribution histogram and KDE split by Churn."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, palette=[COLOR_NO_CHURN, COLOR_CHURN],
                 bins=30, ax=ax, alpha=0.6)
    ax.set_title("Customer Tenure Distribution by Churn Status", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tenure (Months)", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)

    return save_figure(fig, filename)


def plot_monthly_charges_by_churn(df, filename="monthly_charges_by_churn.png"):
    """Plots MonthlyCharges box plot & density split by Churn."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=axes[0], legend=False)
    axes[0].set_title("Monthly Charges Boxplot by Churn", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Churn Status")
    axes[0].set_ylabel("Monthly Charges ($)")

    # KDE
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=axes[1], alpha=0.5)
    axes[1].set_title("Monthly Charges Density by Churn", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Monthly Charges ($)")

    return save_figure(fig, filename)


def plot_total_charges_by_churn(df, filename="total_charges_by_churn.png"):
    """Plots TotalCharges box plot & density split by Churn."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    sns.boxplot(data=df, x='Churn', y='TotalCharges', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=axes[0], legend=False)
    axes[0].set_title("Total Charges Boxplot by Churn", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Churn Status")
    axes[0].set_ylabel("Total Charges ($)")

    # KDE
    sns.kdeplot(data=df, x='TotalCharges', hue='Churn', fill=True, palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=axes[1], alpha=0.5)
    axes[1].set_title("Total Charges Density by Churn", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Total Charges ($)")

    return save_figure(fig, filename)


def plot_correlation_heatmap(df, filename="correlation_heatmap.png"):
    """Plots correlation matrix heatmap for numeric features and Churn."""
    df_corr = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']].copy()
    df_corr['Churn_Numeric'] = (df['Churn'] == 'Yes').astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df_corr.corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="vlag", center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Numerical Features Correlation Matrix Heatmap", fontsize=14, fontweight='bold')

    return save_figure(fig, filename)


def plot_multivariate_contract_charges(df, filename="multivariate_contract_charges.png"):
    """Multivariate plot: Contract + MonthlyCharges + Churn."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Contract', y='MonthlyCharges', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=ax)
    ax.set_title("Multivariate Analysis: Monthly Charges by Contract Type & Churn Status", fontsize=14, fontweight='bold')
    ax.set_xlabel("Contract Type", fontsize=12)
    ax.set_ylabel("Monthly Charges ($)", fontsize=12)
    ax.legend(title="Churn Status", labels=["No Churn", "Churned"])

    return save_figure(fig, filename)


def plot_demographics_churn(df, filename="demographics_churn.png"):
    """Subplots of demographic variables vs Churn."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    titles = ['Gender vs Churn', 'Senior Citizen (0/1) vs Churn', 'Partner Status vs Churn', 'Dependents Status vs Churn']

    for i, col in enumerate(demo_cols):
        ax = axes[i // 2, i % 2]
        sns.countplot(data=df, x=col, hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=ax)
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel("Customer Count")
        ax.legend(title="Churn", labels=["No", "Yes"])

    return save_figure(fig, filename)


def plot_services_churn(df, filename="services_churn.png"):
    """Subplots of value-added security & tech support services vs Churn."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(data=df, x='OnlineSecurity', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=axes[0])
    axes[0].set_title("Online Security Adoption vs Churn", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Online Security Service")
    axes[0].set_ylabel("Customer Count")

    sns.countplot(data=df, x='TechSupport', hue='Churn', palette=[COLOR_NO_CHURN, COLOR_CHURN], ax=axes[1])
    axes[1].set_title("Tech Support Adoption vs Churn", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Tech Support Service")
    axes[1].set_ylabel("Customer Count")

    return save_figure(fig, filename)
