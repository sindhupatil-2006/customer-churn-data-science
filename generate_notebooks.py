"""
Notebook Generation Script for 6-Week Data Science Project
Author: Sindhu Patil (Data Science Intern)
"""

import os
import json

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_cell(cell_type, source):
    """Creates a Jupyter notebook cell dictionary."""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }


def create_notebook(filename, cells):
    """Generates a standard .ipynb file from cells list."""
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.14.3"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {path}")


def build_all_notebooks():
    # ---------------------------------------------------------
    # WEEK 1 NOTEBOOK
    # ---------------------------------------------------------
    w1_cells = [
        make_cell("markdown", "# Week 1 — Data Acquisition, Cleaning and Preprocessing\n**Student Name:** Sindhu Patil | **Internship:** Data Science\n\n## Objective\nAcquire the Telco Customer Churn dataset, inspect missing values, duplicates, out-of-range values, clean data, encode binary/categorical variables, scale numerical features, and save processed data."),
        make_cell("code", "import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.data_cleaning import acquire_dataset, load_raw_data, inspect_raw_data, clean_telco_data\nfrom src.preprocessing import preprocess_telco_data\nfrom src.visualization import plot_churn_distribution"),
        make_cell("markdown", "### 1. Data Acquisition & Inspection\nLoading raw IBM Telco Customer Churn dataset (7043 rows, 21 columns)."),
        make_cell("code", "raw_path = acquire_dataset('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')\ndf_raw = load_raw_data(raw_path)\nprint('Raw Shape:', df_raw.shape)\ndf_raw.head()"),
        make_cell("code", "df_raw.info()"),
        make_cell("code", "df_raw.describe(include='all')"),
        make_cell("markdown", "### 2. Identifying Missing & Erroneous Values\nFinding whitespace characters in `TotalCharges` column."),
        make_cell("code", "space_mask = df_raw['TotalCharges'].astype(str).str.strip() == ''\nprint(f'Found {space_mask.sum()} empty string values in TotalCharges.')\ndf_raw[space_mask][['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']]"),
        make_cell("markdown", "### 3. Data Cleaning Execution\nConverting `TotalCharges` to float, imputing missing values with median, and verifying zero duplicates."),
        make_cell("code", "df_clean = clean_telco_data(df_raw)\nprint('Cleaned Shape:', df_clean.shape)\nprint('Missing values after cleaning:', df_clean.isnull().sum().sum())"),
        make_cell("markdown", "### 4. Categorical Encoding & Feature Scaling\nApplying binary encoding, One-Hot Encoding, and StandardScaler."),
        make_cell("code", "X_scaled, y, feat_names, scaler, customer_ids = preprocess_telco_data(df_clean, drop_id=True)\nprint('Encoded Features Shape:', X_scaled.shape)\nprint('Target Distribution:\n', y.value_counts(normalize=True))"),
        make_cell("markdown", "### 5. Data Visualization\nGenerating target churn distribution plot."),
        make_cell("code", "plot_churn_distribution(df_clean, '../outputs/figures/week1_churn_distribution.png')\nplt.figure(figsize=(6, 4))\nsns.countplot(data=df_clean, x='Churn', palette=['#2ecc71', '#e74c3c'])\nplt.title('Target Churn Distribution')\nplt.show()")
    ]
    create_notebook("week1_data_cleaning.ipynb", w1_cells)

    # ---------------------------------------------------------
    # WEEK 2 NOTEBOOK
    # ---------------------------------------------------------
    w2_cells = [
        make_cell("markdown", "# Week 2 — Exploratory Data Analysis & Visualization\n**Student Name:** Sindhu Patil | **Internship:** Data Science\n\n## Objective\nPerform thorough univariate, bivariate, and correlation analysis on demographics, subscription services, financial contract terms, and churn patterns."),
        make_cell("code", "import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.data_cleaning import load_raw_data, clean_telco_data\nfrom src.visualization import plot_demographics_vs_churn, plot_services_and_contract_vs_churn, plot_financial_distributions, plot_correlation_heatmap\n\ndf_clean = pd.read_csv('../data/processed/cleaned_telco_churn.csv')"),
        make_cell("markdown", "### 1. Demographic Analysis vs Churn\nExamining gender, senior status, partner, and dependent relationships with churn."),
        make_cell("code", "plot_demographics_vs_churn(df_clean, '../outputs/figures/week2_demographics_churn.png')\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\nsns.countplot(data=df_clean, x='Partner', hue='Churn', ax=axes[0])\nsns.countplot(data=df_clean, x='Dependents', hue='Churn', ax=axes[1])\nplt.tight_layout()\nplt.show()"),
        make_cell("markdown", "### 2. Contract & Service Types vs Churn\nComparing churn rates across Month-to-Month vs 1-Year vs 2-Year contracts and Fiber Optic internet service."),
        make_cell("code", "plot_services_and_contract_vs_churn(df_clean, '../outputs/figures/week2_contract_services_churn.png')\nplt.figure(figsize=(8, 4))\nsns.countplot(data=df_clean, x='Contract', hue='Churn', palette=['#2ecc71', '#e74c3c'])\nplt.title('Churn by Contract Type')\nplt.show()"),
        make_cell("markdown", "### 3. Financial Features Distributions\nAnalyzing tenure, MonthlyCharges, and TotalCharges distributions by Churn status."),
        make_cell("code", "plot_financial_distributions(df_clean, '../outputs/figures/week2_financial_distributions.png')\nplt.figure(figsize=(8, 4))\nsns.kdeplot(data=df_clean, x='tenure', hue='Churn', fill=True)\nplt.title('Tenure Distribution Density')\nplt.show()"),
        make_cell("markdown", "### 4. Correlation Analysis\nCorrelation heatmap of numeric variables with Churn."),
        make_cell("code", "df_corr = df_clean[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents']].copy()\ndf_corr['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})\nplot_correlation_heatmap(df_corr, '../outputs/figures/week2_correlation_heatmap.png')")
    ]
    create_notebook("week2_eda.ipynb", w2_cells)

    # ---------------------------------------------------------
    # WEEK 3 NOTEBOOK
    # ---------------------------------------------------------
    w3_cells = [
        make_cell("markdown", "# Week 3 — Unsupervised Learning & Customer Segmentation\n**Student Name:** Sindhu Patil | **Internship:** Data Science\n\n## Objective\nSegment customers into distinct behavioral cohorts using K-Means clustering, optimize cluster count $k$ via Elbow & Silhouette metrics, profile segment characteristics, and visualize 2D PCA projections."),
        make_cell("code", "import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.preprocessing import preprocess_telco_data\nfrom src.clustering import find_optimal_k, train_kmeans, profile_clusters, apply_pca\nfrom src.visualization import plot_elbow_and_silhouette, plot_cluster_pca\n\ndf_clean = pd.read_csv('../data/processed/cleaned_telco_churn.csv')\nX_scaled, y, feat_names, scaler, ids = preprocess_telco_data(df_clean)"),
        make_cell("markdown", "### 1. Determining Optimal Clusters ($k$)\nCalculating Inertia (Elbow method) and Silhouette Scores for $k \\in [2, 6]$."),
        make_cell("code", "k_range, inertias, sil_scores = find_optimal_k(X_scaled, k_range=range(2, 7))\nplot_elbow_and_silhouette(k_range, inertias, sil_scores, '../outputs/figures/week3_elbow_silhouette.png')\nfor k, s, i in zip(k_range, sil_scores, inertias):\n    print(f'k={k} | Silhouette Score: {s:.4f} | Inertia: {i:.2f}')"),
        make_cell("markdown", "### 2. Training K-Means & Profiling Segments\nFitting K-Means with $k=3$ and calculating mean tenure, charges, and churn rate per cluster."),
        make_cell("code", "kmeans_model, labels, path = train_kmeans(X_scaled, n_clusters=3, model_filename='../outputs/models/kmeans_model.joblib')\nprofiles, df_clustered, cluster_map = profile_clusters(df_clean, labels)\nprofiles"),
        make_cell("markdown", "### 3. PCA Cluster Projection\nVisualizing cluster separation in 2D principal component space."),
        make_cell("code", "X_pca, pca_obj = apply_pca(X_scaled, n_components=2)\ncentroids_pca = pca_obj.transform(kmeans_model.cluster_centers_)\nnames = [cluster_map[i] for i in range(3)]\nplot_cluster_pca(X_pca, labels, centroids_pca, names, '../outputs/figures/week3_cluster_pca.png')")
    ]
    create_notebook("week3_clustering.ipynb", w3_cells)

    # ---------------------------------------------------------
    # WEEK 4 NOTEBOOK
    # ---------------------------------------------------------
    w4_cells = [
        make_cell("markdown", "# Week 4 — Supervised Machine Learning\n**Student Name:** Sindhu Patil | **Internship:** Data Science\n\n## Objective\nBuild binary classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) to predict customer churn, evaluate Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-Fold Stratified CV, and extract feature importances."),
        make_cell("code", "import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nfrom src.preprocessing import preprocess_telco_data, get_train_test_split\nfrom src.ml_models import train_and_compare_ml_models\nfrom src.visualization import plot_roc_curves, plot_confusion_matrices, plot_feature_importance\n\ndf_clean = pd.read_csv('../data/processed/cleaned_telco_churn.csv')\nX_scaled, y, feat_names, scaler, ids = preprocess_telco_data(df_clean)\nX_train, X_test, y_train, y_test = get_train_test_split(X_scaled, y, test_size=0.2)"),
        make_cell("markdown", "### 1. Training & Comparing Candidate ML Models"),
        make_cell("code", "results, comp_df, best_name, best_path, feat_imp = train_and_compare_ml_models(X_train, X_test, y_train, y_test, feat_names)\ncomp_df"),
        make_cell("markdown", "### 2. Model Evaluation Plots\nROC curves, confusion matrices, and feature importances."),
        make_cell("code", "plot_roc_curves(results, '../outputs/figures/week4_roc_curves.png')\ncm_dict = {name: info['confusion_matrix'] for name, info in results.items()}\nplot_confusion_matrices(cm_dict, '../outputs/figures/week4_confusion_matrices.png')\nif feat_imp is not None:\n    plot_feature_importance(feat_names, feat_imp, top_n=12, filename='../outputs/figures/week4_feature_importance.png')")
    ]
    create_notebook("week4_machine_learning.ipynb", w4_cells)

    # ---------------------------------------------------------
    # WEEK 5 NOTEBOOK
    # ---------------------------------------------------------
    w5_cells = [
        make_cell("markdown", "# Week 5 — Deep Learning\n**Student Name:** Sindhu Patil | **Internship:** Data Science\n\n## Objective\nConstruct a PyTorch Feed-Forward Neural Network (`Dense(64)->ReLU->Dropout(0.3)->Dense(32)->ReLU->Dropout(0.2)->Dense(1)->Sigmoid`), train with BCE Loss & Adam, track loss/accuracy curves, and benchmark against Week 4 best traditional ML model."),
        make_cell("code", "import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nfrom src.preprocessing import preprocess_telco_data, get_train_test_split\nfrom src.deep_learning import train_deep_learning_model\nfrom src.visualization import plot_dl_training_history\n\ndf_clean = pd.read_csv('../data/processed/cleaned_telco_churn.csv')\nX_scaled, y, feat_names, scaler, ids = preprocess_telco_data(df_clean)\nX_train, X_test, y_train, y_test = get_train_test_split(X_scaled, y, test_size=0.2)"),
        make_cell("markdown", "### 1. Training PyTorch Neural Network"),
        make_cell("code", "dl_metrics = train_deep_learning_model(X_train, X_test, y_train, y_test, epochs=40, batch_size=64)\nprint('Neural Network Test Accuracy:', dl_metrics['Accuracy'])\nprint('Neural Network Test F1-Score:', dl_metrics['F1-Score'])\nprint('Neural Network Test ROC-AUC:', dl_metrics['ROC-AUC'])"),
        make_cell("markdown", "### 2. Loss & Accuracy History Plot"),
        make_cell("code", "plot_dl_training_history(\n    dl_metrics['train_losses'], dl_metrics['val_losses'],\n    dl_metrics['train_accs'], dl_metrics['val_accs'],\n    '../outputs/figures/week5_dl_history.png'\n)")
    ]
    create_notebook("week5_deep_learning.ipynb", w5_cells)

    # ---------------------------------------------------------
    # WEEK 6 NOTEBOOK
    # ---------------------------------------------------------
    w6_cells = [
        make_cell("markdown", "# Week 6 — Integrative Capstone & Business Strategy\n**Student Name:** Sindhu Patil | **Internship:** Data Science\n\n## Objective\nIntegrate the entire Data Science pipeline end-to-end: data cleaning, EDA, customer segmentation, traditional machine learning, deep learning, master benchmark comparison, and executive strategic business recommendations."),
        make_cell("code", "import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nfrom src.evaluation import build_master_comparison_table, generate_business_recommendations\nfrom src.visualization import plot_master_model_comparison\n\nml_comp_df = pd.read_csv('../outputs/results/ml_model_comparison.csv')\n# Load Master Comparison\nmaster_df = pd.read_csv('../outputs/results/master_model_comparison.csv')\nmaster_df"),
        make_cell("markdown", "### 1. Master Model Benchmark Chart"),
        make_cell("code", "plot_master_model_comparison(master_df, '../outputs/figures/week6_master_model_comparison.png')"),
        make_cell("markdown", "### 2. Final Business Recommendations & Action Plan\n- Contract Incentives for Month-to-Month users.\n- Value Bundling for Fiber Optic subscribers.\n- Early retention triggers during initial 12 months tenure.")
    ]
    create_notebook("week6_capstone.ipynb", w6_cells)


if __name__ == "__main__":
    build_all_notebooks()
