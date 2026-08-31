"""
End-to-End Execution Pipeline Script
Author: Sindhu Patil (Data Science Intern)
Project: Customer Churn Prediction and Customer Segmentation
"""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("."))

from src.data_cleaning import acquire_dataset, load_raw_data, inspect_raw_data, clean_telco_data
from src.preprocessing import preprocess_telco_data, get_train_test_split
from src.visualization import (
    plot_churn_distribution, plot_demographics_vs_churn, plot_services_and_contract_vs_churn,
    plot_financial_distributions, plot_correlation_heatmap, plot_elbow_and_silhouette,
    plot_cluster_pca, plot_roc_curves, plot_confusion_matrices, plot_feature_importance,
    plot_dl_training_history, plot_master_model_comparison
)
from src.clustering import find_optimal_k, train_kmeans, profile_clusters, apply_pca
from src.ml_models import train_and_compare_ml_models
from src.deep_learning import train_deep_learning_model
from src.evaluation import build_master_comparison_table, generate_business_recommendations


def main():
    print("==================================================================")
    print("STARTING 6-WEEK DATA SCIENCE INTERNSHIP PIPELINE EXECUTION")
    print("Student Name: Sindhu Patil | Project: Customer Churn & Segmentation")
    print("==================================================================")

    # ---------------------------------------------------------
    # WEEK 1: DATA ACQUISITION & CLEANING
    # ---------------------------------------------------------
    print("\n--- [WEEK 1] Data Acquisition & Cleaning ---")
    raw_path = acquire_dataset()
    df_raw = load_raw_data(raw_path)
    raw_info = inspect_raw_data(df_raw)
    print(f"Raw Dataset Shape: {raw_info['shape']}")
    df_clean = clean_telco_data(df_raw)
    
    # Save Week 1 Figure
    fig1 = plot_churn_distribution(df_clean)
    print(f"Generated Figure: {fig1}")

    # ---------------------------------------------------------
    # WEEK 2: EXPLORATORY DATA ANALYSIS & VISUALIZATION
    # ---------------------------------------------------------
    print("\n--- [WEEK 2] Exploratory Data Analysis & Visualization ---")
    fig2 = plot_demographics_vs_churn(df_clean)
    fig3 = plot_services_and_contract_vs_churn(df_clean)
    fig4 = plot_financial_distributions(df_clean)
    
    # Get numeric correlations
    X_num, _, _, _, _ = preprocess_telco_data(df_clean, drop_id=True)
    X_num['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    fig5 = plot_correlation_heatmap(X_num[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner', 'Dependents', 'Churn']])
    print(f"Generated Week 2 Figures: {fig2}, {fig3}, {fig4}, {fig5}")

    # ---------------------------------------------------------
    # WEEK 3: UNSUPERVISED LEARNING & CUSTOMER SEGMENTATION
    # ---------------------------------------------------------
    print("\n--- [WEEK 3] Unsupervised Learning & Clustering ---")
    X_scaled, y, feat_names, scaler, customer_ids = preprocess_telco_data(df_clean, drop_id=True)
    
    k_range, inertias, sil_scores = find_optimal_k(X_scaled, k_range=range(2, 7))
    fig6 = plot_elbow_and_silhouette(k_range, inertias, sil_scores)
    
    # Train K-Means with optimal k=3
    best_k = 3
    kmeans_model, cluster_labels, kmeans_path = train_kmeans(X_scaled, n_clusters=best_k)
    profiles_df, df_clustered, cluster_map = profile_clusters(df_clean, cluster_labels)
    print("\nCluster Profiles Summary:\n", profiles_df[['Cluster', 'Segment_Name', 'Customer_Count', 'Avg_Tenure', 'Avg_Monthly_Charges', 'Churn_Rate']])
    
    # PCA Plot
    X_pca, pca_obj = apply_pca(X_scaled, n_components=2)
    centroids_pca = pca_obj.transform(kmeans_model.cluster_centers_)
    cluster_names = [cluster_map[i] for i in range(best_k)]
    fig7 = plot_cluster_pca(X_pca, cluster_labels, centroids_pca, cluster_names)
    print(f"Generated Week 3 Figures: {fig6}, {fig7}")

    # ---------------------------------------------------------
    # WEEK 4: SUPERVISED MACHINE LEARNING
    # ---------------------------------------------------------
    print("\n--- [WEEK 4] Supervised Machine Learning ---")
    X_train, X_test, y_train, y_test = get_train_test_split(X_scaled, y, test_size=0.2)
    
    ml_results, ml_comp_df, best_ml_name, best_ml_path, feat_imp = train_and_compare_ml_models(
        X_train, X_test, y_train, y_test, feat_names
    )
    print("\nML Models Benchmark:\n", ml_comp_df)

    # ML Figures
    fig8 = plot_roc_curves(ml_results)
    cm_dict = {name: info['confusion_matrix'] for name, info in ml_results.items()}
    fig9 = plot_confusion_matrices(cm_dict)
    if feat_imp is not None:
        fig10 = plot_feature_importance(feat_names, feat_imp, top_n=12)
    print(f"Generated Week 4 Figures: {fig8}, {fig9}")

    # ---------------------------------------------------------
    # WEEK 5: DEEP LEARNING
    # ---------------------------------------------------------
    print("\n--- [WEEK 5] Deep Learning (PyTorch Architecture) ---")
    dl_metrics = train_deep_learning_model(X_train, X_test, y_train, y_test, epochs=40, batch_size=64)
    print(f"Deep Learning Metrics: Accuracy={dl_metrics['Accuracy']:.4f}, Precision={dl_metrics['Precision']:.4f}, Recall={dl_metrics['Recall']:.4f}, F1={dl_metrics['F1-Score']:.4f}, ROC-AUC={dl_metrics['ROC-AUC']:.4f}")
    
    fig11 = plot_dl_training_history(
        dl_metrics['train_losses'], dl_metrics['val_losses'],
        dl_metrics['train_accs'], dl_metrics['val_accs']
    )
    print(f"Generated Week 5 Figure: {fig11}")

    # ---------------------------------------------------------
    # WEEK 6: INTEGRATIVE CAPSTONE & FINAL EVALUATION
    # ---------------------------------------------------------
    print("\n--- [WEEK 6] Integrative Capstone & Master Benchmark ---")
    master_comp_df, master_csv = build_master_comparison_table(ml_comp_df, dl_metrics)
    print("\nMaster Model Performance Benchmark:\n", master_comp_df)
    
    fig12 = plot_master_model_comparison(master_comp_df)
    recommendations = generate_business_recommendations(master_comp_df, profiles_df)
    print(f"Generated Week 6 Master Figure: {fig12}")

    print("\n==================================================================")
    print("PIPELINE EXECUTION COMPLETE SUCCESSFULLY!")
    print("All figures, models, and results saved in outputs/")
    print("==================================================================")


if __name__ == "__main__":
    main()
