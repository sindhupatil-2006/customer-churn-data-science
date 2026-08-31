"""
Master Model Evaluation & Capstone Integration Module
Author: Sindhu Patil (Data Science Intern)
"""

import os
import json
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join("outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_master_comparison_table(ml_comp_df, dl_metrics):
    """
    Combines Traditional Machine Learning metrics with Deep Learning metrics
    into a unified benchmark comparison table.
    """
    dl_row = pd.DataFrame([{
        "Model": dl_metrics["Model"],
        "Accuracy": dl_metrics["Accuracy"],
        "Precision": dl_metrics["Precision"],
        "Recall": dl_metrics["Recall"],
        "F1-Score": dl_metrics["F1-Score"],
        "ROC-AUC": dl_metrics["ROC-AUC"],
        "CV_F1_Mean": np.nan  # DL uses Validation Set Early Stopping
    }])

    master_df = pd.concat([ml_comp_df, dl_row], ignore_index=True)
    master_df = master_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    master_csv_path = os.path.join(RESULTS_DIR, "master_model_comparison.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"Saved Master Model Comparison Table to {master_csv_path}")

    return master_df, master_csv_path


def generate_business_recommendations(master_df, cluster_profiles):
    """
    Formulates evidence-backed business insights and action plans.
    """
    best_model_name = master_df.iloc[0]["Model"]
    best_f1 = master_df.iloc[0]["F1-Score"]
    best_auc = master_df.iloc[0]["ROC-AUC"]

    recommendations = {
        "Best_Model": best_model_name,
        "Best_F1_Score": float(best_f1),
        "Best_ROC_AUC": float(best_auc),
        "Key_Churn_Drivers": [
            "Month-to-month contracts (highest churn risk)",
            "Fiber optic internet service (high price sensitivity)",
            "Absence of online security & tech support services",
            "Electronic check payment method"
        ],
        "Actionable_Strategies": [
            "Offer contract extension incentives (discounts for 1-year/2-year plans)",
            "Bundle free TechSupport and OnlineSecurity for Fiber Optic subscribers",
            "Promote automated payment methods (Auto Credit Card / Bank Transfer)",
            "Proactive customer support intervention during the first 12 months of tenure"
        ],
        "Customer_Segments_Summary": cluster_profiles.to_dict(orient="records")
    }

    summary_json_path = os.path.join(RESULTS_DIR, "capstone_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(recommendations, f, indent=4)
    print(f"Saved Capstone Summary & Recommendations to {summary_json_path}")

    return recommendations
