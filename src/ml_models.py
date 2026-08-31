"""
Supervised Machine Learning Module for Churn Prediction
Author: Sindhu Patil (Data Science Intern)
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

MODELS_DIR = os.path.join("outputs", "models")
RESULTS_DIR = os.path.join("outputs", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_models(random_state=42):
    """Instantiates candidate machine learning classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
    }
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5):
    """
    Fits model, evaluates on test set, computes cross-validation scores,
    and returns comprehensive metrics dictionary.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc_score,
        "CV_F1_Mean": cv_scores.mean(),
        "CV_F1_Std": cv_scores.std(),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
        "confusion_matrix": cm,
        "model_object": model
    }
    return metrics


def train_and_compare_ml_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains all candidate ML models, compares their metrics, selects the best model,
    and saves results to disk.
    """
    models = get_models()
    results = {}
    summary_rows = []

    for name, model in models.items():
        print(f"Training {name}...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        
        summary_rows.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"],
            "CV_F1_Mean": metrics["CV_F1_Mean"]
        })

    comp_df = pd.DataFrame(summary_rows).sort_values(by="ROC-AUC", ascending=False)
    
    # Save comparison table CSV
    comp_csv_path = os.path.join(RESULTS_DIR, "ml_model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"Saved ML model comparison to {comp_csv_path}")

    # Select best model based on F1-Score & ROC-AUC balance
    best_model_name = comp_df.iloc[0]["Model"]
    best_model_obj = results[best_model_name]["model_object"]
    best_model_path = os.path.join(MODELS_DIR, "best_ml_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    print(f"Best ML Model: {best_model_name} (Saved to {best_model_path})")

    # Extract feature importance if available
    feature_importances = None
    if hasattr(best_model_obj, "feature_importances_"):
        feature_importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, "coef_"):
        feature_importances = np.abs(best_model_obj.coef_[0])

    return results, comp_df, best_model_name, best_model_path, feature_importances
