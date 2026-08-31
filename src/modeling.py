"""
Supervised Learning Machine Learning Pipeline Module
Author: Sindhu Patil (Data Science Intern)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score, precision_recall_curve
)
import joblib

MODELS_DIR = os.path.join("outputs", "models")
RESULTS_DIR = os.path.join("outputs", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Creates a Scikit-learn ColumnTransformer:
    - StandardScaler for numerical features
    - OneHotEncoder for categorical features (handling unknown categories safely)
    Prevents data leakage by encapsulating transformation logic inside Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor


def build_model_pipeline(classifier, preprocessor):
    """Builds a complete ML Pipeline with preprocessing and classifier."""
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return model_pipeline


def evaluate_cv_performance(model_pipeline, X, y, cv_folds=5, random_state=42):
    """
    Performs 5-Fold Stratified Cross-Validation on the pipeline.
    Returns dictionary of mean scores and standard deviations for Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model_pipeline, X, y, cv=cv, scoring=scoring)

    summary = {}
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        summary[metric] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores))
        }

    return summary, cv_results


def evaluate_test_performance(model_pipeline, X_train, X_test, y_train, y_test, model_name="Logistic Regression"):
    """
    Fits pipeline on training data ONLY and evaluates test set metrics.
    Returns metrics dict, y_pred, y_prob, confusion_matrix.
    """
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    if hasattr(model_pipeline, "predict_proba"):
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'Model': model_name,
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1_Score': float(f1),
        'ROC_AUC': float(roc_auc),
        'Average_Precision': float(avg_prec),
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1])
    }

    # Save trained model artifact
    model_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_model.joblib")
    joblib.dump(model_pipeline, model_path)
    print(f"Saved trained {model_name} pipeline model to {model_path}")

    return metrics, y_pred, y_prob, cm


def extract_logistic_coefficients(model_pipeline, numerical_features, categorical_features):
    """
    Extracts transformed feature names and mapped coefficients from fitted Logistic Regression pipeline.
    """
    ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numerical_features + cat_feature_names
    coefs = model_pipeline.named_steps['classifier'].coef_[0]

    df_coef = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefs,
        'Abs_Coef': np.abs(coefs)
    }).sort_values(by='Abs_Coef', ascending=False)

    return df_coef


if __name__ == "__main__":
    df = pd.read_csv(os.path.join("data", "processed", "cleaned_telco_churn.csv"))
    X = df.drop(columns=['customerID', 'Churn'])
    y = (df['Churn'] == 'Yes').astype(int)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    lr_pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42), preprocessor)

    cv_sum, _ = evaluate_cv_performance(lr_pipe, X, y)
    metrics, y_pred, y_prob, cm = evaluate_test_performance(lr_pipe, X_train, X_test, y_train, y_test)
    df_coef = extract_logistic_coefficients(lr_pipe, num_cols, cat_cols)

    print("Test Set Metrics:\n", metrics)
    print("Top 5 Coefficients:\n", df_coef.head(5))
