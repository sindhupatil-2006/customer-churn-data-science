# Customer Churn Prediction and Customer Segmentation — Portfolio Project

**Student Name:** Sindhu Patil  
**Internship:** Data Science  
**Target Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7,043 Records, 21 Features)  
**Technologies:** Python 3.14, Pandas, NumPy, Scikit-Learn, PyTorch, Keras 3, Matplotlib, Seaborn, Jupyter, python-docx, Joblib

---

## 📌 Repository Overview
This repository contains a professional Data Science internship portfolio project for **Customer Churn Prediction and Customer Segmentation**. Built progressively across weekly milestones, it delivers reproducible Python source packages, executable Jupyter notebooks, high-resolution statistical visualizations, saved model artifacts, and executive Word reports.

---

## 📅 Project Structure & Roadmap

```
customer-churn-data-science/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Raw dataset (7,043 rows, 21 cols)
│   └── processed/
│       └── cleaned_telco_churn.csv                   # Cleaned & processed dataset
├── notebooks/
│   ├── week1_data_cleaning.ipynb                    # Week 1 Data Acquisition & Cleaning Notebook
│   ├── week2_eda.ipynb                              # Week 2 Exploratory Data Analysis Notebook
│   ├── week3_clustering.ipynb                       # Week 3 Customer Clustering Notebook
│   ├── week4_supervised_learning.ipynb              # Week 4 Supervised Learning Notebook
│   └── week5_deep_learning.ipynb                    # Week 5 Deep Learning Notebook
├── src/
│   ├── __init__.py                                  # Package initializer
│   ├── data_cleaning.py                             # Data loading, audit & cleaning module
│   ├── preprocessing.py                            # Categorical encoding & scaling module
│   ├── visualization.py                            # Statistical plotting & visualization module
│   ├── clustering.py                               # K-Means & PCA segmentation module
│   ├── modeling.py                                 # Scikit-learn Pipeline modeling module
│   └── deep_learning.py                            # Keras Deep Learning Neural Network module
├── outputs/
│   ├── figures/
│   │   ├── week1/                                   # Week 1 Plots
│   │   ├── week2/                                   # Week 2 EDA Plots
│   │   ├── week3/                                   # Week 3 Clustering Plots
│   │   ├── week4/                                   # Week 4 Classification Plots
│   │   └── week5/                                   # Week 5 Deep Learning Plots
│   │       ├── neural_network_architecture.png
│   │       ├── training_history_loss.png
│   │       ├── training_history_accuracy.png
│   │       ├── confusion_matrix.png
│   │       ├── roc_curve.png
│   │       ├── precision_recall_curve.png
│   │       ├── threshold_analysis.png
│   │       └── week4_vs_week5_comparison.png
│   ├── models/
│   │   ├── kmeans_model.joblib                      # Trained K-Means model artifact
│   │   ├── logistic_regression_model.joblib         # Trained Logistic Regression model artifact
│   │   ├── random_forest_model.joblib               # Trained Random Forest model artifact
│   │   └── deep_learning_nn_model.keras             # Trained Keras Deep Neural Network model artifact
│   └── results/
│       ├── cluster_profiles_summary.csv             # Cluster profile metrics table
│       ├── week4_model_results.csv                  # Supervised model evaluation results CSV
│       └── week5_results.csv                        # Deep Learning model evaluation results CSV
├── reports/
│   ├── week1_report.docx                            # Week 1 Word Executive Report
│   ├── week2_report.docx                            # Week 2 Word Executive Report
│   ├── week3_report.docx                            # Week 3 Word Executive Report
│   ├── week4_report.docx                            # Week 4 Word Executive Report
│   └── week5_deep_learning_report.docx              # Week 5 Word Executive Report
├── requirements.txt                                 # Required Python dependencies
├── README.md                                        # Repository documentation
└── .gitignore                                       # Git ignore rules
```

---

## 📊 Week 1 — Data Acquisition, Cleaning & Preprocessing
- **Source:** IBM Telco Customer Churn Dataset (7,043 rows × 21 columns).
- **Data Quality Audit:** Identified 11 whitespace string records (`" "`) in `TotalCharges` corresponding to newly onboarded accounts (`tenure = 0`).
- **Treatment:** Converted blank strings to `NaN` and imputed using column median ($1,397.47). Verified zero duplicate rows.
- **Preprocessing:** Mapped binary target `Churn` (Yes: 1, No: 0; 26.54% overall churn rate), encoded categorical variables (30 total features), and scaled continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.

---

## 📈 Week 2 — Exploratory Data Analysis & Visualization
- **Target Imbalance:** Retained 5,174 (73.46%) vs Churned 1,869 (26.54%).
- **Contract Impact:** Month-to-Month contracts churn at 42.71% vs 2.83% for 2-Year contracts.
- **Internet Service Impact:** Fiber Optic subscribers churn at 41.89% vs 18.96% for DSL.
- **Payment Method Impact:** Electronic Check users churn at 45.29%.
- **Tenure Impact:** 0–12 Month subscribers churn at 47.44% vs 6.61% for 61–72 Months.
- **Correlations:** Tenure ($r = -0.352$), MonthlyCharges ($r = +0.193$), TotalCharges ($r = -0.199$).

---

## 🤖 Week 3 — Unsupervised Learning & Customer Segmentation
- **Model:** K-Means Clustering ($K=3$, `random_state=42`).
- **Cluster 0 (High-Value Loyal Customers):** 2,362 customers (33.54%), Avg Tenure = 55.06 mos, Avg Monthly Bill = $89.62, Churn Rate = **15.37%**.
- **Cluster 1 (New High-Charge At-Risk Customers):** 3,155 customers (44.80%), Avg Tenure = 16.26 mos, Avg Monthly Bill = $67.28, Churn Rate = **44.15%**.
- **Cluster 2 (Long-Term Budget Customers):** 1,526 customers (21.67%), Avg Tenure = 30.55 mos, Avg Monthly Bill = $21.08, Churn Rate = **7.40%**.

---

## 🎯 Week 4 — Supervised Learning Model Implementation
- **Primary Model:** Logistic Regression (`Pipeline` with `StandardScaler` + `OneHotEncoder`).
- **5-Fold CV Scores:** Accuracy = 0.8056, Precision = 0.6590, Recall = 0.5532, F1 = 0.6013, ROC-AUC = 0.8449.
- **Test Performance (1,409 Samples):** Accuracy = 0.8055, Precision = 0.6572, Recall = 0.5588, F1 = 0.6040, ROC-AUC = 0.8419.
- **Top Predictive Coefficients:** `tenure` (-1.2339), `Contract_Two year` (-0.7627), `InternetService_Fiber optic` (+0.6431), `Contract_Month-to-month` (+0.5873).

---

## 🧠 Week 5 — Deep Learning Application in Data Science

### Problem Statement & Architecture
Build a deep Feed-Forward Neural Network in Keras to predict customer churn (`Churn` -> 1: Yes, 0: No). Data was split into **70% Train (4,930 rows)**, **15% Validation (1,056 rows)**, and **15% Test (1,057 rows)** using target stratification. Preprocessing (`StandardScaler` for numerical attributes, `OneHotEncoder` for categorical predictors) was fitted strictly on training data (`X_train`) to eliminate data leakage.

```
Input (27 Features)
   ↓
Dense(64, ReLU) ──> Dropout(0.30)
   ↓
Dense(32, ReLU) ──> Dropout(0.20)
   ↓
Dense(16, ReLU)
   ↓
Output Dense(1, Sigmoid)
```
- **Total Parameters:** 4,593 (all trainable).
- **Optimization & Loss:** Adam Optimizer ($lr = 0.001$), Binary Cross-Entropy loss, balanced class weighting (`compute_class_weight`).
- **Callbacks:** `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`.

---

### Test Set Performance & Framework Comparison (1,057 Test Samples)

| Model Framework | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Week 4 Logistic Regression | **0.8055** | **0.6572** | 0.5588 | 0.6040 | 0.8419 | **0.6543** |
| **Week 5 Deep Learning Neural Network** | 0.7701 | 0.5509 | **0.7536** | **0.6346** | **0.8436** | 0.6315 |

*Analytical Finding:* The Keras Deep Neural Network with class weighting achieved an outstanding **75.36% Recall** (capturing **211 out of 280 churners** in the test set, compared to 55.88% for Logistic Regression) and a **higher F1-Score (0.6346 vs 0.6040)** and **higher ROC-AUC (0.8436 vs 0.8419)**.

---

### Test Set Confusion Matrix Breakdown (Deep Learning Neural Network)

| | Predicted Retained (`0`) | Predicted Churned (`1`) |
| :--- | :---: | :---: |
| **Actual Retained (`0`)** | **True Negative (TN): 603** | False Positive (FP): 174 |
| **Actual Churned (`1`)** | False Negative (FN): 69 | **True Positive (TP): 211** |

---

### Executive Business Recommendations
1. **High-Recall Retention Workflow:** For proactive retention campaigns where missing a churning customer incurs high lifetime value loss, the **Deep Learning Neural Network is the superior choice**, capturing 3 out of every 4 churners.
2. **Threshold Fine-Tuning:** In cost-sensitive retention campaigns, tuning the neural network probability threshold to **0.55** balances Precision (0.601) and Recall (0.693).

---

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Asset & Report Generator Scripts:**
   ```bash
   python generate_week5_assets.py
   ```

3. **Launch Executable Notebooks:**
   ```bash
   jupyter notebook notebooks/week5_deep_learning.ipynb
   ```

---

## 📄 Key Deliverables
- **Cleaned Data File:** [`data/processed/cleaned_telco_churn.csv`](file:///c:/yurathna/data/processed/cleaned_telco_churn.csv)
- **Trained Model Artifacts:**
  - [`outputs/models/logistic_regression_model.joblib`](file:///c:/yurathna/outputs/models/logistic_regression_model.joblib)
  - [`outputs/models/random_forest_model.joblib`](file:///c:/yurathna/outputs/models/random_forest_model.joblib)
  - [`outputs/models/deep_learning_nn_model.keras`](file:///c:/yurathna/outputs/models/deep_learning_nn_model.keras)
- **Model Results CSV:** [`outputs/results/week5_results.csv`](file:///c:/yurathna/outputs/results/week5_results.csv)
- **Executive Word Reports:**
  - [`reports/week1_report.docx`](file:///c:/yurathna/reports/week1_report.docx)
  - [`reports/week2_report.docx`](file:///c:/yurathna/reports/week2_report.docx)
  - [`reports/week3_report.docx`](file:///c:/yurathna/reports/week3_report.docx)
  - [`reports/week4_report.docx`](file:///c:/yurathna/reports/week4_report.docx)
  - [`reports/week5_deep_learning_report.docx`](file:///c:/yurathna/reports/week5_deep_learning_report.docx)
- **Week 5 Notebook:** [`notebooks/week5_deep_learning.ipynb`](file:///c:/yurathna/notebooks/week5_deep_learning.ipynb)
- **Visualization Figures Directory:** [`outputs/figures/week5/`](file:///c:/yurathna/outputs/figures/week5/)
