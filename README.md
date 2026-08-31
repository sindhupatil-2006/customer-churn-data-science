# Customer Churn Prediction and Customer Segmentation — Portfolio Project

**Student Name:** Sindhu Patil  
**Internship:** Data Science  
**Target Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7,043 Records, 21 Features)  
**Technologies:** Python 3.14, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Jupyter, python-docx, Joblib

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
│   └── week4_supervised_learning.ipynb              # Week 4 Supervised Learning Notebook
├── src/
│   ├── __init__.py                                  # Package initializer
│   ├── data_cleaning.py                             # Data loading, audit & cleaning module
│   ├── preprocessing.py                            # Categorical encoding & scaling module
│   ├── visualization.py                            # Statistical plotting & visualization module
│   ├── clustering.py                               # K-Means & PCA segmentation module
│   └── modeling.py                                 # Scikit-learn Pipeline modeling module
├── outputs/
│   ├── figures/
│   │   ├── week1/                                   # Week 1 Plots
│   │   ├── week2/                                   # Week 2 EDA Plots
│   │   ├── week3/                                   # Week 3 Clustering Plots
│   │   └── week4/                                   # Week 4 Classification Plots
│   │       ├── confusion_matrix.png
│   │       ├── roc_curve.png
│   │       ├── precision_recall_curve.png
│   │       ├── threshold_analysis.png
│   │       ├── model_comparison.png
│   │       └── top_logistic_coefficients.png
│   ├── models/
│   │   ├── kmeans_model.joblib                      # Trained K-Means model artifact
│   │   ├── logistic_regression_model.joblib         # Trained Logistic Regression model artifact
│   │   └── random_forest_model.joblib               # Trained Random Forest model artifact
│   └── results/
│       ├── cluster_profiles_summary.csv             # Cluster profile metrics table
│       └── week4_model_results.csv                  # Supervised model evaluation results CSV
├── reports/
│   ├── week1_report.docx                            # Week 1 Word Executive Report
│   ├── week2_report.docx                            # Week 2 Word Executive Report
│   ├── week3_report.docx                            # Week 3 Word Executive Report
│   └── week4_report.docx                            # Week 4 Word Executive Report
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

### Problem Statement & Methodology
Build a supervised binary classification pipeline to predict subscriber churn (`Churn` -> 1: Yes, 0: No). Data leakage was completely eliminated by encapsulating preprocessing (`StandardScaler` for numerical attributes, `OneHotEncoder` for categorical predictors) and classification (`LogisticRegression`) inside a unified Scikit-learn `Pipeline`. Data was split into 5,634 training rows and 1,409 testing rows using an 80/20 stratified split.

### 5-Fold Stratified Cross-Validation Performance (Logistic Regression)
- **Accuracy:** Mean = **0.8056** $\pm$ 0.0109
- **Precision:** Mean = **0.6590** $\pm$ 0.0216
- **Recall:** Mean = **0.5532** $\pm$ 0.0317
- **F1-Score:** Mean = **0.6013** $\pm$ 0.0273
- **ROC-AUC:** Mean = **0.8449** $\pm$ 0.0134

---

### Model Performance Comparison (Test Set Evaluation — 1,409 Samples)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **0.8055** | **0.6572** | **0.5588** | **0.6040** | **0.8419** | **0.6543** |
| Random Forest Classifier | 0.7821 | 0.6159 | 0.4759 | 0.5370 | 0.8179 | 0.6133 |

*Model Selection Decision:* **Logistic Regression was selected as the primary baseline classifier** because it outperformed Random Forest across Recall (**0.5588** vs 0.4759), F1-Score (**0.6040** vs 0.5370), and ROC-AUC (**0.8419** vs 0.8179), while offering clear coefficient interpretability.

---

### Test Set Confusion Matrix Breakdown (Logistic Regression)

| | Predicted Retained (`0`) | Predicted Churned (`1`) |
| :--- | :---: | :---: |
| **Actual Retained (`0`)** | **True Negative (TN): 926** | False Positive (FP): 109 |
| **Actual Churned (`1`)** | False Negative (FN): 165 | **True Positive (TP): 209** |

---

### Key Feature Coefficients (Logistic Regression)

- **Protective Factors against Churn (Negative Coefficients):**
  1. `tenure`: **-1.2339** (Longer customer tenure strongly lowers churn risk)
  2. `Contract_Two year`: **-0.7627** (Two-year contract strongly insulates against churn)
  3. `InternetService_DSL`: **-0.6402** (DSL service lowers churn risk compared to Fiber)
- **Risk Factors for Churn (Positive Coefficients):**
  1. `InternetService_Fiber optic`: **+0.6431** (Fiber optic service increases churn likelihood)
  2. `Contract_Month-to-month`: **+0.5873** (Month-to-month contract increases churn likelihood)
  3. `TotalCharges`: **+0.5103** (High cumulative spend increases churn sensitivity)

---

### Executive Business Recommendations
1. **Dynamic Risk Scoring:** Use model probability scores ($y_{prob}$) to rank customers by churn risk in CRM dashboards.
2. **Targeted Threshold Optimization:** Adjust the classification threshold from 0.5 to 0.35 in high-risk campaigns to increase Recall (capturing ~70% of churners) for proactive retention offers.
3. **Focused Intervention:** Priority retention budget should target Month-to-Month Fiber Optic subscribers with tenure under 12 months.

---

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Asset & Report Generator Scripts:**
   ```bash
   python generate_week4_assets.py
   ```

3. **Launch Executable Notebooks:**
   ```bash
   jupyter notebook notebooks/week4_supervised_learning.ipynb
   ```

---

## 📄 Key Deliverables
- **Cleaned Data File:** [`data/processed/cleaned_telco_churn.csv`](file:///c:/yurathna/data/processed/cleaned_telco_churn.csv)
- **Trained Model Artifacts:**
  - [`outputs/models/logistic_regression_model.joblib`](file:///c:/yurathna/outputs/models/logistic_regression_model.joblib)
  - [`outputs/models/random_forest_model.joblib`](file:///c:/yurathna/outputs/models/random_forest_model.joblib)
- **Model Results CSV:** [`outputs/results/week4_model_results.csv`](file:///c:/yurathna/outputs/results/week4_model_results.csv)
- **Executive Word Reports:**
  - [`reports/week1_report.docx`](file:///c:/yurathna/reports/week1_report.docx)
  - [`reports/week2_report.docx`](file:///c:/yurathna/reports/week2_report.docx)
  - [`reports/week3_report.docx`](file:///c:/yurathna/reports/week3_report.docx)
  - [`reports/week4_report.docx`](file:///c:/yurathna/reports/week4_report.docx)
- **Week 4 Notebook:** [`notebooks/week4_supervised_learning.ipynb`](file:///c:/yurathna/notebooks/week4_supervised_learning.ipynb)
- **Visualization Figures Directory:** [`outputs/figures/week4/`](file:///c:/yurathna/outputs/figures/week4/)
