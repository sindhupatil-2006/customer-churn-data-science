# Customer Churn Prediction and Customer Segmentation — Final Capstone Portfolio Project

**Student Name:** Sindhu Patil  
**Internship Program:** Data Science Internship (6-Week Program)  
**Target Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7,043 Records, 21 Features)  
**Technologies:** Python 3.14, Pandas, NumPy, Scikit-Learn, PyTorch, Keras 3, Matplotlib, Seaborn, Jupyter, python-docx, Joblib

---

## 📌 Master Capstone Overview
This repository contains the complete 6-week Data Science Internship Capstone Project on **Customer Churn Prediction and Customer Segmentation**, developed by **Sindhu Patil**. It presents an end-to-end, reproducible Data Science workflow combining data acquisition, cleaning, exploratory visualization, unsupervised K-Means customer segmentation ($K=3$), baseline Logistic Regression classification, and Keras Deep Learning neural networks.

---

## 📁 Repository Structure

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
│   ├── week5_deep_learning.ipynb                    # Week 5 Deep Learning Notebook
│   └── week6_capstone.ipynb                         # Week 6 Final Capstone Integration Notebook
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
│   │   ├── week5/                                   # Week 5 Deep Learning Plots
│   │   └── week6/                                   # Week 6 Capstone Plots
│   │       ├── capstone_pipeline.png
│   │       ├── final_churn_distribution.png
│   │       ├── eda_key_relationships.png
│   │       ├── cluster_segmentation_summary.png
│   │       ├── logistic_regression_confusion_matrix.png
│   │       ├── neural_network_training_history.png
│   │       ├── final_model_comparison.png
│   │       └── customer_risk_distribution.png
│   ├── models/
│   │   ├── kmeans_model.joblib                      # Trained K-Means model artifact
│   │   ├── logistic_regression_model.joblib         # Trained Logistic Regression model artifact
│   │   ├── random_forest_model.joblib               # Trained Random Forest model artifact
│   │   └── deep_learning_nn_model.keras             # Trained Keras Deep Neural Network model artifact
│   └── results/
│       ├── cluster_profiles_summary.csv             # Cluster profile metrics table
│       ├── week4_model_results.csv                  # Supervised model evaluation results CSV
│       ├── week5_results.csv                        # Deep Learning model evaluation results CSV
│       └── capstone_summary_dashboard.csv           # Master Capstone evaluation results CSV
├── reports/
│   ├── week1_report.docx                            # Week 1 Word Executive Report
│   ├── week2_report.docx                            # Week 2 Word Executive Report
│   ├── week3_report.docx                            # Week 3 Word Executive Report
│   ├── week4_report.docx                            # Week 4 Word Executive Report
│   ├── week5_deep_learning_report.docx              # Week 5 Word Executive Report
│   └── week6_capstone_report.docx                   # Week 6 Final Capstone Executive Report
├── requirements.txt                                 # Required Python dependencies
├── README.md                                        # Repository documentation
└── .gitignore                                       # Git ignore rules
```

---

## 🚀 End-to-End Capstone Workflow & Milestones

### Week 1 — Data Acquisition, Cleaning & Preprocessing
- **Audit:** 7,043 customer records × 21 columns. Identified 11 whitespace string records (`" "`) in `TotalCharges` (`tenure = 0`).
- **Treatment:** Converted blank strings to `NaN`, imputed using median ($1,397.47), verified 0 duplicates and 0 missing values.
- **Preprocessing:** Target `Churn` mapped (Yes: 1, No: 0; 26.54% overall churn rate). Categoricals One-Hot Encoded and numerical features scaled via `StandardScaler`.

### Week 2 — Exploratory Data Analysis & Visualization
- **Overall Churn Rate:** 26.54% (1,869 churners vs 5,174 retained).
- **Key Churn Drivers:** Month-to-Month contracts (42.71% churn), Fiber Optic internet (41.89% churn), Electronic Check payment (45.29% churn), early tenure < 12 months (47.44% churn).
- **Correlations:** `tenure` ($r = -0.352$), `MonthlyCharges` ($r = +0.193$), `TotalCharges` ($r = -0.199$).

### Week 3 — Unsupervised Learning & Customer Segmentation
- **Methodology:** K-Means Clustering ($K=3$, `random_state=42`) on 18 behavioral features (strictly excluding `Churn` and `customerID`).
- **Cluster Profiles ($K=3$):**
  - **Cluster 0 (High-Value Loyal Customers):** 2,362 customers (33.54%), Avg Tenure = 55.06 mos, Avg Monthly Bill = $89.62, Churn Rate = **15.37%**.
  - **Cluster 1 (New High-Charge At-Risk Customers):** 3,155 customers (44.80%), Avg Tenure = 16.26 mos, Avg Monthly Bill = $67.28, Churn Rate = **44.15%** (accounts for 74.5% of total company churners).
  - **Cluster 2 (Long-Term Budget Customers):** 1,526 customers (21.67%), Avg Tenure = 30.55 mos, Avg Monthly Bill = $21.08, Churn Rate = **7.40%**.

### Week 4 — Supervised Machine Learning
- **Models:** Logistic Regression (`Pipeline` with `StandardScaler` + `OneHotEncoder`) & Random Forest Classifier.
- **Logistic Regression Test Results (1,409 Samples):** Accuracy = **0.8055**, Precision = **0.6572**, Recall = **0.5588**, F1-Score = **0.6040**, ROC-AUC = **0.8419**.
- **Top Predictive Coefficients:** `tenure` (-1.2339), `Contract_Two year` (-0.7627), `InternetService_Fiber optic` (+0.6431), `Contract_Month-to-month` (+0.5873).

### Week 5 — Deep Learning Application
- **Architecture:** Keras Sequential Neural Network: Input(27) -> Dense(64, ReLU) -> Dropout(0.30) -> Dense(32, ReLU) -> Dropout(0.20) -> Dense(16, ReLU) -> Dense(1, Sigmoid). Total trainable parameters: 4,593.
- **Deep Learning Test Results (1,057 Samples):** Accuracy = **0.7701**, Precision = **0.5509**, **Recall = 0.7536**, **F1-Score = 0.6346**, **ROC-AUC = 0.8436**.
- **Recall Advantage:** Caught **211 out of 280 churners** in the test set, outperforming Logistic Regression recall (55.88%).

---

## 🏆 Master Model Performance Dashboard

| Model Framework | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **0.8055** | **0.6572** | 0.5588 | 0.6040 | 0.8419 | Interpretable linear baseline & feature coefficient analysis. |
| Random Forest Classifier | 0.7821 | 0.6159 | 0.4759 | 0.5370 | 0.8179 | Non-linear ensemble comparison. |
| **Deep Neural Network (Keras)** | 0.7701 | 0.5509 | **0.7536** | **0.6346** | **0.8436** | **Primary High-Recall Retention Warning System.** |

---

## 💡 Executive Business Recommendations

1. **Targeted Onboarding & Contract Conversion (Cluster 1):** Focus 80% of retention budget on Cluster 1 (44.15% churn). Offer a 15% billing discount to convert Month-to-Month subscribers into 1-Year plans.
2. **Bundled Value Services:** Include free `OnlineSecurity` and `TechSupport` with Fiber Optic service packages to reduce high early cancellation rates ($41.89\% \rightarrow <20\%$).
3. **Automated Payment Incentive:** Offer a $5/month bill credit for switching from Electronic Check to automated Credit Card or ACH bank transfers.
4. **Deploy Deep Learning Risk Scoring:** Utilize the Keras Deep Learning model (75.36% Recall) to score accounts dynamically and flag high-risk subscribers prior to cancellation.

---

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Capstone Asset Generator Script:**
   ```bash
   python generate_week6_assets.py
   ```

3. **Launch Executable Notebooks:**
   ```bash
   jupyter notebook notebooks/week6_capstone.ipynb
   ```

---

## 📄 Key Project Deliverables
- **Cleaned Dataset:** [`data/processed/cleaned_telco_churn.csv`](file:///c:/yurathna/data/processed/cleaned_telco_churn.csv)
- **Saved Model Artifacts:**
  - [`outputs/models/kmeans_model.joblib`](file:///c:/yurathna/outputs/models/kmeans_model.joblib)
  - [`outputs/models/logistic_regression_model.joblib`](file:///c:/yurathna/outputs/models/logistic_regression_model.joblib)
  - [`outputs/models/random_forest_model.joblib`](file:///c:/yurathna/outputs/models/random_forest_model.joblib)
  - [`outputs/models/deep_learning_nn_model.keras`](file:///c:/yurathna/outputs/models/deep_learning_nn_model.keras)
- **Master Capstone Dashboard CSV:** [`outputs/results/capstone_summary_dashboard.csv`](file:///c:/yurathna/outputs/results/capstone_summary_dashboard.csv)
- **Executive Word Reports:**
  - [`reports/week1_report.docx`](file:///c:/yurathna/reports/week1_report.docx)
  - [`reports/week2_report.docx`](file:///c:/yurathna/reports/week2_report.docx)
  - [`reports/week3_report.docx`](file:///c:/yurathna/reports/week3_report.docx)
  - [`reports/week4_report.docx`](file:///c:/yurathna/reports/week4_report.docx)
  - [`reports/week5_deep_learning_report.docx`](file:///c:/yurathna/reports/week5_deep_learning_report.docx)
  - [`reports/week6_capstone_report.docx`](file:///c:/yurathna/reports/week6_capstone_report.docx)
- **Capstone Notebook:** [`notebooks/week6_capstone.ipynb`](file:///c:/yurathna/notebooks/week6_capstone.ipynb)
- **Capstone Figures Directory:** [`outputs/figures/week6/`](file:///c:/yurathna/outputs/figures/week6/)
