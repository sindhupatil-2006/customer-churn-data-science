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
│   └── week3_clustering.ipynb                       # Week 3 Customer Clustering Notebook
├── src/
│   ├── __init__.py                                  # Package initializer
│   ├── data_cleaning.py                             # Data loading, audit & cleaning module
│   ├── preprocessing.py                            # Categorical encoding & scaling module
│   ├── visualization.py                            # Statistical plotting & visualization module
│   └── clustering.py                               # K-Means & PCA segmentation module
├── outputs/
│   ├── figures/
│   │   ├── week1/
│   │   │   └── week1_churn_distribution.png         # Week 1 Target distribution plot
│   │   ├── week2/                                   # Week 2 EDA Plots
│   │   │   ├── churn_distribution.png
│   │   │   ├── churn_by_contract.png
│   │   │   ├── churn_by_internet_service.png
│   │   │   ├── churn_by_payment_method.png
│   │   │   ├── tenure_distribution.png
│   │   │   ├── monthly_charges_by_churn.png
│   │   │   ├── total_charges_by_churn.png
│   │   │   ├── correlation_heatmap.png
│   │   │   ├── multivariate_contract_charges.png
│   │   │   ├── demographics_churn.png
│   │   │   └── services_churn.png
│   │   └── week3/                                   # Week 3 Clustering Plots
│   │       ├── elbow_method.png
│   │       ├── silhouette_scores.png
│   │       ├── cluster_sizes.png
│   │       ├── cluster_pca_2d.png
│   │       ├── avg_tenure_by_cluster.png
│   │       ├── avg_monthly_charges_by_cluster.png
│   │       ├── churn_rate_by_cluster.png
│   │       └── contract_distribution_by_cluster.png
│   ├── models/
│   │   └── kmeans_model.joblib                      # Saved K-Means trained model artifact
│   └── results/
│       └── cluster_profiles_summary.csv             # Cluster profile metrics table
├── reports/
│   ├── week1_report.docx                            # Week 1 Word Executive Report
│   ├── week2_report.docx                            # Week 2 Word Executive Report
│   └── week3_report.docx                            # Week 3 Word Executive Report
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

### Objective & Methodology
Perform customer segmentation using K-Means clustering on scaled behavioral and subscription features. `Churn` target and `customerID` were strictly **excluded** during cluster formation to prevent data leakage. Candidate cluster counts ($K=2$ to $K=10$) were evaluated using the Elbow Method (Inertia) and Silhouette Scores.

### K Evaluation Metrics (Inertia & Silhouette)

| Number of Clusters (K) | Inertia (WCSS) | Silhouette Score |
| :---: | :---: | :---: |
| K = 2 | 138,761.84 | 0.3460 |
| **K = 3 (Selected)** | **119,898.75** | **0.2290** |
| K = 4 | 105,053.98 | 0.2502 |
| K = 5 | 99,455.02 | 0.2081 |
| K = 6 | 94,716.24 | 0.2040 |
| K = 7 | 91,276.09 | 0.1487 |
| K = 8 | 88,407.79 | 0.1418 |
| K = 9 | 85,894.84 | 0.1447 |
| K = 10 | 84,059.77 | 0.1370 |

*Reason for Selecting K=3:* $K=3$ represents the major Elbow reduction inflection point, provides robust silhouette stability, and forms 3 highly actionable, distinct business cohorts.

---

### Empirical Cluster Profiles & Churn Analysis ($K=3$)

| Cluster | Segment Name | Customer Count | Share (%) | Avg Tenure | Avg Monthly Bill | Avg Total Spend | Observed Churn Rate (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | **High-Value Loyal Customers** | 2,362 | 33.54% | 55.06 mos | $89.62 | $4,939.70 | **15.37%** |
| **Cluster 1** | **New High-Charge At-Risk Customers** | 3,155 | 44.80% | 16.26 mos | $67.28 | $1,072.73 | **44.15%** |
| **Cluster 2** | **Long-Term Budget Customers** | 1,526 | 21.67% | 30.55 mos | $21.08 | $668.10 | **7.40%** |

#### Cluster Behavioral Characteristics:
1. **Cluster 0 (High-Value Loyal Customers):**
   - High average tenure (55.06 months, median 59.0 mos) and high monthly spend ($89.62/mo).
   - 41.66% on Two-Year contracts; 60.50% use Fiber Optic.
   - Low observed churn rate (**15.37%**).
2. **Cluster 1 (New High-Charge At-Risk Customers):**
   - Short average tenure (16.26 months, median 12.0 mos) with moderate high spend ($67.28/mo).
   - 87.61% on Month-to-Month contracts; 52.84% Fiber Optic.
   - Highest observed churn rate (**44.15%**), accounting for 1,393 of the 1,869 total churners.
3. **Cluster 2 (Long-Term Budget Customers):**
   - Moderate tenure (30.55 months) with low monthly charges ($21.08/mo).
   - 100% have No Internet Service (basic landline voice only).
   - Lowest observed churn rate (**7.40%**).

---

### Executive Segment Business Strategies
- **New High-Charge At-Risk Customers (Cluster 1):** Focus 80% of retention budget here. Provide 0–12 month onboarding check-ins, offer 15% discounts to convert Month-to-Month plans to 1-Year contracts, and bundle free TechSupport.
- **High-Value Loyal Customers (Cluster 0):** Implement VIP loyalty rewards, long-term contract renewal incentives, and priority customer service to protect high-margin revenue.
- **Long-Term Budget Customers (Cluster 2):** Maintain low baseline pricing; offer optional digital add-ons without increasing core service costs.

---

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Asset & Report Generator Scripts:**
   ```bash
   python generate_week3_assets.py
   ```

3. **Launch Executable Notebooks:**
   ```bash
   jupyter notebook notebooks/week3_clustering.ipynb
   ```

---

## 📄 Key Deliverables
- **Cleaned Data File:** [`data/processed/cleaned_telco_churn.csv`](file:///c:/yurathna/data/processed/cleaned_telco_churn.csv)
- **Trained K-Means Model Artifact:** [`outputs/models/kmeans_model.joblib`](file:///c:/yurathna/outputs/models/kmeans_model.joblib)
- **Week 1 Report:** [`reports/week1_report.docx`](file:///c:/yurathna/reports/week1_report.docx)
- **Week 2 Report:** [`reports/week2_report.docx`](file:///c:/yurathna/reports/week2_report.docx)
- **Week 3 Report:** [`reports/week3_report.docx`](file:///c:/yurathna/reports/week3_report.docx)
- **Week 3 Notebook:** [`notebooks/week3_clustering.ipynb`](file:///c:/yurathna/notebooks/week3_clustering.ipynb)
- **Visualization Figures Directory:** [`outputs/figures/week3/`](file:///c:/yurathna/outputs/figures/week3/)
