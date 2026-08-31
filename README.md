# Customer Churn Prediction and Customer Segmentation — Week 1

**Student Name:** Sindhu Patil  
**Internship:** Data Science  
**Week:** 1 — Data Acquisition, Cleaning and Preprocessing  
**Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  

---

## 📌 Project Overview
This repository contains the completed **Week 1** project for the Data Science Internship. The objective of Week 1 is to acquire, inspect, clean, preprocess, and encode the IBM Telco Customer Churn dataset, establishing a sanitized, production-ready dataset for subsequent exploratory analysis and machine learning.

---

## 📂 Project Structure

```
customer-churn-data-science/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Original dataset (7,043 rows, 21 cols)
│   └── processed/
│       └── cleaned_telco_churn.csv                   # Cleaned & processed dataset
├── notebooks/
│   └── week1_data_cleaning.ipynb                    # Executable Week 1 Notebook
├── src/
│   ├── __init__.py                                  # Package initializer
│   ├── data_cleaning.py                             # Data loading, audit & cleaning module
│   └── preprocessing.py                            # Categorical encoding & scaling module
├── outputs/
│   └── figures/
│       └── week1/
│           └── week1_churn_distribution.png         # Target distribution plot
├── reports/
│   └── week1_report.docx                            # Formal Word Executive Report
├── requirements.txt                                 # Required Python dependencies
├── README.md                                        # Week 1 documentation
└── .gitignore                                       # Git ignore file
```

---

## 📊 Week 1 Dataset & Cleaning Summary

- **Source URL:** `https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Total Records:** **7,043**
- **Total Attributes:** **21**
- **Target Variable:** `Churn` (Binary: `"Yes"` / `"No"`)
- **Missing / Erroneous Values:** **11 whitespace string records (`" "`)** discovered in `TotalCharges` (all corresponding to new accounts with `tenure = 0`).
- **Treatment:** Converted blank strings to `NaN` and imputed using column **median** (**$1,397.47**).
- **Duplicate Records:** **0 duplicate rows**.
- **Outliers:** `tenure` (0–72 mos), `MonthlyCharges` ($18.25–$118.75), `TotalCharges` ($18.80–$8,684.80) retained as legitimate customer transactions.
- **Preprocessing:** Target mapped to binary `(1, 0)`, binary features mapped, multi-class categoricals One-Hot Encoded (30 total features), and continuous features scaled via `StandardScaler`.

---

## 🛠️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Data Cleaning & Preprocessing Script:**
   ```bash
   python src/data_cleaning.py
   python src/preprocessing.py
   ```

3. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/week1_data_cleaning.ipynb
   ```

---

## 📄 Deliverables
- **Processed CSV:** [`data/processed/cleaned_telco_churn.csv`](file:///c:/yurathna/data/processed/cleaned_telco_churn.csv)
- **Executive Word Report:** [`reports/week1_report.docx`](file:///c:/yurathna/reports/week1_report.docx)
- **Notebook:** [`notebooks/week1_data_cleaning.ipynb`](file:///c:/yurathna/notebooks/week1_data_cleaning.ipynb)
