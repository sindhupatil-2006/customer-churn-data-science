# Customer Churn Prediction and Customer Segmentation — Portfolio Project

**Student Name:** Sindhu Patil  
**Internship:** Data Science  
**Target Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7,043 Records, 21 Features)  
**Technologies:** Python 3.14, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Jupyter, python-docx

---

## 📌 Repository Overview
This repository contains a professional Data Science internship portfolio project for **Customer Churn Prediction and Customer Segmentation**. Built progressively across weekly milestones, it delivers reproducible Python source packages, executable Jupyter notebooks, high-resolution statistical visualizations, and executive Word reports.

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
│   └── week2_eda.ipynb                              # Week 2 Exploratory Data Analysis Notebook
├── src/
│   ├── __init__.py                                  # Package initializer
│   ├── data_cleaning.py                             # Data loading, audit & cleaning module
│   ├── preprocessing.py                            # Categorical encoding & scaling module
│   └── visualization.py                            # Statistical plotting & visualization module
├── outputs/
│   └── figures/
│       ├── week1/
│       │   └── week1_churn_distribution.png         # Week 1 Target distribution plot
│       └── week2/
│           ├── churn_distribution.png               # Target Churn Donut & Countplot
│           ├── churn_by_contract.png                # Contract type vs Churn
│           ├── churn_by_internet_service.png        # Internet service vs Churn
│           ├── churn_by_payment_method.png          # Payment method vs Churn
│           ├── tenure_distribution.png              # Tenure KDE & distribution by Churn
│           ├── monthly_charges_by_churn.png         # MonthlyCharges Boxplot & KDE
│           ├── total_charges_by_churn.png           # TotalCharges Boxplot & KDE
│           ├── correlation_heatmap.png              # Correlation matrix heatmap
│           ├── multivariate_contract_charges.png    # Contract + MonthlyCharges vs Churn
│           ├── demographics_churn.png               # Demographics subplots vs Churn
│           └── services_churn.png                   # Value-added services subplots vs Churn
├── reports/
│   ├── week1_report.docx                            # Week 1 Word Executive Report
│   └── week2_report.docx                            # Week 2 Word Executive Report
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

### Objective
Perform data-backed Exploratory Data Analysis (EDA) on `data/processed/cleaned_telco_churn.csv` to quantify relationships between demographics, contract terms, service subscriptions, payment methods, tenure cohorts, and customer churn.

### Key Descriptive Statistics (Numerical Features)

| Feature | Mean | Std Dev | Min | Q1 (25%) | Median | Q3 (75%) | Max | IQR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tenure** | 32.37 mos | 24.56 mos | 0.00 | 9.00 mos | 29.00 mos | 55.00 mos | 72.00 mos | 46.00 mos |
| **MonthlyCharges** | $64.76 | $30.09 | $18.25 | $35.59 | $70.35 | $89.85 | $118.75 | $54.35 |
| **TotalCharges** | $2,281.92 | $2,265.27 | $18.80 | $401.45 | $1,397.47 | $3,786.60 | $8,684.80 | $3,384.38 |

### Primary Empirical Findings & Churn Crosstabs

1. **Target Class Imbalance:**
   - **Retained Customers (`No`):** 5,174 (73.46%)
   - **Churned Customers (`Yes`):** 1,869 (26.54%)
   - Class imbalance ratio: ~2.77:1.

2. **Contract Type Impact:**
   - **Month-to-Month:** 3,875 customers, 1,655 churned $\rightarrow$ **42.71% Churn Rate**
   - **One Year:** 1,473 customers, 166 churned $\rightarrow$ **11.27% Churn Rate**
   - **Two Year:** 1,695 customers, 48 churned $\rightarrow$ **2.83% Churn Rate**

3. **Internet Service Provider Impact:**
   - **Fiber Optic:** 3,096 customers, 1,297 churned $\rightarrow$ **41.89% Churn Rate**
   - **DSL:** 2,421 customers, 459 churned $\rightarrow$ **18.96% Churn Rate**
   - **No Internet Service:** 1,526 customers, 113 churned $\rightarrow$ **7.40% Churn Rate**

4. **Payment Method Impact:**
   - **Electronic Check:** 2,365 customers, 1,071 churned $\rightarrow$ **45.29% Churn Rate**
   - **Mailed Check:** 1,612 customers, 308 churned $\rightarrow$ **19.11% Churn Rate**
   - **Bank Transfer (Automatic):** 1,544 customers, 258 churned $\rightarrow$ **16.71% Churn Rate**
   - **Credit Card (Automatic):** 1,522 customers, 232 churned $\rightarrow$ **15.24% Churn Rate**

5. **Tenure Cohort Churn Rates:**
   - **0–12 Months:** 2,186 customers, 1,037 churned $\rightarrow$ **47.44% Churn Rate**
   - **13–24 Months:** 1,024 customers, 294 churned $\rightarrow$ **28.71% Churn Rate**
   - **25–48 Months:** 1,594 customers, 325 churned $\rightarrow$ **20.39% Churn Rate**
   - **49–60 Months:** 832 customers, 120 churned $\rightarrow$ **14.42% Churn Rate**
   - **61–72 Months:** 1,407 customers, 93 churned $\rightarrow$ **6.61% Churn Rate**

6. **Numerical Feature Correlations with Churn:**
   - `tenure`: **-0.352** (Moderate negative correlation; longer tenure strongly reduces churn risk)
   - `MonthlyCharges`: **+0.193** (Positive correlation; higher monthly fees increase churn risk)
   - `TotalCharges`: **-0.199** (Negative correlation; cumulative spend aligns with long tenure)

### Executive Business Strategies Derived from EDA
- **Contract Transition Strategy:** Offer a 15% annual billing discount to incentivize Month-to-Month subscribers to transition into 1-Year or 2-Year contracts.
- **Fiber Optic Service Bundling:** Bundle free `OnlineSecurity` and `TechSupport` add-ons for Fiber Optic subscribers to lower high early cancellation rates.
- **Automatic Payment Incentive:** Offer a $5 monthly bill credit for switching from Electronic Check to Auto Credit Card or ACH bank transfers.

---

## 🛠️ How to Run

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Asset & Report Generator Scripts:**
   ```bash
   python generate_week2_assets.py
   ```

3. **Launch Executable Notebooks:**
   ```bash
   jupyter notebook notebooks/week2_eda.ipynb
   ```

---

## 📄 Key Deliverables
- **Cleaned Data File:** [`data/processed/cleaned_telco_churn.csv`](file:///c:/yurathna/data/processed/cleaned_telco_churn.csv)
- **Week 1 Report:** [`reports/week1_report.docx`](file:///c:/yurathna/reports/week1_report.docx)
- **Week 2 Executive Report:** [`reports/week2_report.docx`](file:///c:/yurathna/reports/week2_report.docx)
- **Week 2 Notebook:** [`notebooks/week2_eda.ipynb`](file:///c:/yurathna/notebooks/week2_eda.ipynb)
- **Visualization Figures Directory:** [`outputs/figures/week2/`](file:///c:/yurathna/outputs/figures/week2/)
