# Data Science Project — ITSM (IT Service Management)

**Client:** ABC Tech &nbsp;|&nbsp; **Category:** ITSM - Machine Learning

---

## Overview

This project applies machine learning to IT Service Management (ITSM) data from ABC Tech to solve four business problems that were contributing to declining customer satisfaction, SLA breaches, and reactive incident management.

The dataset contains approximately 46,606 incident records spanning 2012 to 2014, sourced directly from a MySQL database. Each record includes ticket metadata, configuration item details, timestamps, priority assignments, and resolution information.

---

## Business Problem

ABC Tech's IT service desk was experiencing the following challenges:

- High priority tickets (P1 and P2) were not being identified and escalated fast enough
- Incident volume was unpredictable, making staffing and resource planning difficult
- Agents were incorrectly assigning priorities, causing unnecessary reassignments
- Incidents requiring infrastructure changes (RFCs) were being missed, leading to recurring failures and unplanned outages

These issues were directly impacting customer satisfaction scores and SLA compliance rates.

---

## ML Objectives

| # | Problem | ML Type | Business Goal |
|---|---|---|---|
| 1 | Predict High Priority Tickets (P1 and P2) | Binary Classification | Flag critical tickets at creation time for immediate escalation |
| 2 | Forecast Monthly Incident Volume | Time Series (SARIMA) | Enable proactive staffing and infrastructure planning |
| 3 | Auto-tag Ticket Priority | Multiclass Classification | Eliminate manual priority mis-categorization by agents |
| 4 | Predict RFC and Possible Failure | Binary Classification | Enable proactive change management before failures occur |

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | MySQL Database — project_itsm |
| Total Records | 46,606 tickets |
| Period | 2012 to 2014 |
| Columns | 25 (ticket metadata, CI details, timestamps, resolution info) |
| Key Target Columns | Priority (1-5), High_Priority (binary), Has_RFC (binary) |

---

## Project Structure

```
ITSM-ML-Project/
│
├── ITSM_ML_Project_Final_v2.ipynb   # Main project notebook (all 4 problems)
├── README.md                         # Project documentation
```

---

## Notebook Structure

The notebook is organised into clearly separated sections:

| Section | Description |
|---|---|
| Step 1 — Install Libraries | All required packages |
| Step 2 — Import Libraries | Imports with cross-validation and metrics |
| Step 3 — Load Data | MySQL connection and data loading |
| Step 4 — Data Overview | Shape, dtypes, missing values, basic statistics |
| Step 5 — Raw Data Quality Check | Inspection of unique values before cleaning |
| Step 6 — Data Cleaning | Fixing NS/NA values, string errors, numeric conversion |
| Step 7 — Feature Engineering | Time features, label encoding, target creation |
| Step 8 — EDA | Priority distribution, Impact-Urgency heatmap, reassignment analysis |
| Problem 1 | High Priority Prediction — Binary Classification |
| Problem 2 | Incident Volume Forecasting — SARIMA |
| Problem 3 | Priority Auto-tagging — Multiclass Classification |
| Problem 4 | RFC Prediction — Binary Classification |
| Final Summary | Results, challenges, solutions, and business recommendations |

---

## Results Summary

### Problem 1 — High Priority Ticket Prediction

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting (Tuned)** | 97.93% | 0.4615 | **0.9265** |
| XGBoost | 98.30% | 0.4522 | 0.8945 |
| Random Forest | 98.36% | 0.4299 | 0.8994 |
| Decision Tree | 97.39% | 0.3822 | 0.7562 |
| Logistic Regression | 82.08% | 0.1119 | 0.8821 |

Best model: **Gradient Boosting** with ROC-AUC = 0.926. F1 is low due to severe class imbalance (only 1.5% of tickets are P1/P2), which is expected and acceptable. ROC-AUC is the primary metric here.

---

### Problem 2 — Incident Volume Forecasting

| Attribute | Detail |
|---|---|
| Model | SARIMA(1,1,1)(1,1,1,12) |
| Training Data | January 2013 to December 2014 (24 months) |
| Forecast Horizon | January 2015 to December 2015 |
| Evaluation | MAE, RMSE, MAPE computed on in-sample fitted values |

---

### Problem 3 — Priority Auto-tagging

| Model | Accuracy | Weighted F1 |
|---|---|---|
| **Gradient Boosting** | **99.73%** | **0.997** |
| Random Forest | 99.70% | 0.997 |
| XGBoost | 99.65% | 0.997 |

High accuracy is expected and valid — Impact and Urgency fields are included because they are available at ticket creation and directly follow the ITIL priority matrix. The model auto-corrects human errors in priority assignment.

---

### Problem 4 — RFC Prediction

| Model | Accuracy | F1 (Has RFC) | ROC-AUC |
|---|---|---|---|
| **Random Forest** | **97.53%** | 0.31 | **0.869** |
| XGBoost | 97.07% | 0.32 | 0.839 |
| Gradient Boosting | 91.86% | 0.17 | 0.846 |
| Logistic Regression | 55.47% | 0.05 | 0.740 |

Decision threshold lowered from 0.5 to 0.1 to maximise RFC recall. Missing a real RFC (False Negative) is far more costly than a false alarm (False Positive).

---

## Key Technical Decisions

### Data Leakage Prevention

| Problem | Leakage Risk | Action Taken |
|---|---|---|
| P1 — High Priority Prediction | Impact and Urgency directly determine Priority | Removed from P1 feature set |
| P3 — Priority Auto-tagging | Same fields included intentionally | Valid — agents fill these at creation; model corrects human errors |
| P4 — RFC Prediction | No_of_Related_Changes is used to create the target | Removed from P4 feature set |

### Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) was applied on the training set only for Problems 1 and 4, where the minority class represents less than 2% of the data. The test set was kept unmodified to reflect real-world distribution.

### Threshold Optimisation (Problem 4)

The RFC prediction model uses a decision threshold of 0.1 instead of the default 0.5. This is a deliberate business decision — since missing a real RFC can cause unplanned outages and SLA breaches, maximising recall on the Has RFC class is the priority even at the cost of some precision.

### Hyperparameter Tuning

GridSearchCV with 3-fold cross-validation was applied to the best-performing model in Problem 1 (Gradient Boosting) to optimise n_estimators, max_depth, and learning_rate.

### Cross-Validation

5-fold cross-validation was performed for Problems 1, 3, and 4 to confirm that results are stable and not a consequence of a particular train-test split.

---

## Data Quality Issues and Fixes

| Issue | Fix Applied |
|---|---|
| NS and NA strings in Impact, Priority, Urgency | pd.to_numeric with errors=coerce, then filled with column mode |
| Mixed string in Urgency (5 - Very Low) | Replaced with numeric 5 before conversion |
| Corrupt 1970 epoch timestamps in Open_Time | Filtered — only 2013 and 2014 data used |
| 2012 data has only 10 records | Excluded from SARIMA training |
| Negative values in SARIMA confidence interval | Clipped to 0 — incident count cannot be negative |
| LabelEncoder reused across columns | Fixed — separate encoder instance saved per column in a dictionary |
| XGBoost requires 0-indexed class labels | Priority 1-5 remapped to 0-4 for training, remapped back to 1-5 for display |

---

## Libraries and Tools

| Category | Libraries |
|---|---|
| Data Handling | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn, xgboost |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Time Series | statsmodels (SARIMAX) |
| Database | mysql-connector-python |

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Install dependencies
```bash
pip install mysql-connector-python pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels imbalanced-learn
```

3. Open the notebook
```bash
jupyter notebook ITSM_ML_Project_Final_v2.ipynb
```

4. Run all cells in order from top to bottom

> **Note:** The MySQL database credentials in the notebook are for academic and demonstration use only.

---

## Business Recommendations

**1. Deploy High Priority Predictor**
Integrate the Gradient Boosting model into the ticketing system to flag P1 and P2 tickets at creation and route them to senior engineers without manual triage.

**2. Use Volume Forecasts for Planning**
Share SARIMA monthly forecasts with HR and infrastructure teams at the start of each quarter to enable data-driven staffing and resource allocation decisions.

**3. Implement Auto-Priority Tagging**
Replace manual priority assignment with the multiclass model to eliminate mis-categorization errors at the point of ticket creation, reducing downstream reassignments.

**4. Enable RFC Early Warning System**
Trigger an automated alert to the change management team whenever the RFC probability exceeds 10%, converting reactive change management into a proactive process.

---

## Project Information

| Field | Detail |
|---|---|
| Client | ABC Tech |
| Dataset Period | 2012 to 2014 |
| Total Records | 46,606 |
