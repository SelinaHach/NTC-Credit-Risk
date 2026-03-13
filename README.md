# NTC-Credit-Risk

# NTC Consumer Credit — ML Pipeline

Predicting charge-off risk for **New-To-Credit (NTC)** consumers — young adults aged 18–25 with little or no credit history. The goal is to help lenders make smarter approval decisions in an underserved segment without over-rejecting potentially good customers.

---

## The Problem

NTC consumers are hard to underwrite. Many have no FICO score yet, thin credit files, and limited financial history. Traditional credit models fail here because they were built on established borrowers.

This project builds a machine learning classifier that predicts whether a consumer will **ever charge off** (default) on a credit obligation — using behavioral signals that don't rely on a full credit history.

**Target variable:** `Ever_ChargeOff` (1 = defaulted, 0 = did not default)

---

## Dataset

`v_credit_data_NTC_v51825.csv` — 2,000 consumer records, 27 features.

### Who is in this dataset?

| Metric | Value |
|--------|-------|
| Total consumers | 2,000 |
| Age range | 18 – 25 years old |
| Average age | 21.5 years |
| Income range | $18,013 – $54,961 |
| Average income | $36,212 |
| No-Hit (no credit file) | **41.5%** |
| Ever charged off | **14.1%** |
| Ever had a collection | 20.4% |
| Ever went bankrupt | 1.5% |
| Currently a student | 41.8% |

### Key Insight: Nearly half have no FICO score
`FICO_V` is only populated for 1,170 out of 2,000 consumers (58.5%). The other 41.5% are No-Hit — they have zero credit history. For those who do have a score, the average is **590** (range: 500–680), firmly in the subprime to near-prime band.

---

## What Drives Charge-Off? (Correlation Analysis)

| Feature | Correlation with Ever_ChargeOff |
|---------|--------------------------------|
| `Delinq_90D_12M` | **+0.571** — strongest signal |
| `Ever_Collection` | **+0.456** |
| `Delinq_60D_Curr` | **+0.449** |
| `Age` | +0.080 |
| `Revolving_Util` | +0.023 |
| `Income` | ~0.000 |

The top three predictors are all **delinquency and collection signals** — how late someone has been on past payments. Income, surprisingly, is nearly uncorrelated with charge-off in this NTC segment.

---

## What We Did — Step by Step

### 1. Exploratory Data Analysis
- Explored all 27 features with summary statistics, value counts, and distribution plots
- Generated correlation heatmaps against both `Ever_ChargeOff` and `FICO_V`
- Identified class imbalance: **85.9% non-chargeoff vs 14.1% chargeoff**

### 2. Outlier Removal (IQR Method)
Removed rows where values fell below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR`:

| Column | Rows Removed |
|--------|-------------|
| Unsecured_Debt | 192 |
| Unpaid_Collection_Trades | 66 |
| Student_Loan_Amt | 31 |
| Months_Since_Bankruptcy | 29 |
| Revolving_Util | 4 |
| ChargeOff_Balance | 3 |

**2,000 → 1,675 rows** after cleaning (325 rows removed, 16.3%)

### 3. Missing Data Handling
- `FICO_V`: 830 missing values (41.5%) — imputed with **999** as a special flag meaning "No-Hit / no credit file"
- `ChargeOff_Balance`: 1,969 missing (98.5%) — not used as a model feature since it only exists after a charge-off happens

### 4. Train / Validation / Test Split (70 / 15 / 15)
| Split | Rows |
|-------|------|
| Train | 1,172 |
| Validation | 251 |
| Test | 252 |

Charge-off rate in training data: **14.9%**

### 5. Feature Engineering
Created new columns to improve model interpretability and signal quality:

- **`DTI`** — Debt-to-Income ratio (`Unsecured_Debt / Income`)
- **`FICO_Band`** — Standard credit tiers: Poor / Fair / Good / Very Good / Exceptional
- **`Utilization_Band`** — Very Low / Low / Moderate / High / Extreme
- **`DTI_Band`** — Target / Reasonable / Less Favorable / High / Very High
- **`LoanAmount_Band`** — Student loan amount bucketed into 5 tiers
- **`Asset_Band`** — Checking account balance bucketed into 6 tiers

### 6. Charge-Off Rate by Segment
Analyzed charge-off rates across FICO and Utilization bands to confirm the model's intuition aligns with credit theory — lower FICO and higher utilization both correspond to elevated charge-off rates in this NTC portfolio.

---

## Models

### Why Two Models?
We trained both **Random Forest** and **XGBoost** to compare performance. Both use different approaches:
- **Random Forest** — builds 100 trees in parallel, averages their votes
- **XGBoost** — builds 100 trees sequentially, each one correcting the last

Both were tuned for class imbalance:
- Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight = 5.71` (ratio of non-chargeoff to chargeoff)

### Why Threshold = 0.7?
Default classification threshold is 0.5. We raised it to **0.7** intentionally. This means the model only flags someone as high-risk if it's at least 70% confident. The trade-off:
- **Higher Precision** — fewer false alarms, fewer good customers wrongly rejected
- **Lower Recall** — some actual charge-offs slip through

For a **startup entering the NTC segment**, this is the right trade-off. The priority is **market expansion and customer acquisition** — over-rejecting is more damaging than under-flagging at this stage.

---

## Model Results

### Random Forest

**Validation Set:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| No Charge-off (0) | 0.927 | 0.977 | 0.952 |
| Charge-off (1) | **0.722** | 0.433 | 0.542 |
| **Validation AUC** | | | **0.955** |

**Test Set:**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| No Charge-off (0) | 0.924 | 0.945 | 0.935 |
| Charge-off (1) | **0.571** | 0.485 | 0.525 |
| **Test AUC** | | | **0.931** |

**Test Confusion Matrix:**
```
               Predicted: No    Predicted: Yes
Actual: No         207              12
Actual: Yes         17              16
```
- 207 good customers correctly approved
- 16 actual charge-offs correctly caught
- 12 good customers wrongly flagged (false positives)
- 17 actual charge-offs missed (false negatives)

---

## Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `Delinq_90D_12M` | **37.8%** |
| 2 | `Ever_Collection` | 18.4% |
| 3 | `Delinq_60D_Curr` | 17.1% |
| 4 | `Revolving_Util` | 3.6% |
| 5 | `Unsecured_Debt` | 3.3% |
| 6 | `Checking_Asset` | 3.3% |
| 7 | `DTI` | 2.9% |
| 8 | `Income` | 2.9% |
| 9 | `FICO_Imputed` | 2.2% |
| 10 | `Age` | 2.2% |

**Key finding:** The top 3 features alone account for 73.3% of predictive power — all delinquency signals. Past payment behavior is overwhelmingly the best predictor of future default, even in a thin-file NTC population.

---

## Output

Final file: `NTC_Consumer_Credit_Output.csv`

Two new columns added:

| Column | Description |
|--------|-------------|
| `Predicted_ChargeOff_Prob` | Probability score (0.0 – 1.0) from XGBoost |
| `Predicted_ChargeOff_Label` | 1 = flagged high-risk (prob ≥ 0.7), 0 = approved |

---


**Dependencies:** `pandas` `numpy` `matplotlib` `seaborn` `scikit-learn` `xgboost`
