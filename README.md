# Loan Default Prediction — 6-Axis Ablation Study

## Problem
Predicting whether a borrower will experience serious financial distress (90+ days delinquent) within two years. The core challenge is a severe 14.2:1 class imbalance — naive models achieve 93% accuracy by predicting no default for everyone, while missing 84% of actual defaulters.

## Dataset
[Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) — 150,000 borrower records with 11 features including delinquency history, debt ratios, income, and credit utilization. Kaggle public score: **0.859 AUC**.

## Approach
Rather than optimizing a single model, this project systematically isolates the effect of 6 independent interventions through a controlled ablation study. Each axis changes one variable while holding everything else constant.

## Ablation Axes

| Axis | What we tested |
|---|---|
| A1 — Model complexity | Logistic Regression → Random Forest → XGBoost → LightGBM |
| A2 — Feature engineering | Raw features vs engineered (delinquency score, age-utilization interaction) |
| A3 — Imbalance handling | No handling vs class weights vs SMOTE |
| A4 — Calibration | Uncalibrated vs Platt scaling vs Isotonic regression |
| A5 — Ensembles | Single model vs Voting vs Stacking |
| A6 — Threshold optimization | Default 0.5 vs F1-optimal vs business-constrained |

## Key Results

| Experiment | ROC-AUC | F1 | Recall | Brier Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.843 | 0.255 | 0.160 | 0.051 |
| LightGBM (A1) | 0.870 | 0.279 | 0.182 | 0.049 |
| + Feature engineering (A2) | 0.870 | 0.273 | 0.177 | 0.049 |
| + Class weights (A3) | 0.869 | 0.352 | 0.773 | 0.135 |
| + Isotonic calibration (A4) | 0.858 | 0.246 | 0.154 | 0.049 |
| + Business threshold 0.09 (A6) | 0.858 | 0.367 | 0.689 | 0.049 |

---
## Kaggle Submission
<img width="2171" height="1058" alt="image" src="https://github.com/user-attachments/assets/bb1f9a10-48cf-421c-ab11-97d247d11f36" />


## Key Findings
- **Boosting > Bagging**: LightGBM outperformed Random Forest across all metrics at default settings
- **Imbalance handling is the biggest lever**: Class weights alone tripled Recall from 16% to 77%
- **SMOTE underperforms on this dataset**: PCA visualization showed heavy class overlap — SMOTE created noisy synthetic examples, confirmed by ROC-AUC dropping from 0.860 to 0.829
- **Engineered features dominate**: Weighted delinquency score became the top feature by gain-based importance (4x over any raw feature)
- **Threshold optimization is high-leverage**: Moving from 0.5 to 0.09 threshold boosted Recall from 15% to 69% with zero model retraining
- **Calibration trades Recall for probability quality**: Isotonic regression improved Brier Score from 0.135 to 0.049 but requires threshold retuning


## How to Run
1. Download `cs-training.csv` from [Kaggle](https://www.kaggle.com/competitions/GiveMeSomeCredit)
2. Place it in the project root
3. Run notebooks in order: `01_eda` → `02_baseline` → `03_ablations`

## Requirements
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn


