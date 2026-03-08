# 🏠 House Price Regression Analysis

> Predicting residential property prices using the Ames Housing dataset with a full end-to-end sklearn regression pipeline.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🎯 Problem Statement

Predict the final sale price of residential homes in Ames, Iowa based on 79 explanatory features. This is a classic regression benchmark used to practice feature preprocessing, model selection, and evaluation.

**Dataset:** [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — 1,460 training samples, 79 features.

---

## 🧪 Approach

### 1. Data & Preprocessing
- Handled missing values with median/mode imputation
- Log-transformed skewed numerical features (including target `SalePrice`)
- One-hot encoded nominal categoricals; ordinal encoded ordered categoricals
- Applied `StandardScaler` to numerical features

### 2. Models Compared
| Model | CV RMSE (log) | Notes |
|---|---|---|
| Ridge Regression | 0.141 | Good baseline, stable |
| Lasso Regression | 0.138 | Auto feature selection |
| ElasticNet | 0.137 | Best linear model |
| **XGBoost** | **0.124** | Best overall |
| Random Forest | 0.131 | Robust, slower |

### 3. Hyperparameter Tuning
Used `GridSearchCV` with 5-fold cross-validation on XGBoost. Tuned `n_estimators`, `max_depth`, `learning_rate`, and `subsample`.

---

## 📊 Results

| Metric | Value |
|---|---|
| Best Model | XGBoost |
| CV RMSE (log scale) | 0.124 |
| Test RMSE (log scale) | 0.128 |
| R² Score | 0.912 |

**Top 5 Features by Importance:**
1. `OverallQual` — Overall material and finish quality
2. `GrLivArea` — Above grade living area (sq ft)
3. `TotalBsmtSF` — Total basement area
4. `GarageCars` — Garage capacity
5. `YearBuilt` — Original construction year

---

## 💡 Key Findings

- Log-transforming `SalePrice` improved all model scores significantly (reduces right skew)
- `OverallQual` alone explains ~65% of price variance
- Lasso automatically zeroed out ~30 low-signal features
- XGBoost outperformed linear models by ~10% RMSE, but ElasticNet is far more interpretable

---

## 🗂️ Repo Structure

```
.
├── data/                         # Data loading instructions (no raw data committed)
│   └── download_data.md
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Feature engineering & pipeline
│   └── 03_modeling.ipynb         # Model training & evaluation
├── src/
│   ├── preprocess.py             # Preprocessing pipeline
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation & plots
├── configs/
│   └── config.yaml               # Hyperparameters
├── results/
│   └── feature_importance.png    # Feature importance chart
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/yourusername/01-regression-analysis
cd 01-regression-analysis
pip install -r requirements.txt

# Download data from Kaggle and place in data/
# Then run:
python src/train.py --config configs/config.yaml
python src/evaluate.py
```

---

## 🔭 Future Work

- [ ] Stack ensemble (XGBoost + Ridge blend)
- [ ] Add SHAP waterfall plots per sample
- [ ] Serve predictions via FastAPI endpoint

---

## 📚 References

- [Ames Housing Dataset — De Cock (2011)](http://jse.amstat.org/v19n3/decock.pdf)
- [XGBoost Paper — Chen & Guestrin (2016)](https://arxiv.org/abs/1603.02754)
