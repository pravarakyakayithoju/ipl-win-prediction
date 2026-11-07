# 🏏 IPL In-Game Win Probability Prediction

A machine learning model that predicts whether the chasing team will win an IPL match, based on the **live state of the run chase** — the same idea behind the win-probability bars on cricket broadcasts.

Built on 260,000+ ball-by-ball records from 16 IPL seasons (2008–2024).

---

## 🎯 Results

| Model | Accuracy (5-fold CV) | ROC-AUC |
|-------|---------------------|---------|
| **Logistic Regression** | **79.1% ± 2.1%** | **0.875** |
| Random Forest (untuned) | 78.3% ± 1.4% | 0.866 |
| Random Forest (GridSearchCV-tuned) | — | 0.872 (CV) |

Naive baseline (always predict majority class): **53.5%**

> **An honest contrast that matters:** a *pre-match* model (teams, toss, recent form only) on this same data scores ~51% — essentially a coin flip, because T20 cricket is high-variance. Switching to *in-game* state pushes accuracy to ~79%. Identifying which features actually carry predictive signal is a core part of the project.

---

## 🔍 How it works

1. **Feature engineering from raw deliveries** — reconstruct the game state (runs, wickets, balls remaining) at the end of overs 5, 10, and 15 of every chase.
2. **Situational features** — runs needed, wickets in hand, balls left, required run rate, current run rate, and "pressure" (the gap between the two rates).
3. **Leakage-safe labeling** — each snapshot uses only information available *at that moment*; the label is the eventual match result.
4. **Multi-snapshot sampling** — training on three points per innings teaches the model a genuine win-probability curve rather than a single-moment trick.

Top predictive features: **required run rate**, **pressure**, **runs needed** — exactly what cricket intuition would expect, confirmed independently by both Gini feature importance and SHAP (below).

---

## ⚙️ Hyperparameter Tuning — GridSearchCV

The Random Forest's original hyperparameters (`n_estimators=300, max_depth=8`) were hand-picked. Ran `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, and `max_features` (72 combinations × 5-fold CV, scored on ROC-AUC):

- **Best params:** `max_depth=6, max_features='sqrt', min_samples_split=10, n_estimators=200`
- **Best CV ROC-AUC:** 0.872 (up from 0.866 untuned)

| | Test Accuracy | Test ROC-AUC |
|---|---|---|
| Untuned RF | 76.47% | 0.8483 |
| **Tuned RF** | **76.63%** | **0.8520** |

The improvement is real but modest (+0.4 points accuracy, +0.004 ROC-AUC on the held-out test set) — systematic tuning mostly *confirmed* the original hand-picked settings were already close to a local optimum for this feature set and dataset size, rather than unlocking a large gain. Logistic Regression remains the strongest model overall at 0.875 CV ROC-AUC, consistent with the earlier finding that the win-probability relationship is largely smooth and monotonic once the right situational features are engineered — exactly the regime where a well-regularized linear model competes with, or slightly beats, a tree ensemble.

---

## 🔬 Model Interpretability — SHAP

Feature importance says *which* features the Random Forest leans on overall; it doesn't say *how* a given feature value pushes an individual prediction, or whether that effect is linear or threshold-like. Added `shap.TreeExplainer` (Lundberg & Lee, *"A Unified Approach to Interpreting Model Predictions,"* NeurIPS 2017, [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)) on the tuned Random Forest to get exact, per-prediction Shapley attributions.

**Result:** SHAP's summary and mean-|SHAP value| bar plots corroborate the Gini-based feature importance independently — `req_rr` (required run rate) and `pressure` dominate, followed by `runs_needed` and `target1`. Having two different attribution methods (impurity-decrease and Shapley values) agree is stronger interpretability evidence than feature importance alone, since Gini importance can be biased toward high-cardinality features, while SHAP values are computed directly from each feature's actual marginal contribution to individual predictions.

**What this validates:** the feature-engineering decisions made during the original pre-match → in-game redesign — `req_rr` and `pressure` weren't included on a hunch, they're confirmed as the dominant predictive signal by two independent interpretability methods.

---

## 📈 Win Probability Curve

The model produces a broadcast-style win-probability graph for any match, updating after every over of the chase. (See the notebook's final cell.)

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — Logistic Regression, Random Forest, GridSearchCV, cross-validation, calibration
- **SHAP** — TreeExplainer for model interpretability
- **matplotlib / seaborn** — EDA and result visualization

---

## ⚙️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap jupyter
```

Place `matches_cleaned.csv` and `deliveries_cleaned.csv` in the same folder, then open and run `ipl_winprob.ipynb` top to bottom. GridSearchCV (72 candidates × 5 folds) and SHAP both run on CPU in a few minutes — no GPU needed.

---

## 📁 Dataset

Kaggle — IPL Complete Dataset 2008–2024
- `matches_cleaned.csv` — 1,095 matches
- `deliveries_cleaned.csv` — 260,000+ ball-by-ball records

---

## 💡 Possible Extensions

- Per-ball snapshots for a smoother probability curve
- Add batting/bowling team identity as features
- XGBoost comparison
- Deploy as a Streamlit live win-probability app
