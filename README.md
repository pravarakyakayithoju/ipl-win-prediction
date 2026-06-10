 # 🏏 IPL In-Game Win Probability Prediction

A machine learning model that predicts whether the chasing team will win an IPL match, based on the **live state of the run chase** — the same idea behind the win-probability bars on cricket broadcasts.

Built on 260,000+ ball-by-ball records from 16 IPL seasons (2008–2024).

---

## 🎯 Results

| Model | Accuracy (5-fold CV) | ROC-AUC |
|-------|---------------------|---------|
| **Logistic Regression** | **79.1% ± 2.1%** | **0.875** |
| Random Forest | 78.3% ± 1.4% | 0.866 |

Naive baseline (always predict majority class): **53.5%**

> **An honest contrast that matters:** a *pre-match* model (teams, toss, recent form only) on this same data scores ~51% — essentially a coin flip, because T20 cricket is high-variance. Switching to *in-game* state pushes accuracy to ~79%. Identifying which features actually carry predictive signal is a core part of the project.

---

## 🔍 How it works

1. **Feature engineering from raw deliveries** — reconstruct the game state (runs, wickets, balls remaining) at the end of overs 5, 10, and 15 of every chase.
2. **Situational features** — runs needed, wickets in hand, balls left, required run rate, current run rate, and "pressure" (the gap between the two rates).
3. **Leakage-safe labeling** — each snapshot uses only information available *at that moment*; the label is the eventual match result.
4. **Multi-snapshot sampling** — training on three points per innings teaches the model a genuine win-probability curve rather than a single-moment trick.

Top predictive features: **required run rate**, **pressure**, **runs needed** — exactly what cricket intuition would expect.

---

## 📈 Win Probability Curve

The model produces a broadcast-style win-probability graph for any match, updating after every over of the chase. (See the notebook's final cell.)

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — Logistic Regression, Random Forest, cross-validation, calibration
- **matplotlib / seaborn** — EDA and result visualization

---

## ⚙️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Place `matches_cleaned.csv` and `deliveries_cleaned.csv` in the same folder, then open and run `ipl_winprob.ipynb` top to bottom.

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
