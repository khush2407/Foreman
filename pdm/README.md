# Predictive Maintenance Using Machine Learning
### ISE 298 Graduate Project — Khush Hemant Patel

An end-to-end Python pipeline that predicts machine failure from sensor
readings, explains each prediction, and converts model performance into a
maintenance policy and a dollar figure.

---

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh
```

Every figure lands in `outputs/figures/`, every table in `outputs/results/`.

Run time is about 2–3 minutes end to end.

---

## Pipeline

| Script | Stage | Produces | Report section |
|---|---|---|---|
| `01_data_understanding.py` | Load, audit, leakage check | 4 CSV tables | Data Collection |
| `02_eda.py` | Exploratory analysis | Figures 1–8 | Data Collection |
| `03_preprocessing.py` | Clean, encode, split | Train/test splits | Methodology |
| `04_feature_engineering.py` | 4 new features, scale, SMOTE | `model_data.npz` | Methodology |
| `05_model_training.py` | 4 models, CV, tuning | Trained models | Analysis Methods |
| `06_evaluation.py` | Test-set scoring | Figures 9–12 | Results |
| `07_shap_analysis.py` | Explainability | Figures 13–15 | Results |
| `08_business_impact.py` | Cost model, policy | Figures 16–17 | Discussion |
| `09_reliability.py` | Tool life (Kaplan-Meier, Weibull), best tool-change age, reliability block diagram, failure dependencies | Figures 18–22 | Results, Discussion |
| `10_plant_analysis.py` | Root cause of each alarm (fishbone, Pareto), demand and usage scenarios with a drift check, critical spare parts (ABC), production line simulation with bottleneck and backup plans | Figures 23–29 | Results, Discussion |

Scripts are numbered because each depends on the previous one's saved output.
Run them in order the first time; after that any script can be re-run alone.

---

## Headline results

| | |
|---|---|
| Best model | Gradient Boosting (tuned) |
| Recall | 86.8% at threshold 0.50 → **91.2%** at the cost-optimal 0.38 |
| Precision | 55.7% → 47.7% |
| PR-AUC | **0.886** |
| ROC-AUC | 0.980 |
| Engineered features' share of model importance | **61.5%** |
| Annual saving vs run-to-failure | **$3.1M** (76%) |

---

## Three findings worth defending in the viva

**1. Five columns had to be dropped for target leakage.**
`Machine failure` is the logical OR of `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
Including them yields ~100% accuracy and a model that has learned nothing about
the sensors. Stage 1 proves the relationship on 9,973 of 10,000 rows.

**2. SMOTE is applied inside each cross-validation fold, not before it.**
Resampling before splitting lets synthetic points derived from a validation
failure leak into the training folds, inflating CV recall by roughly 10–15
points. Stage 5 splits first, then balances only the training portion of each
fold.

**3. Accuracy is reported but never used to decide anything.**
Predicting "no failure" every time scores 96.6% accuracy and catches zero
failures — it beats Logistic Regression on that metric. Model selection uses
PR-AUC; the operating threshold is chosen by cost.

---

## Dataset

AI4I 2020 Predictive Maintenance Dataset — 10,000 cycles, 14 columns, no
missing values, 3.39% failure rate.

Feature set used for modelling (10 columns after engineering):

**Original** — `Type` (ordinal L/M/H), air temperature, process temperature,
rotational speed, torque, tool wear.

**Engineered**, each motivated by a specific EDA figure:

| Feature | Formula | Motivated by |
|---|---|---|
| `Power [W]` | torque × angular velocity | Fig 6 — failures sit outside a safe power band |
| `Temp_diff [K]` | process temp − air temp | Fig 8 — heat failures follow the temperature gap |
| `Wear_x_Torque` | tool wear × torque | Fig 7 — overstrain follows a product threshold |
| `Temp_power_ratio` | temp gap ÷ (power/1000) | thermal headroom vs mechanical work |

---

## Cost assumptions

Stage 8's conclusions follow from four numbers, stated openly so a reader can
substitute their own. `08_cost_sensitivity.csv` shows the conclusion holds
across a 10x–100x range of the critical ratio.

| Outcome | Cost | Meaning |
|---|---|---|
| Missed failure (FN) | $12,000 | emergency repair, unplanned line stop |
| Caught failure (TP) | $1,500 | planned intervention in a scheduled window |
| False alarm (FP) | $400 | inspection, no fault found |
| Correct no-call (TN) | $0 | — |

---

## Dependency fallbacks

Three libraries have automatic fallbacks so the pipeline runs on a machine
where they cannot be installed. The scripts print which path they took.

| Preferred | Fallback | Effect on results |
|---|---|---|
| `xgboost` | `sklearn.HistGradientBoostingClassifier` | Comparable; scores shift slightly |
| `imbalanced-learn` SMOTE | built-in implementation in `04`/`05` | Identical algorithm |
| `shap` | permutation importance + partial dependence + local ablation | Same three questions answered, coarser attribution |

With all three installed you get the version described in the proposal. The
committed results were produced using the fallback path.

---

## Reproducibility

`random_state=42` throughout — the split, SMOTE, and every model. Re-running
`run_all.sh` reproduces every number in this README exactly.
