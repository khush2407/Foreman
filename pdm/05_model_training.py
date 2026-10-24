"""
STAGE 5 — MODEL TRAINING
=========================
Four classifiers, trained and cross-validated on the same data, so the
comparison in Stage 6 is fair.

  1. Logistic Regression  — linear baseline. If a linear boundary were enough,
                            nothing more complex would be justified.
  2. Random Forest        — bagged trees. Handles the non-linear failure
                            envelope seen in Figure 6 without any tuning.
  3. Gradient Boosting    — sequentially corrects its own errors. Usually the
                            strongest family on tabular data like this.
  4. SVM (RBF kernel)     — a geometric approach, included to test whether a
                            smooth curved boundary beats axis-aligned splits.

*** THE CROSS-VALIDATION SUBTLETY THAT MOST PROJECTS GET WRONG ***

SMOTE must be applied INSIDE each CV fold, not before the CV loop.

If you SMOTE first and then cross-validate, synthetic points generated from a
failure in fold 3 can land in folds 1, 2, 4 and 5. The model then sees
near-copies of validation points during training, and the CV score comes out
inflated — often by 10-15 points of recall. It looks like a great model right
up until it meets real data.

So the loop below splits first, then SMOTEs only the training part of each
fold. The validation part of every fold keeps the real 3.4% failure rate.

Scoring metric: PR-AUC (average precision), not accuracy and not ROC-AUC.
With 3.4% positives, ROC-AUC is flattered by the huge number of true negatives.
PR-AUC only looks at how well the model ranks the rare positives, which is the
actual job.
"""
import numpy as np
import pandas as pd
import time
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, recall_score, precision_score

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
RESULTS = ROOT / "outputs" / "results"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5

print("=" * 78)
print("STAGE 5 — MODEL TRAINING")
print("=" * 78)

d = np.load(PROC / "model_data.npz")
X_train, y_train = d["X_train_scaled"], d["y_train"]
X_test,  y_test  = d["X_test_scaled"],  d["y_test"]
X_bal,   y_bal   = d["X_train_balanced"], d["y_train_balanced"]
feature_names = pd.read_csv(PROC / "feature_names.csv")["feature"].tolist()

print(f"\nTraining data : {X_train.shape[0]:,} rows x {X_train.shape[1]} features "
      f"({int(y_train.sum())} failures)")
print(f"Test data     : {X_test.shape[0]:,} rows ({int(y_test.sum())} failures)")


# ── SMOTE (same implementation as Stage 4, re-used per fold) ──────────────
def smote(X, y, k=5, random_state=42):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    X_min = X[y == 1]
    n_needed = int((y == 0).sum() - (y == 1).sum())
    if n_needed <= 0 or len(X_min) <= k:
        return X, y
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    idx = idx[:, 1:]
    base_i  = rng.randint(0, len(X_min), n_needed)
    neigh_i = idx[base_i, rng.randint(0, k, n_needed)]
    synth = X_min[base_i] + rng.rand(n_needed, 1) * (X_min[neigh_i] - X_min[base_i])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_needed, dtype=y.dtype)])


try:
    from imblearn.over_sampling import SMOTE as ImbSMOTE
    def balance(X, y, seed=RANDOM_STATE):
        return ImbSMOTE(random_state=seed, k_neighbors=5).fit_resample(X, y)
    SMOTE_IMPL = "imbalanced-learn"
except ImportError:
    def balance(X, y, seed=RANDOM_STATE):
        return smote(X, y, k=5, random_state=seed)
    SMOTE_IMPL = "built-in fallback"


# ── Gradient boosting: XGBoost if available, sklearn otherwise ────────────
try:
    from xgboost import XGBClassifier
    def make_gb(**kw):
        params = dict(n_estimators=300, max_depth=5, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.9,
                      random_state=RANDOM_STATE, eval_metric="logloss",
                      n_jobs=-1)
        params.update(kw)
        return XGBClassifier(**params)
    GB_NAME, GB_IMPL = "XGBoost", "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def make_gb(**kw):
        params = dict(max_iter=300, max_depth=5, learning_rate=0.1,
                      random_state=RANDOM_STATE)
        # translate the xgboost-style names used in the tuning grid
        if "n_estimators" in kw: params["max_iter"] = kw.pop("n_estimators")
        params.update({k: v for k, v in kw.items()
                       if k in ("max_depth", "learning_rate", "max_iter")})
        return HistGradientBoostingClassifier(**params)
    GB_NAME, GB_IMPL = "Gradient Boosting", "sklearn HistGradientBoosting"

print(f"\nSMOTE implementation      : {SMOTE_IMPL}")
print(f"Gradient boosting backend : {GB_IMPL}")


# ── 5.1 Model definitions ─────────────────────────────────────────────────
def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1),
        GB_NAME: make_gb(),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            random_state=RANDOM_STATE),
    }


# ── 5.2 Cross-validation with SMOTE inside each fold ──────────────────────
print("\n" + "-" * 78)
print("[5.2] 5-fold stratified cross-validation (SMOTE applied within folds)")
print("-" * 78)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []

for name, _ in build_models().items():
    scores = {"pr_auc": [], "recall": [], "precision": []}
    t0 = time.time()

    for fold, (tr, va) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[tr], y_train[tr]
        X_va, y_va = X_train[va], y_train[va]     # <- never resampled

        X_tr_bal, y_tr_bal = balance(X_tr, y_tr, seed=RANDOM_STATE + fold)

        model = build_models()[name]
        model.fit(X_tr_bal, y_tr_bal)

        proba = model.predict_proba(X_va)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        scores["pr_auc"].append(average_precision_score(y_va, proba))
        scores["recall"].append(recall_score(y_va, pred, zero_division=0))
        scores["precision"].append(precision_score(y_va, pred, zero_division=0))

    elapsed = time.time() - t0
    row = {
        "model":         name,
        "cv_pr_auc":     np.mean(scores["pr_auc"]),
        "cv_pr_auc_std": np.std(scores["pr_auc"]),
        "cv_recall":     np.mean(scores["recall"]),
        "cv_precision":  np.mean(scores["precision"]),
        "fit_seconds":   elapsed,
    }
    cv_rows.append(row)
    print(f"  {name:<22} PR-AUC {row['cv_pr_auc']:.4f} "
          f"(+/- {row['cv_pr_auc_std']:.4f})   "
          f"recall {row['cv_recall']:.3f}   "
          f"precision {row['cv_precision']:.3f}   [{elapsed:5.1f}s]")

cv_df = pd.DataFrame(cv_rows).sort_values("cv_pr_auc", ascending=False)
cv_df.to_csv(RESULTS / "05_cv_scores.csv", index=False)

best_name = cv_df.iloc[0]["model"]
print(f"\n  Best model by cross-validated PR-AUC: {best_name}")


# ── 5.3 Hyperparameter tuning on the winner ───────────────────────────────
print("\n" + "-" * 78)
print(f"[5.3] Hyperparameter tuning — {best_name}")
print("-" * 78)

GRIDS = {
    "Random Forest": {
        "n_estimators":     [200, 400],
        "max_depth":        [8, 14, None],
        "min_samples_leaf": [1, 3],
    },
    GB_NAME: {
        "n_estimators":  [200, 400],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    },
    "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
    "SVM (RBF)":           {"C": [1.0, 10.0], "gamma": ["scale", 0.1]},
}

grid = GRIDS[best_name]
combos = list(ParameterGrid(grid))
print(f"  Searching {len(combos)} combinations x {N_FOLDS} folds "
      f"= {len(combos) * N_FOLDS} fits\n")

tuning_rows = []
combo_scores = []          # keep the ORIGINAL dicts so types survive intact
for params in combos:
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train)):
        X_tr_bal, y_tr_bal = balance(X_train[tr], y_train[tr],
                                     seed=RANDOM_STATE + fold)
        if best_name == "Random Forest":
            m = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
        elif best_name == GB_NAME:
            m = make_gb(**params)
        elif best_name == "Logistic Regression":
            m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)
        else:
            m = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **params)
        m.fit(X_tr_bal, y_tr_bal)
        fold_scores.append(
            average_precision_score(y_train[va], m.predict_proba(X_train[va])[:, 1]))

    mean_score = float(np.mean(fold_scores))
    tuning_rows.append({**params, "cv_pr_auc": mean_score})
    combo_scores.append((mean_score, params))
    print(f"    {str(params):<62} PR-AUC {mean_score:.4f}")

tuning_df = pd.DataFrame(tuning_rows).sort_values("cv_pr_auc", ascending=False)
tuning_df.to_csv(RESULTS / "05_tuning_results.csv", index=False)

# Take the winning dict straight from the grid. Reading it back out of the
# DataFrame would silently turn ints into floats (a mixed-dtype column is
# stored as float64), and sklearn rejects max_depth=5.0.
best_score, best_params = max(combo_scores, key=lambda t: t[0])
untuned_score = float(cv_df.iloc[0]["cv_pr_auc"])

print(f"\n  Best parameters : {best_params}")
print(f"  Tuned CV PR-AUC : {best_score:.4f}")
print(f"  Untuned CV PR-AUC: {untuned_score:.4f}")
print(f"  Improvement     : {best_score - untuned_score:+.4f}")


# ── 5.4 Fit final models on the full balanced training set ────────────────
print("\n" + "-" * 78)
print("[5.4] Fitting final models on the full SMOTE-balanced training set")
print("-" * 78)

final_models = build_models()

# swap in the tuned version of the winner
if best_name == "Random Forest":
    final_models[best_name] = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, **best_params)
elif best_name == GB_NAME:
    final_models[best_name] = make_gb(**best_params)
elif best_name == "Logistic Regression":
    final_models[best_name] = LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE, **best_params)
else:
    final_models[best_name] = SVC(
        kernel="rbf", probability=True, random_state=RANDOM_STATE, **best_params)

predictions = {}
for name, model in final_models.items():
    t0 = time.time()
    model.fit(X_bal, y_bal)
    proba = model.predict_proba(X_test)[:, 1]
    predictions[name] = proba
    joblib.dump(model, MODELS / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
    print(f"  {name:<22} fitted in {time.time()-t0:5.1f}s  ->  saved")

np.savez_compressed(RESULTS / "05_test_predictions.npz",
                    y_test=y_test, **{k: v for k, v in predictions.items()})

with open(RESULTS / "05_best_model.txt", "w") as f:
    f.write(f"{best_name}\n{best_params}\n")

print(f"\n  Test-set probabilities saved for all {len(final_models)} models.")
print("\n" + "=" * 78)
print("STAGE 5 COMPLETE")
print("=" * 78)
