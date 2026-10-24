"""
TRAIN THE FAILURE-MODE CLASSIFIERS
===================================
The binary model already answers "will this cycle fail?". It cannot say
WHICH failure is coming, and a technician needs that to know where to look.

This trains one classifier per failure mode, each predicting its own label
from the sensor readings alone.

WHY THIS IS NOT THE LEAKAGE PROBLEM
  Using TWF/HDF/PWF/OSF as *inputs* to predict the aggregate 'Machine
  failure' target is leakage — their disjunction reconstructs the answer.
  Using each one as its own *target*, predicted from sensors, is a
  different and legitimate task. It is what Matzka (2020) does in the
  paper that introduced this dataset.

WHAT TO EXPECT
  Three of the four modes are generated from deterministic rules written
  in terms of quantities the engineered features encode directly:
      HDF  temp_diff < 8.6 K AND speed < 1380 rpm
      PWF  power outside 3500-9000 W
      OSF  tool wear x torque above a quality-dependent threshold
  So those three should be recovered almost exactly — which is itself the
  result worth reporting: the pipeline rediscovers the generating physics.

  TWF fires at a random wear time between 200 and 240 minutes, so it is
  only partly predictable. RNF is pure noise by construction and is
  excluded — no model can predict it, and saying so is more honest than
  reporting a meaningless score for it.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (average_precision_score, recall_score,
                             precision_score, roc_auc_score, confusion_matrix)

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42

MODES = {
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "TWF": "Tool wear failure",
}

print("=" * 76)
print("TRAINING FAILURE-MODE CLASSIFIERS")
print("=" * 76)

df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")


# ── features (identical to the binary pipeline) ──────────────────────────
def engineer(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    omega = X["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["Power [W]"]        = X["Torque [Nm]"] * omega
    X["Temp_diff [K]"]    = X["Process temperature [K]"] - X["Air temperature [K]"]
    X["Wear_x_Torque"]    = X["Tool wear [min]"] * X["Torque [Nm]"]
    X["Temp_power_ratio"] = X["Temp_diff [K]"] / (X["Power [W]"] / 1000)
    return X


BASE = ["Type_encoded", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

df["Type_encoded"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
X_all = engineer(df[BASE])
FEATURES = list(X_all.columns)


def smote(X, y, k=5, seed=RANDOM_STATE):
    """Same implementation the binary pipeline uses."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    X_min = X[y == 1]
    need = int((y == 0).sum() - (y == 1).sum())
    if need <= 0 or len(X_min) <= k:
        return X, y
    nn = NearestNeighbors(n_neighbors=min(k, len(X_min) - 1) + 1).fit(X_min)
    idx = nn.kneighbors(X_min)[1][:, 1:]
    kk = idx.shape[1]
    base = rng.randint(0, len(X_min), need)
    neigh = idx[base, rng.randint(0, kk, need)]
    synth = X_min[base] + rng.rand(need, 1) * (X_min[neigh] - X_min[base])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(need, dtype=y.dtype)])


try:
    from imblearn.over_sampling import SMOTE as _S
    balance = lambda X, y: _S(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(X, y)
except ImportError:
    balance = lambda X, y: smote(X, y)

try:
    from xgboost import XGBClassifier
    def make_model():
        return XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08,
                             subsample=.9, colsample_bytree=.9, n_jobs=-1,
                             random_state=RANDOM_STATE, eval_metric="logloss")
    BACKEND = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def make_model():
        return HistGradientBoostingClassifier(max_iter=250, max_depth=5,
                                              learning_rate=0.08,
                                              random_state=RANDOM_STATE)
    BACKEND = "sklearn HistGradientBoosting"

print(f"\nBackend: {BACKEND}")
print(f"Features: {len(FEATURES)}\n")

results, bundle = [], {}

for code, label in MODES.items():
    y = df[code].values
    n_pos = int(y.sum())

    # Keep DataFrames through the split so the scaler stores column names —
    # the app passes named rows, and a scaler fitted on a bare array warns
    # on every single prediction.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y, test_size=.20, random_state=RANDOM_STATE, stratify=y)

    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

    X_bal, y_bal = balance(X_tr_s, y_tr)
    m = make_model()
    m.fit(X_bal, y_bal)

    proba = m.predict_proba(X_te_s)[:, 1]
    pred = (proba >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    row = {
        "code": code, "label": label, "positives": n_pos,
        "test_positives": int(y_te.sum()),
        "recall": recall_score(y_te, pred, zero_division=0),
        "precision": precision_score(y_te, pred, zero_division=0),
        "pr_auc": average_precision_score(y_te, proba),
        "roc_auc": roc_auc_score(y_te, proba) if y_te.sum() else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }
    results.append(row)
    bundle[code] = {"model": m, "scaler": sc}

    print(f"  {code} — {label}")
    print(f"     {n_pos} positives in full data, {int(y_te.sum())} in test")
    print(f"     recall {row['recall']:.3f}   precision {row['precision']:.3f}   "
          f"PR-AUC {row['pr_auc']:.3f}")
    print(f"     caught {tp}, missed {fn}, false alarms {fp}\n")

joblib.dump({"modes": bundle, "features": FEATURES}, MODELS / "mode_models.joblib")

res = pd.DataFrame(results)
res.to_csv(ROOT / "models" / "mode_performance.csv", index=False)

print("-" * 76)
print(res[["code", "label", "positives", "recall", "precision", "pr_auc"]]
      .round(3).to_string(index=False))

print("\n" + "-" * 76)
print("RNF (random failure) is deliberately excluded: the dataset generates it")
print("with a flat 0.1% probability independent of every sensor reading, so it")
print("is unpredictable by construction. Reporting a score for it would be")
print("misleading. It accounts for about 5.6% of flagged failures, which puts")
print("a real ceiling on any model's achievable recall.")

print("\nSaved: models/mode_models.joblib, models/mode_performance.csv")
print("=" * 76)
