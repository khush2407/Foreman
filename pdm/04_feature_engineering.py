"""
STAGE 4 — FEATURE ENGINEERING, SCALING, AND CLASS BALANCING
============================================================
Goal: give the model the variables that actually describe the failure physics,
then put everything on a comparable scale, then fix the class imbalance.

ORDER MATTERS, and it is deliberate:

  engineer -> scale (fit on train) -> SMOTE (train only)

  * Engineering is row-wise arithmetic, so it is leakage-free on either side
    of the split. It runs first so the new columns get scaled too.
  * Scaling is FITTED ON TRAIN ONLY and then applied to test. Fitting on the
    full dataset would leak the test set's mean and variance into training.
  * SMOTE is applied to TRAIN ONLY. Synthesising minority points in the test
    set would mean reporting accuracy on data that does not exist.

THE FOUR ENGINEERED FEATURES — each one comes from a Stage 2 figure:

  1. Power [W] = Torque x AngularVelocity        (from Figure 6)
     Figure 6 showed failures cluster outside a safe power band. Power is the
     quantity the machine's own failure rule is written in, but the model was
     only being shown its two factors separately.

  2. Temp_diff [K] = Process temp - Air temp     (from Figure 8)
     Figure 4 showed the two temperatures are 0.88 correlated, so individually
     they are nearly redundant. Figure 8 showed their DIFFERENCE is what
     separates heat-dissipation failures. The difference de-correlates them.

  3. Wear_x_Torque [min-Nm] = Tool wear x Torque (from Figure 7)
     Figure 7 showed overstrain failures sit above a wear x torque threshold.
     A tree can only approximate that hyperbola with many axis-aligned splits;
     handing it the product makes the boundary a single split.

  4. Temp_power_ratio = Temp_diff / (Power/1000)
     A composite thermal-load proxy: how much heat headroom exists relative to
     the mechanical work being done.

SMOTE (Synthetic Minority Over-sampling Technique):
  Rather than duplicating the 271 training failures, SMOTE draws new points
  along the line segments between a failure and its k nearest failure
  neighbours. That fills the minority region instead of stacking copies on
  the same coordinates, which is what makes it better than naive oversampling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RANDOM_STATE = 42

print("=" * 78)
print("STAGE 4 — FEATURE ENGINEERING, SCALING, BALANCING")
print("=" * 78)

X_train = pd.read_csv(PROC / "X_train_raw.csv")
X_test  = pd.read_csv(PROC / "X_test_raw.csv")
y_train = pd.read_csv(PROC / "y_train.csv").squeeze("columns")
y_test  = pd.read_csv(PROC / "y_test.csv").squeeze("columns")


# ── 4.1 Engineered features ───────────────────────────────────────────────
def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature construction. No cross-row statistics, so this is
    safe to apply independently to train and test — there is nothing to leak."""
    X = X.copy()

    # Angular velocity in rad/s, then mechanical power in watts
    omega = X["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["Power [W]"] = X["Torque [Nm]"] * omega

    # Heat dissipation headroom
    X["Temp_diff [K]"] = X["Process temperature [K]"] - X["Air temperature [K]"]

    # Overstrain proxy
    X["Wear_x_Torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]

    # Thermal load relative to mechanical work (kW to keep the ratio readable)
    X["Temp_power_ratio"] = X["Temp_diff [K]"] / (X["Power [W]"] / 1000)

    return X


X_train_fe = engineer(X_train)
X_test_fe  = engineer(X_test)

NEW = ["Power [W]", "Temp_diff [K]", "Wear_x_Torque", "Temp_power_ratio"]
print(f"\n[4.1] Engineered {len(NEW)} new features")
print(f"      Feature count: {X_train.shape[1]} -> {X_train_fe.shape[1]}")

# How much signal did each new feature actually add?
print("\n      Correlation with the target (training set):")
print("      " + "-" * 52)
corrs = X_train_fe.corrwith(y_train).abs().sort_values(ascending=False)
for name, c in corrs.items():
    tag = "  <- engineered" if name in NEW else ""
    print(f"      {name:<28} {c:.4f}{tag}")

best_new = corrs[corrs.index.isin(NEW)].idxmax()
best_old = corrs[~corrs.index.isin(NEW)].idxmax()
print(f"\n      Strongest engineered feature: {best_new} ({corrs[best_new]:.4f})")
print(f"      Strongest original feature:   {best_old} ({corrs[best_old]:.4f})")


# ── 4.2 Scaling ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_fe)     # FIT on train
X_test_sc  = scaler.transform(X_test_fe)          # APPLY to test — never refit

print(f"\n[4.2] StandardScaler fitted on TRAIN ONLY, applied to both")
print(f"      Train mean after scaling: {X_train_sc.mean():+.2e}  (~0 as expected)")
print(f"      Train std  after scaling: {X_train_sc.std():.4f}   (~1 as expected)")
print(f"      Test  mean after scaling: {X_test_sc.mean():+.4f}  "
      f"(NOT exactly 0 — correct, it used the training mean)")
print("      Scaling matters for Logistic Regression and SVM, which are")
print("      distance/geometry based. Trees are scale-invariant but are given")
print("      the same matrix so every model sees identical inputs.")


# ── 4.3 SMOTE ─────────────────────────────────────────────────────────────
def smote(X, y, k=5, random_state=42):
    """Minimal faithful SMOTE implementation.

    For each synthetic sample: pick a random minority point, pick one of its k
    nearest minority neighbours, and place a new point at a random position on
    the segment joining them. Used as a fallback when imbalanced-learn is not
    installed; the library version is preferred when available.
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    X_min = X[y == 1]
    n_needed = int((y == 0).sum() - (y == 1).sum())

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    idx = idx[:, 1:]                                  # column 0 is the point itself

    base_i  = rng.randint(0, len(X_min), n_needed)
    neigh_i = idx[base_i, rng.randint(0, k, n_needed)]
    gap     = rng.rand(n_needed, 1)
    synth   = X_min[base_i] + gap * (X_min[neigh_i] - X_min[base_i])

    return (np.vstack([X, synth]),
            np.concatenate([y, np.ones(n_needed, dtype=y.dtype)]))


try:
    from imblearn.over_sampling import SMOTE as ImbSMOTE
    X_bal, y_bal = ImbSMOTE(random_state=RANDOM_STATE, k_neighbors=5) \
        .fit_resample(X_train_sc, y_train)
    smote_impl = "imbalanced-learn"
except ImportError:
    X_bal, y_bal = smote(X_train_sc, y_train, k=5, random_state=RANDOM_STATE)
    smote_impl = "built-in fallback (identical algorithm)"

y_bal = np.asarray(y_bal)
print(f"\n[4.3] SMOTE applied to the TRAINING SET ONLY  [{smote_impl}]")
print(f"      Before:  {len(y_train):,} rows | {int(np.sum(y_train)):>5} failures "
      f"({np.mean(y_train):.2%})")
print(f"      After:   {len(y_bal):,} rows | {int(y_bal.sum()):>5} failures "
      f"({y_bal.mean():.2%})")
print(f"      Synthetic samples created: {len(y_bal) - len(y_train):,}")
print(f"\n      The TEST set is untouched: {len(y_test):,} rows, "
      f"{int(y_test.sum())} failures ({y_test.mean():.2%})")
print("      This is the whole point — the model trains on a balanced world")
print("      but is scored on the real, imbalanced one.")


# ── 4.4 Persist everything ────────────────────────────────────────────────
np.savez_compressed(
    PROC / "model_data.npz",
    X_train_balanced=X_bal,       y_train_balanced=y_bal,
    X_train_scaled=X_train_sc,    y_train=np.asarray(y_train),
    X_test_scaled=X_test_sc,      y_test=np.asarray(y_test),
)
joblib.dump(scaler, PROC / "scaler.joblib")
pd.Series(list(X_train_fe.columns)).to_csv(PROC / "feature_names.csv",
                                           index=False, header=["feature"])
X_train_fe.to_csv(PROC / "X_train_engineered.csv", index=False)
X_test_fe.to_csv(PROC / "X_test_engineered.csv",   index=False)

print(f"\n[4.4] Saved to {PROC}")
print(f"      model_data.npz  — balanced train, scaled train, scaled test")
print(f"      scaler.joblib   — needed to score new data in production")
print("\n" + "=" * 78)
print("STAGE 4 COMPLETE")
print("=" * 78)
