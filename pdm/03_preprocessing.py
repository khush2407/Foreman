"""
STAGE 3 — PREPROCESSING
========================
Goal: turn the raw CSV into a clean modelling table and split it into train
and test sets, WITHOUT leaking any information from test into train.

Decisions made here (all justified in the report's Methodology section):

  1. DROP identifier columns (UDI, Product ID) — they are row labels, not
     physics. Product ID even encodes the answer partially (its first letter
     is the Type), so keeping it would be laundered leakage.

  2. DROP the five failure sub-type columns (TWF/HDF/PWF/OSF/RNF). Stage 1
     proved 'Machine failure' is the OR of these flags. Keeping them gives the
     model the answer and produces a fake ~100% score.

  3. ENCODE 'Type' ordinally (L=0, M=1, H=2) rather than one-hot. The three
     levels are an ordered quality scale, and Stage 1 showed failure rate
     decreases monotonically L -> M -> H, so the ordering is real information.

  4. SPLIT BEFORE ANY SCALING OR RESAMPLING. Stratified 80/20 so the test set
     keeps the same 3.39% failure rate as the full dataset.

Everything that LEARNS from the data (the scaler's mean/std, SMOTE's synthetic
points) is fitted in Stage 4 on the TRAINING SET ONLY.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20

print("=" * 78)
print("STAGE 3 — PREPROCESSING")
print("=" * 78)

df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")
print(f"\nRaw dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── 3.1 Drop identifiers ──────────────────────────────────────────────────
ID_COLS = ["UDI", "Product ID"]
print(f"\n[3.1] Dropping identifier columns: {ID_COLS}")
print("      Reason: row labels carry no physical meaning. 'Product ID' also")
print("      embeds the Type letter, which would be a second copy of a feature.")

# ── 3.2 Drop leakage columns ──────────────────────────────────────────────
LEAK_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print(f"\n[3.2] Dropping failure sub-type columns: {LEAK_COLS}")
print("      Reason: TARGET LEAKAGE. Stage 1 showed Machine failure = OR(these).")
print("      A model given these columns scores ~100% and learns nothing about")
print("      the sensors. This is the single most common mistake made with this")
print("      dataset in published student work.")

# Keep a copy of the sub-types for later descriptive use (Stage 8 needs them
# to break savings down by failure cause) — but they never enter X.
df[LEAK_COLS + ["Machine failure"]].to_csv(PROC / "failure_modes_reference.csv",
                                           index=False)

# ── 3.3 Encode Type ───────────────────────────────────────────────────────
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
df["Type_encoded"] = df["Type"].map(TYPE_MAP)
print(f"\n[3.3] Encoded 'Type' ordinally: {TYPE_MAP}")
print("      Reason: an ordered quality scale, not unordered categories.")
print("      Stage 1 failure rates: L 3.92%  >  M 2.77%  >  H 2.09%  (monotonic)")

# ── 3.4 Assemble the modelling table ──────────────────────────────────────
FEATURES = [
    "Type_encoded",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\n[3.4] Feature matrix assembled")
print(f"      X: {X.shape[0]:,} rows x {X.shape[1]} features")
print(f"      Features: {FEATURES}")
print(f"      Target:   '{TARGET}'  ({y.sum()} positives, {y.mean():.2%})")

# ── 3.5 Stratified train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,                 # <- keeps the failure rate identical in both
)

print(f"\n[3.5] Stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split "
      f"(random_state={RANDOM_STATE})")
print(f"      Train: {len(X_train):,} rows | {int(y_train.sum())} failures "
      f"({y_train.mean():.2%})")
print(f"      Test:  {len(X_test):,} rows | {int(y_test.sum())} failures "
      f"({y_test.mean():.2%})")
print("      The two failure rates match, which confirms stratification worked.")
print("\n      NOTE: the test set is now sealed. It is never scaled using its own")
print("      statistics and never resampled. It stays a faithful sample of")
print("      reality so the reported metrics mean something.")

# ── 3.6 Persist ───────────────────────────────────────────────────────────
X_train.to_csv(PROC / "X_train_raw.csv", index=False)
X_test.to_csv(PROC / "X_test_raw.csv",   index=False)
y_train.to_csv(PROC / "y_train.csv", index=False)
y_test.to_csv(PROC / "y_test.csv",   index=False)

print(f"\n[3.6] Saved raw (unscaled) splits to {PROC}")
print("\n" + "=" * 78)
print("STAGE 3 COMPLETE")
print("=" * 78)
