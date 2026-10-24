"""
STAGE 1 — DATA ACQUISITION & UNDERSTANDING
===========================================
Goal: know exactly what is in the dataset before writing a single line of
modelling code. Everything printed here goes into the "Data Collection"
section of the report.

What this script answers:
  1. What shape is the data and what are the column types?
  2. Are there missing values or duplicate records?
  3. How rare is a failure? (the class-imbalance question)
  4. How do the 5 failure sub-types relate to the overall failure flag?
  5. What are the descriptive statistics for each sensor?

The single most important finding in this script is the LEAKAGE CHECK at the
end: the five sub-type columns are components of the target, not inputs.
"""
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT    = Path(__file__).parent
RESULTS = ROOT / "outputs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("STAGE 1 — DATA UNDERSTANDING")
print("=" * 78)

# ── 1.1 Load ──────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")
print(f"\n[1.1] Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

# ── 1.2 Structure ─────────────────────────────────────────────────────────
print("[1.2] Column types and non-null counts")
print("-" * 78)
structure = pd.DataFrame({
    "dtype":     df.dtypes.astype(str),
    "non_null":  df.notna().sum(),
    "nulls":     df.isna().sum(),
    "n_unique":  df.nunique(),
})
print(structure.to_string())
structure.to_csv(RESULTS / "01_column_structure.csv")

# ── 1.3 Data quality ──────────────────────────────────────────────────────
print("\n[1.3] Data quality checks")
print("-" * 78)
print(f"  Total missing values ........ {df.isna().sum().sum()}")
print(f"  Duplicate rows .............. {df.duplicated().sum()}")
print(f"  Duplicate Product IDs ....... {df['Product ID'].duplicated().sum()}")
print(f"  UDI is a clean 1..N index ... {bool((df['UDI'].values == np.arange(1, len(df) + 1)).all())}")

# ── 1.4 Target balance ────────────────────────────────────────────────────
print("\n[1.4] Target variable: 'Machine failure'")
print("-" * 78)
n_fail = int(df["Machine failure"].sum())
n_ok   = len(df) - n_fail
print(f"  No failure (0) .............. {n_ok:,}  ({n_ok/len(df):.2%})")
print(f"  Failure    (1) .............. {n_fail:,}  ({n_fail/len(df):.2%})")
print(f"  Imbalance ratio ............. 1 failure for every {n_ok/n_fail:.0f} healthy cycles")
print("\n  --> This is severe imbalance. A model that predicts 'no failure' every")
print("      single time would score 96.6% accuracy while being useless. This is")
print("      why the project reports RECALL and PR-AUC, not accuracy.")

# ── 1.5 Failure sub-types ─────────────────────────────────────────────────
print("\n[1.5] Failure sub-types")
print("-" * 78)
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LABELS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}
modes = pd.DataFrame({
    "description": [LABELS[m] for m in MODES],
    "count":       [int(df[m].sum()) for m in MODES],
})
modes["pct_of_all_rows"] = (modes["count"] / len(df) * 100).round(2)
modes["pct_of_failures"] = (modes["count"] / n_fail * 100).round(1)
print(modes.to_string())
modes.to_csv(RESULTS / "01_failure_modes.csv")
print(f"\n  Sum of sub-type flags ....... {modes['count'].sum()}")
print(f"  Actual failure rows ......... {n_fail}")
print("  (They differ because some rows carry more than one flag, and because")
print("   of the RNF quirk described in 1.6.)")

# ── 1.6 LEAKAGE CHECK — the critical finding ──────────────────────────────
print("\n[1.6] *** LEAKAGE CHECK — most important step in this script ***")
print("-" * 78)
any_mode = df[MODES].max(axis=1)
agree    = (any_mode == df["Machine failure"])
print(f"  Rows where OR(sub-types) == Machine failure: {agree.sum():,} / {len(df):,}")

mismatch = df[~agree]
print(f"  Mismatched rows: {len(mismatch)}")
if len(mismatch):
    print("\n  Breakdown of the mismatches:")
    print(mismatch[MODES + ["Machine failure"]].sum().to_string())
    print("\n  These are RNF (random failure) rows flagged 1 while 'Machine failure'")
    print("  stayed 0 — a documented quirk of this dataset. RNF is random noise by")
    print("  construction and carries no predictive signal.")

print("\n  CONCLUSION: 'Machine failure' is essentially the OR of the sub-type")
print("  flags. Feeding TWF/HDF/PWF/OSF/RNF into a model that predicts")
print("  'Machine failure' would leak the answer and produce a fake ~100% score.")
print("  ==> All five sub-type columns are DROPPED from the feature set.")
print("      They are kept only for descriptive analysis of failure causes.")

# ── 1.7 Descriptive statistics ────────────────────────────────────────────
print("\n[1.7] Descriptive statistics — sensor readings")
print("-" * 78)
SENSORS = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
desc = df[SENSORS].describe().T
desc["range"] = desc["max"] - desc["min"]
desc["cv_%"]  = (desc["std"] / desc["mean"] * 100).round(2)   # coefficient of variation
print(desc.round(2).to_string())
desc.round(3).to_csv(RESULTS / "01_descriptive_stats.csv")

# ── 1.8 Product quality mix ───────────────────────────────────────────────
print("\n[1.8] Product quality variant ('Type')")
print("-" * 78)
qual = pd.DataFrame({
    "count":        df["Type"].value_counts(),
    "share":        df["Type"].value_counts(normalize=True).round(4),
    "failures":     df.groupby("Type")["Machine failure"].sum(),
    "failure_rate": df.groupby("Type")["Machine failure"].mean().round(4),
}).loc[["L", "M", "H"]]
qual.index = ["L (Low)", "M (Medium)", "H (High)"]
print(qual.to_string())
qual.to_csv(RESULTS / "01_quality_mix.csv")
print("\n  Low-quality parts fail most often and high-quality least — so 'Type'")
print("  carries real signal and belongs in the feature set.")

print("\n" + "=" * 78)
print("STAGE 1 COMPLETE — results written to outputs/results/")
print("=" * 78)
