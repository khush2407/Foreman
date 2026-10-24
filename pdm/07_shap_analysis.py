"""
STAGE 7 — MODEL EXPLAINABILITY
===============================
A maintenance supervisor will not shut down a machine because "the model said
so." This stage converts the model from a black box into something that gives
a reason with every alarm.

Three questions, three figures:

  F13  GLOBALLY, which features drive failure predictions?
  F14  In WHICH DIRECTION does each feature push the prediction?
  F15  For ONE specific alarm, why did this cycle get flagged?

METHOD
  Preferred: SHAP (SHapley Additive exPlanations). SHAP comes from cooperative
  game theory — it treats each feature as a "player" and fairly divides the
  model's output among them. Its guarantee is that the attributions sum exactly
  to the prediction, which no other importance method provides.

  Fallback (used when the shap package is unavailable): permutation importance
  plus partial dependence plus local ablation. These are slower and coarser but
  answer the same three questions and are entirely defensible in a report.
    - Permutation importance: shuffle one feature, measure how much PR-AUC
      drops. A big drop means the model relied on it.
    - Partial dependence: sweep one feature across its range, average the
      model's response. Shows the DIRECTION and shape of the effect.
    - Local ablation: for one row, replace each feature with the population
      median and see how far the predicted probability moves. That isolates
      each feature's contribution to this specific alarm.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import average_precision_score

import pdm_style as S
S.apply_style()

ROOT    = Path(__file__).parent
PROC    = ROOT / "data" / "processed"
FIG     = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
MODELS  = ROOT / "outputs" / "models"

print("=" * 78)
print("STAGE 7 — MODEL EXPLAINABILITY")
print("=" * 78)

d = np.load(PROC / "model_data.npz")
X_test, y_test = d["X_test_scaled"], d["y_test"]
X_bal,  y_bal  = d["X_train_balanced"], d["y_train_balanced"]
features = pd.read_csv(PROC / "feature_names.csv")["feature"].tolist()

best_name = open(RESULTS / "05_best_model.txt").read().splitlines()[0].strip()
model = joblib.load(MODELS / f"{best_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
print(f"\nExplaining: {best_name}")
print(f"Test set  : {X_test.shape[0]:,} cycles x {len(features)} features")

# Unscaled values, for reading real units in the local explanation
X_test_raw = pd.read_csv(PROC / "X_test_engineered.csv")

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
print(f"SHAP available: {HAVE_SHAP}"
      f"{'' if HAVE_SHAP else '  -> using permutation / PDP / ablation fallback'}")


# ── 7.1 Global importance ─────────────────────────────────────────────────
print("\n[7.1] Global feature importance")
print("-" * 78)

if HAVE_SHAP:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    importance = np.abs(shap_values).mean(axis=0)
    imp_label  = "Mean |SHAP value|"
else:
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1,
        scoring=lambda est, X, y: average_precision_score(y, est.predict_proba(X)[:, 1]))
    importance = perm.importances_mean
    imp_err    = perm.importances_std
    imp_label  = "Drop in PR-AUC when shuffled"

imp = pd.DataFrame({"feature": features, "importance": importance}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
imp["share"] = imp["importance"] / imp["importance"].sum()
imp["cumulative"] = imp["share"].cumsum()
print(imp.to_string(index=False))
imp.to_csv(RESULTS / "07_feature_importance.csv", index=False)

ENGINEERED = {"Power [W]", "Temp_diff [K]", "Wear_x_Torque", "Temp_power_ratio"}
eng_share = imp.loc[imp["feature"].isin(ENGINEERED), "share"].sum()
top3 = imp.head(3)
print(f"\n  Top 3 features explain {top3['share'].sum():.1%} of total importance.")
print(f"  The 4 ENGINEERED features account for {eng_share:.1%} — direct evidence")
print(f"  that the Stage 4 feature engineering was worth doing.")

# ═══ F13 ── Global importance ════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.4, 4.6))
order = imp.iloc[::-1]
cols = [S.AMBER if f in ENGINEERED else S.BLUE for f in order["feature"]]
bars = ax.barh(order["feature"], order["importance"], color=cols, height=.66, zorder=3)
if not HAVE_SHAP:
    errs = [imp_err[features.index(f)] for f in order["feature"]]
    ax.errorbar(order["importance"], range(len(order)), xerr=errs, fmt="none",
                ecolor=S.INK_MUTED, elinewidth=1, capsize=3, zorder=4)
for y, v in enumerate(order["importance"]):
    ax.text(v + imp["importance"].max() * .02, y, f"{v:.4f}",
            va="center", fontsize=8.5, color=S.INK_MUTED)
ax.set_xlim(0, imp["importance"].max() * 1.22)
ax.set_xlabel(imp_label)
ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=S.AMBER),
           plt.Rectangle((0, 0), 1, 1, color=S.BLUE)]
ax.legend(handles, ["Engineered feature", "Original sensor"], loc="lower right")
ax.set_title(f"Figure 13 — Global feature importance ({best_name})\n"
             f"Engineered features carry {eng_share:.0%} of the model's decision weight",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig13_feature_importance.png")


# ── 7.2 Direction of effect ───────────────────────────────────────────────
print("\n[7.2] Direction of effect for the top features")
print("-" * 78)

top_feats = imp.head(4)["feature"].tolist()
top_idx   = [features.index(f) for f in top_feats]

# ═══ F14 ── Beeswarm (SHAP) or partial dependence (fallback) ═════════════
if HAVE_SHAP:
    plt.figure(figsize=(8.6, 5.2))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    fig = plt.gcf()
    fig.suptitle("Figure 14 — SHAP beeswarm: how each feature pushes the prediction",
                 x=.008, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .93])
    S.save(fig, FIG / "fig14_shap_beeswarm.png")
else:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for ax, f, fi in zip(axes, top_feats, top_idx):
        pd_res = partial_dependence(model, X_test, [fi], kind="average",
                                    grid_resolution=45)
        grid_scaled = pd_res["grid_values"][0]
        # Convert the scaled x-axis back to real engineering units
        mu, sd = X_test_raw[f].mean(), X_test_raw[f].std()
        grid_real = grid_scaled * sd + mu
        ax.plot(grid_real, pd_res["average"][0], color=S.BLUE, zorder=3)
        ax.fill_between(grid_real, pd_res["average"][0].min(), pd_res["average"][0],
                        color=S.BLUE, alpha=.12, zorder=2)
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("value (real units)", fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Predicted P(failure)")
    fig.suptitle("Figure 14 — Partial dependence: direction and shape of each "
                 "feature's effect",
                 x=.008, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .88])
    S.save(fig, FIG / "fig14_partial_dependence.png")

for f in top_feats:
    fi = features.index(f)
    hi = X_test[:, fi] > np.percentile(X_test[:, fi], 75)
    lo = X_test[:, fi] < np.percentile(X_test[:, fi], 25)
    r_hi = model.predict_proba(X_test[hi])[:, 1].mean()
    r_lo = model.predict_proba(X_test[lo])[:, 1].mean()
    arrow = "increases" if r_hi > r_lo else "decreases"
    print(f"  {f:<22} high values -> P(fail) {r_hi:.3f} | "
          f"low values -> {r_lo:.3f}   ({arrow} risk)")


# ── 7.3 Explaining one alarm ──────────────────────────────────────────────
print("\n[7.3] Local explanation — one specific predicted failure")
print("-" * 78)

proba_all = model.predict_proba(X_test)[:, 1]
# Pick a true positive the model was confident about
candidates = np.where((y_test == 1) & (proba_all > 0.8))[0]
row = int(candidates[0])
print(f"  Test row {row}: actual = FAILURE, predicted P(failure) = {proba_all[row]:.3f}")

if HAVE_SHAP:
    contrib = shap_values[row]
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = np.ravel(base_val)[-1]
    contrib_label = "SHAP contribution"
else:
    # Local ablation: swap each feature to the population median, one at a
    # time, and record how far the predicted probability falls.
    median_row = np.median(X_test, axis=0)
    contrib = np.zeros(len(features))
    for i in range(len(features)):
        perturbed = X_test[row].copy()
        perturbed[i] = median_row[i]
        contrib[i] = proba_all[row] - model.predict_proba(perturbed.reshape(1, -1))[0, 1]
    base_val = proba_all[row] - contrib.sum()
    contrib_label = "Drop in P(failure) if set to median"

local = pd.DataFrame({
    "feature":     features,
    "value":       X_test_raw.iloc[row][features].values,
    "contribution": contrib,
}).sort_values("contribution", key=np.abs, ascending=False)
print("\n" + local.to_string(index=False))
local.to_csv(RESULTS / "07_local_explanation.csv", index=False)

# ═══ F15 ── Local explanation ════════════════════════════════════════════
plot_df = local.head(7).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
cols = [S.FAIL_RED if c > 0 else S.OK_GREEN for c in plot_df["contribution"]]
labels = [f"{f}\n= {v:,.1f}" for f, v in zip(plot_df["feature"], plot_df["value"])]
ax.barh(range(len(plot_df)), plot_df["contribution"], color=cols, height=.62, zorder=3)
ax.set_yticks(range(len(plot_df))); ax.set_yticklabels(labels, fontsize=8.5)
ax.axvline(0, color=S.INK_MUTED, lw=1)
span = np.abs(plot_df["contribution"]).max()
for y, c in enumerate(plot_df["contribution"]):
    ax.text(c + np.sign(c) * span * .04, y, f"{c:+.3f}",
            va="center", ha="left" if c > 0 else "right",
            fontsize=8.5, color=S.INK_MUTED)
ax.set_xlim(-span * 1.45, span * 1.45)
ax.set_xlabel(contrib_label)
ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=S.FAIL_RED),
           plt.Rectangle((0, 0), 1, 1, color=S.OK_GREEN)]
ax.legend(handles, ["Pushes toward failure", "Pushes toward healthy"],
          loc="lower right")
ax.set_title(f"Figure 15 — Why cycle #{row} was flagged   "
             f"(predicted P(failure) = {proba_all[row]:.2f}, actual = failure)\n"
             f"This is the reason text that would accompany a real alarm",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig15_local_explanation.png")

driver = local.iloc[0]
print(f"\n  Plain-language alarm text a supervisor would receive:")
print(f'    "Machine flagged at {proba_all[row]:.0%} failure risk. Main driver:')
print(f'     {driver["feature"]} at {driver["value"]:,.1f}."')

print("\n" + "=" * 78)
print("STAGE 7 COMPLETE — 3 figures written")
print("=" * 78)
