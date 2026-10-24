"""
STAGE 6 — MODEL EVALUATION
===========================
All four models are scored on the SEALED test set: 2,000 real cycles with the
real 3.4% failure rate. Nothing here was resampled.

WHY ACCURACY IS BANNED FROM THIS REPORT
  A model that predicts "no failure" 2,000 times scores 96.6% accuracy and
  catches zero failures. Accuracy is reported in the table only to demonstrate
  that point, never used to choose anything.

THE METRICS THAT MATTER, AND WHY
  Recall     = of all real failures, what fraction did we catch?
               A miss means unplanned downtime. This is the expensive error.
  Precision  = of all the alarms we raised, what fraction were real?
               A false alarm means an unnecessary inspection. Cheap, annoying.
  F1         = harmonic mean of the two.
  PR-AUC     = quality of the ranking across ALL thresholds. Threshold-free,
               and unlike ROC-AUC it is not inflated by the 1,932 easy negatives.
  ROC-AUC    = reported for comparability with published literature, with the
               caveat above.

Figures produced: 9 (confusion matrices), 10 (ROC), 11 (PR), 12 (comparison).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

import pdm_style as S
S.apply_style()

ROOT    = Path(__file__).parent
FIG     = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"

print("=" * 78)
print("STAGE 6 — MODEL EVALUATION")
print("=" * 78)

d = np.load(RESULTS / "05_test_predictions.npz")
y_test = d["y_test"]
MODEL_NAMES = [k for k in d.files if k != "y_test"]
probas = {name: d[name] for name in MODEL_NAMES}

# Fixed colour per model — identity never changes between figures
COLORS = dict(zip(MODEL_NAMES, S.CATEGORICAL))

print(f"\nTest set: {len(y_test):,} cycles | {int(y_test.sum())} real failures "
      f"({y_test.mean():.2%})")
print(f"Models evaluated: {', '.join(MODEL_NAMES)}\n")


# ── 6.1 Metric table ──────────────────────────────────────────────────────
rows = []
for name, p in probas.items():
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall":    recall_score(y_test, pred, zero_division=0),
        "F1":        f1_score(y_test, pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, p),
        "PR-AUC":    average_precision_score(y_test, p),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    })

res = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)

print("[6.1] Test-set performance at the default 0.50 threshold")
print("-" * 78)
show = res.copy()
for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
    show[c] = show[c].map("{:.4f}".format)
print(show.to_string(index=False))
res.to_csv(RESULTS / "06_model_comparison.csv", index=False)

best = res.iloc[0]["Model"]
print(f"\n  Winner by PR-AUC: {best}")

# The accuracy trap, demonstrated numerically
naive_acc = 1 - y_test.mean()
print(f"\n  Reference point — a model that always predicts 'no failure':")
print(f"    Accuracy {naive_acc:.4f}  |  Recall 0.0000  |  failures caught: 0 of {int(y_test.sum())}")
print(f"    It beats Logistic Regression on accuracy. That is why accuracy is")
print(f"    not the selection metric in this study.")

print(f"\n[6.2] Full classification report — {best}")
print("-" * 78)
print(classification_report(y_test, (probas[best] >= 0.5).astype(int),
                            target_names=["No failure", "Failure"], digits=4))
with open(RESULTS / "06_classification_report.txt", "w") as f:
    f.write(classification_report(y_test, (probas[best] >= 0.5).astype(int),
                                  target_names=["No failure", "Failure"], digits=4))


# ═══ F9 ── Confusion matrices ════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
for ax, name in zip(axes, res["Model"]):
    cm = confusion_matrix(y_test, (probas[name] >= 0.5).astype(int))
    ax.imshow(cm, cmap=S.SEQ_CMAP, vmin=0, vmax=cm.max())
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            # Colour the two error cells so mistakes are legible at a glance
            label = [["TN", "FP"], ["FN", "TP"]][i][j]
            ax.text(j, i - .12, f"{v:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if v > cm.max() * .5 else S.INK)
            ax.text(j, i + .24, label, ha="center", va="center", fontsize=9,
                    color="white" if v > cm.max() * .5 else S.INK_MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: OK", "Pred: Fail"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: OK", "Actual: Fail"], fontsize=9)
    r = recall_score(y_test, (probas[name] >= .5).astype(int), zero_division=0)
    pr = precision_score(y_test, (probas[name] >= .5).astype(int), zero_division=0)
    ax.set_title(f"{name}\nrecall {r:.2f} · precision {pr:.2f}", fontsize=10.5)
fig.suptitle("Figure 9 — Confusion matrices on the 2,000-cycle test set  "
             "(FN = missed failure, the expensive error)",
             x=.008, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .88])
S.save(fig, FIG / "fig09_confusion_matrices.png")


# ═══ F10 ── ROC curves ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.6, 5.4))
for name in res["Model"]:
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    ax.plot(fpr, tpr, color=COLORS[name], label=f"{name}  ({auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], color=S.INK_MUTED, lw=1.1, ls=(0, (4, 3)),
        label="Random  (0.500)")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_xlim(-.01, 1); ax.set_ylim(0, 1.01)
ax.legend(loc="lower right", title="Model  (AUC)", title_fontsize=9)
ax.set_title("Figure 10 — ROC curves\nAll curves look strong — "
             "a known optimism of ROC under class imbalance",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig10_roc_curves.png")


# ═══ F11 ── Precision-Recall curves (the honest view) ════════════════════
fig, ax = plt.subplots(figsize=(6.6, 5.4))
for name in res["Model"]:
    prec, rec, _ = precision_recall_curve(y_test, probas[name])
    ax.plot(rec, prec, color=COLORS[name],
            label=f"{name}  ({average_precision_score(y_test, probas[name]):.3f})")
base = y_test.mean()
ax.axhline(base, color=S.INK_MUTED, lw=1.1, ls=(0, (4, 3)))
# Right-aligned so it never runs into the legend in the lower-left corner
ax.text(.985, base + .025, f"Random baseline = {base:.3f}", ha="right",
        color=S.INK_MUTED, fontsize=9)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
ax.legend(loc="lower left", title="Model  (PR-AUC)", title_fontsize=9)
ax.set_title("Figure 11 — Precision-Recall curves\n"
             "The separation ROC hid: boosting and forests clearly lead",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig11_pr_curves.png")


# ═══ F12 ── Model comparison ═════════════════════════════════════════════
metrics = ["Precision", "Recall", "F1", "PR-AUC"]
fig, ax = plt.subplots(figsize=(9.6, 4.4))
n_m, n_mod = len(metrics), len(res)
group_w, bar_w = .78, .78 / len(res)
xs = np.arange(n_m)
for i, name in enumerate(res["Model"]):
    vals = [res.loc[res["Model"] == name, m].values[0] for m in metrics]
    offs = xs - group_w / 2 + bar_w * (i + .5)
    ax.bar(offs, vals, width=bar_w * .88, color=COLORS[name], label=name, zorder=3)
    for x, v in zip(offs, vals):
        ax.text(x, v + .018, f"{v:.2f}", ha="center", fontsize=8,
                color=S.INK_MUTED, rotation=0)
ax.set_xticks(xs); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.09)
ax.set_ylabel("Score")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=4)
ax.set_title("Figure 12 — Model comparison on the test set  "
             "(accuracy deliberately excluded)",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig12_model_comparison.png")


print("\n[6.3] Interpretation for the report")
print("-" * 78)
b = res.iloc[0]
print(f"  {b['Model']} caught {int(b['TP'])} of {int(b['TP'] + b['FN'])} real failures "
      f"({b['Recall']:.1%} recall)")
print(f"  and raised {int(b['FP'])} false alarms out of "
      f"{int(b['FP'] + b['TN']):,} healthy cycles ({b['Precision']:.1%} precision).")
print(f"  It missed {int(b['FN'])} failures — Stage 8 prices that miss in dollars")
print(f"  and shows the threshold can be moved to trade alarms for catches.")
print(f"\n  Note the ROC-AUC / PR-AUC gap: "
      f"{b['ROC-AUC']:.3f} vs {b['PR-AUC']:.3f}.")
print("  Reporting only ROC-AUC would have overstated this model substantially.")

print("\n" + "=" * 78)
print("STAGE 6 COMPLETE — 4 figures written")
print("=" * 78)
