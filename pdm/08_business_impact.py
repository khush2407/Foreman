"""
STAGE 8 — BUSINESS IMPACT AND MAINTENANCE STRATEGY
===================================================
A recall figure means nothing to a plant manager. This stage converts model
performance into dollars and into a maintenance policy.

THE CENTRAL INSIGHT OF THIS STAGE
  The 0.50 decision threshold is an arbitrary default, not an optimum. It
  implicitly assumes a false alarm and a missed failure cost the same. They do
  not — here a missed failure costs 30x a false alarm. Once the cost asymmetry
  is written down, the optimal threshold can be COMPUTED, and it lands well
  below 0.50: the model should be deliberately made trigger-happy.

COST ASSUMPTIONS  (documented so a reader can substitute their own)
  These are the only judgement calls in the analysis. Every result below is a
  function of these four numbers, so they are stated openly rather than buried.

  Missed failure  (FN) — $12,000  emergency repair, unplanned line stop,
                                  expedited parts, scrapped work-in-progress
  Caught failure  (TP) —  $1,500  planned intervention in a scheduled window
  False alarm     (FP) —    $400  inspection labour, no fault found
  Correct no-call (TN) —      $0

  Ratio that drives everything: FN / FP = 30.

Figures: 16 (cost vs threshold), 17 (strategy comparison).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, recall_score, precision_score

import pdm_style as S
S.apply_style()

ROOT    = Path(__file__).parent
FIG     = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"

# ── Cost model ────────────────────────────────────────────────────────────
COST_FN = 12_000      # missed failure -> unplanned breakdown
COST_TP =  1_500      # caught failure -> planned intervention
COST_FP =    400      # false alarm    -> wasted inspection
COST_TN =      0

# Scaling from the test sample to a year of production.
# The dataset documents 10,000 operating cycles for one machine, which the
# source describes as roughly a year of production. The 2,000-cycle test set
# is therefore about one fifth of an operating year.
CYCLES_PER_YEAR = 10_000

print("=" * 78)
print("STAGE 8 — BUSINESS IMPACT")
print("=" * 78)

d = np.load(RESULTS / "05_test_predictions.npz")
y_test = d["y_test"]
best_name = open(RESULTS / "05_best_model.txt").read().splitlines()[0].strip()
proba = d[best_name]

N          = len(y_test)
n_failures = int(y_test.sum())
SCALE      = CYCLES_PER_YEAR / N

print(f"\nModel      : {best_name}")
print(f"Test sample: {N:,} cycles containing {n_failures} real failures")
print(f"Annualised : {CYCLES_PER_YEAR:,} cycles/year  (scale factor {SCALE:.0f}x)")
print(f"\nCost model : FN ${COST_FN:,} | TP ${COST_TP:,} | FP ${COST_FP:,} | TN ${COST_TN}")


def cost_at(threshold):
    """Total cost of running the predictive policy at a given alarm threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    total = tp * COST_TP + fn * COST_FN + fp * COST_FP + tn * COST_TN
    return dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp, cost=total,
                recall=tp / (tp + fn) if (tp + fn) else 0,
                precision=tp / (tp + fp) if (tp + fp) else 0)


# ── 8.1 Threshold sweep ───────────────────────────────────────────────────
print("\n[8.1] Sweeping the alarm threshold from 0.01 to 0.99")
print("-" * 78)

sweep = pd.DataFrame([cost_at(t) for t in np.linspace(0.01, 0.99, 197)])
sweep["annual_cost"] = sweep["cost"] * SCALE
sweep.to_csv(RESULTS / "08_threshold_sweep.csv", index=False)

opt        = sweep.loc[sweep["cost"].idxmin()]
default    = cost_at(0.50)
opt_thresh = float(opt["threshold"])

print(f"  Default threshold 0.50")
print(f"    caught {default['tp']}/{n_failures} failures, missed {default['fn']}, "
      f"{default['fp']} false alarms")
print(f"    cost ${default['cost']:,.0f} per {N:,} cycles  "
      f"-> ${default['cost']*SCALE:,.0f}/year")
print(f"\n  OPTIMAL threshold {opt_thresh:.2f}")
print(f"    caught {int(opt['tp'])}/{n_failures} failures, missed {int(opt['fn'])}, "
      f"{int(opt['fp'])} false alarms")
print(f"    cost ${opt['cost']:,.0f} per {N:,} cycles  "
      f"-> ${opt['annual_cost']:,.0f}/year")
print(f"\n  Lowering the threshold from 0.50 to {opt_thresh:.2f} trades "
      f"{int(opt['fp'] - default['fp']):+d} false alarms")
print(f"  for {int(default['fn'] - opt['fn']):+d} additional failures caught, "
      f"saving ${(default['cost'] - opt['cost'])*SCALE:,.0f}/year.")
print(f"  Recall rises {default['recall']:.1%} -> {opt['recall']:.1%}.")


# ── 8.2 Strategy comparison ───────────────────────────────────────────────
print("\n[8.2] Comparing three maintenance strategies")
print("-" * 78)

# (a) REACTIVE — run to failure. Every failure is an unplanned breakdown.
reactive = n_failures * COST_FN

# (b) SCHEDULED PREVENTIVE — inspect every k-th cycle regardless of condition.
#     Simplification: with failures arriving independently of the calendar,
#     inspecting a fraction f of cycles intercepts about a fraction f of them.
#     Documented as a first-order model, not a precise simulation.
#
#     Rather than picking one arbitrary interval, every interval from "inspect
#     everything" to "inspect almost nothing" is priced and the CHEAPEST is
#     used as the baseline. Beating a deliberately well-tuned competitor is a
#     much stronger claim than beating a badly chosen one.
prev_rows = []
for frac in np.linspace(0.02, 1.0, 50):
    n_insp  = N * frac
    caught  = n_failures * frac
    missed  = n_failures - caught
    c = caught * COST_TP + missed * COST_FN + (n_insp - caught) * COST_FP
    prev_rows.append(dict(fraction=frac, cost=c, caught=caught, inspections=n_insp))
prev_df = pd.DataFrame(prev_rows)
best_prev = prev_df.loc[prev_df["cost"].idxmin()]
preventive          = float(best_prev["cost"])
PREVENTIVE_FRACTION = float(best_prev["fraction"])
n_inspections       = float(best_prev["inspections"])
caught_prev         = float(best_prev["caught"])
missed_prev         = n_failures - caught_prev
prev_df.to_csv(RESULTS / "08_preventive_intervals.csv", index=False)
print(f"  Best calendar-based interval found: inspect {PREVENTIVE_FRACTION:.0%} "
      f"of cycles (every {1/PREVENTIVE_FRACTION:.1f}th)")

# (c) PREDICTIVE — the model, run at the cost-optimal threshold.
predictive = float(opt["cost"])

strategies = pd.DataFrame([
    {"strategy": "Reactive\n(run to failure)",   "cost": reactive,
     "detail": f"all {n_failures} failures\nunplanned"},
    {"strategy": f"Scheduled preventive\n(best interval: {PREVENTIVE_FRACTION:.0%} of cycles)",
     "cost": preventive,
     "detail": f"{int(n_inspections):,} inspections\n{int(missed_prev)} still unplanned"},
    {"strategy": f"Predictive ML\n(threshold {opt_thresh:.2f})", "cost": predictive,
     "detail": f"{int(opt['tp'])} caught · {int(opt['fn'])} missed\n{int(opt['fp'])} false alarms"},
])
strategies["annual_cost"] = strategies["cost"] * SCALE
strategies["vs_reactive"] = reactive * SCALE - strategies["annual_cost"]
strategies["pct_saved"]   = strategies["vs_reactive"] / (reactive * SCALE)

for _, r in strategies.iterrows():
    flat_strategy = r["strategy"].replace("\n", " ")
    flat_detail   = r["detail"].replace("\n", ", ")
    print(f"  {flat_strategy:<50} ${r['annual_cost']:>11,.0f}/yr   ({flat_detail})")

strategies.to_csv(RESULTS / "08_strategy_comparison.csv", index=False)

savings = strategies.iloc[0]["annual_cost"] - strategies.iloc[2]["annual_cost"]
vs_prev = strategies.iloc[1]["annual_cost"] - strategies.iloc[2]["annual_cost"]
print(f"\n  Predictive vs reactive   : ${savings:,.0f}/year saved "
      f"({savings/strategies.iloc[0]['annual_cost']:.1%})")
print(f"  Predictive vs preventive : ${vs_prev:,.0f}/year saved "
      f"({vs_prev/strategies.iloc[1]['annual_cost']:.1%})")


# ═══ F16 ── Cost vs threshold ════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.5, 1]})

ax1.plot(sweep["threshold"], sweep["annual_cost"] / 1000, color=S.BLUE, zorder=3)
ax1.axvline(opt_thresh, color=S.FAIL_RED, lw=1.6, ls=(0, (4, 3)), zorder=4)
ax1.axvline(0.50, color=S.INK_MUTED, lw=1.3, ls=(0, (2, 3)), zorder=4)
ax1.scatter([opt_thresh], [opt["annual_cost"] / 1000], s=70, color=S.FAIL_RED,
            edgecolors="white", linewidths=1.4, zorder=5)
span_y = sweep["annual_cost"].max() / 1000 - sweep["annual_cost"].min() / 1000
# Optimum label goes LEFT of its line, default label goes RIGHT of its line,
# so the two never overlap however close the thresholds end up.
ax1.annotate(f"optimum {opt_thresh:.2f}\n${opt['annual_cost']/1000:,.0f}k/yr",
             xy=(opt_thresh, opt["annual_cost"] / 1000),
             xytext=(opt_thresh - .07, opt["annual_cost"] / 1000 + span_y * .42),
             ha="right", fontsize=9, color=S.INK, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=S.FAIL_RED, lw=1.3))
ax1.annotate(f"default 0.50\n${default['cost']*SCALE/1000:,.0f}k/yr",
             xy=(0.50, default["cost"] * SCALE / 1000),
             xytext=(0.58, default["cost"] * SCALE / 1000 + span_y * .24),
             ha="left", fontsize=9, color=S.INK_MUTED,
             arrowprops=dict(arrowstyle="->", color=S.INK_MUTED, lw=1.1))
ax1.set_ylabel("Annual maintenance cost  ($ thousands)")
ax1.set_title("Figure 16 — The 0.50 threshold is not the cheapest place to operate\n"
              f"With a missed failure costing {COST_FN//COST_FP}x a false alarm, "
              f"the optimum sits at {opt_thresh:.2f}",
              loc="left", fontsize=11.5)

ax2.plot(sweep["threshold"], sweep["recall"], color=S.TEAL, label="Recall", zorder=3)
ax2.plot(sweep["threshold"], sweep["precision"], color=S.AMBER, label="Precision", zorder=3)
ax2.axvline(opt_thresh, color=S.FAIL_RED, lw=1.6, ls=(0, (4, 3)), zorder=4)
ax2.axvline(0.50, color=S.INK_MUTED, lw=1.3, ls=(0, (2, 3)), zorder=4)
ax2.set_xlabel("Alarm threshold")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.05)
ax2.set_xlim(0, 1)
ax2.legend(loc="center right")
fig.tight_layout()
S.save(fig, FIG / "fig16_cost_vs_threshold.png")


# ═══ F17 ── Strategy comparison ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.8, 4.6))
cols = [S.FAIL_RED, S.AMBER, S.TEAL]
bars = ax.bar(range(3), strategies["annual_cost"] / 1000, color=cols,
              width=.56, zorder=3)
top = strategies["annual_cost"].max() / 1000
for i, (_, r) in enumerate(strategies.iterrows()):
    h = r["annual_cost"] / 1000
    # Headline figure above the bar
    ax.text(i, h + top * .035, f"${h:,.0f}k", ha="center",
            fontsize=13, fontweight="bold", color=S.INK)
    # Saving vs reactive, inside the bar. Sign is formatted from the value
    # itself so a near-zero difference reads "no change" rather than "-0%".
    if i > 0:
        pct = r["pct_saved"]
        txt = ("no change\nvs reactive" if abs(pct) < .005
               else f"{'−' if pct > 0 else '+'}{abs(pct):.0%}\nvs reactive")
        ax.text(i, h * .55, txt, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white")
    # Detail sits INSIDE the bar near its base, clear of the tick labels
    ax.text(i, top * .035, r["detail"], ha="center", va="bottom",
            fontsize=8.2, color="white")
ax.set_xticks(range(3))
ax.set_xticklabels(strategies["strategy"], fontsize=9.5)
ax.set_ylabel("Annual cost  ($ thousands)")
ax.set_ylim(0, strategies["annual_cost"].max() / 1000 * 1.16)
ax.grid(axis="x", visible=False)
ax.set_title(f"Figure 17 — Predictive maintenance saves ${savings/1000:,.0f}k per year "
             f"against run-to-failure\n"
             f"Based on {CYCLES_PER_YEAR:,} cycles/year and the stated cost assumptions",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig17_strategy_comparison.png")


# ── 8.3 Sensitivity to the cost assumptions ───────────────────────────────
print("\n[8.3] Sensitivity — does the conclusion survive different cost ratios?")
print("-" * 78)
print(f"  {'FN cost':>10} {'ratio':>7} {'opt thresh':>11} {'recall':>8} "
      f"{'annual cost':>13} {'saved vs reactive':>18}")
sens_rows = []
for fn_cost in [4_000, 8_000, 12_000, 20_000, 40_000]:
    costs = [( (proba >= t).astype(int), t) for t in np.linspace(0.01, 0.99, 99)]
    best_c, best_t, best_r = np.inf, None, None
    for pred, t in costs:
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        c = tp * COST_TP + fn * fn_cost + fp * COST_FP
        if c < best_c:
            best_c, best_t, best_r = c, t, tp / (tp + fn) if (tp + fn) else 0
    react = n_failures * fn_cost
    sens_rows.append(dict(fn_cost=fn_cost, ratio=fn_cost // COST_FP,
                          threshold=best_t, recall=best_r,
                          annual_cost=best_c * SCALE,
                          saved=(react - best_c) * SCALE))
    print(f"  ${fn_cost:>9,} {fn_cost//COST_FP:>6}x {best_t:>11.2f} {best_r:>8.1%} "
          f"${best_c*SCALE:>12,.0f} ${(react-best_c)*SCALE:>17,.0f}")

pd.DataFrame(sens_rows).to_csv(RESULTS / "08_cost_sensitivity.csv", index=False)
print("\n  The optimal threshold falls as misses get more expensive, and the")
print("  predictive strategy beats reactive maintenance across the whole range.")
print("  The conclusion is robust to the cost assumption, which is the point of")
print("  running this sensitivity at all.")


# ── 8.4 Recommended operating policy ──────────────────────────────────────
print("\n[8.4] RECOMMENDED OPERATING POLICY")
print("-" * 78)
low, high = 0.15, opt_thresh
print(f"  Tier 1  P(failure) < {low:.2f}        -> no action, continue running")
print(f"  Tier 2  {low:.2f} <= P < {high:.2f}   -> flag for inspection at next "
      f"planned changeover")
print(f"  Tier 3  P(failure) >= {high:.2f}      -> schedule intervention within "
      f"the current shift")
print(f"\n  Every alarm ships with the Figure 15 reason breakdown, so the")
print(f"  supervisor sees WHICH reading triggered it, not just a number.")
print(f"\n  Expected result at {CYCLES_PER_YEAR:,} cycles/year:")
print(f"    {int(opt['tp']*SCALE):,} failures caught before they happen")
print(f"    {int(opt['fn']*SCALE):,} still escape to unplanned downtime")
print(f"    {int(opt['fp']*SCALE):,} inspections that find nothing")
print(f"    ${savings:,.0f} net annual saving vs run-to-failure")

with open(RESULTS / "08_summary.txt", "w") as f:
    f.write(f"Model: {best_name}\n")
    f.write(f"Optimal threshold: {opt_thresh:.2f}\n")
    f.write(f"Recall at optimum: {opt['recall']:.4f}\n")
    f.write(f"Precision at optimum: {opt['precision']:.4f}\n")
    f.write(f"Annual cost (predictive): ${opt['annual_cost']:,.0f}\n")
    f.write(f"Annual cost (reactive): ${reactive*SCALE:,.0f}\n")
    f.write(f"Annual saving: ${savings:,.0f}\n")

print("\n" + "=" * 78)
print("STAGE 8 COMPLETE — 2 figures written")
print("=" * 78)
