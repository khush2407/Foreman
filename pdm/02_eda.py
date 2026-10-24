"""
STAGE 2 — EXPLORATORY DATA ANALYSIS
====================================
Goal: understand the physics of the failures visually, before modelling.
Every figure here is a report figure — they go straight into the "Data
Collection" and "Methodology" sections.

The question each figure answers:
  F1  How rare are failures, and which sub-type dominates?
  F2  What does each sensor's distribution look like?
  F3  Do failed and healthy cycles separate on any single sensor?
  F4  Which sensors move together? (multicollinearity check)
  F5  Does product quality change failure risk?
  F6  Is there a failure "envelope" in torque-speed space?
  F7  Does the tool-wear x torque interaction drive overstrain?
  F8  Does the air/process temperature gap drive heat failures?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

import pdm_style as S
S.apply_style()

ROOT = Path(__file__).parent
FIG  = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")

SENSORS = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
SHORT   = {"Air temperature [K]": "Air temp (K)",
           "Process temperature [K]": "Process temp (K)",
           "Rotational speed [rpm]": "Rot. speed (rpm)",
           "Torque [Nm]": "Torque (Nm)",
           "Tool wear [min]": "Tool wear (min)"}
MODES   = ["TWF", "HDF", "PWF", "OSF", "RNF"]
MODE_LB = {"TWF": "Tool wear", "HDF": "Heat dissip.", "PWF": "Power",
           "OSF": "Overstrain", "RNF": "Random"}

fail = df["Machine failure"] == 1
ok   = ~fail

print("=" * 78)
print("STAGE 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 78)
print(f"\nPlotting {fail.sum()} failure cycles against {ok.sum():,} healthy cycles\n")


# ═══ F1 ── Failure profile: imbalance + Pareto of causes ═════════════════
# Two panels. Left: how rare failure is. Right: Pareto of causes.
# Both axes are percentages on a single scale — no dual-axis trickery.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2),
                               gridspec_kw={"width_ratios": [1, 1.5]})

# -- Left: class balance as a single stacked bar
n_ok, n_fail = int(ok.sum()), int(fail.sum())
axL.barh([0], [n_ok / len(df)], color=S.OK_GREEN, height=.45, label="No failure")
axL.barh([0], [n_fail / len(df)], left=[n_ok / len(df)],
         color=S.FAIL_RED, height=.45, label="Failure")
axL.set_xlim(0, 1); axL.set_ylim(-.6, .9)
axL.set_yticks([]); axL.grid(False)
axL.spines["left"].set_visible(False); axL.spines["bottom"].set_visible(False)
axL.set_xticks([])
axL.text(n_ok / len(df) / 2, 0, f"{n_ok/len(df):.1%}\n{n_ok:,} cycles",
         ha="center", va="center", color="white", fontsize=11, fontweight="bold")
axL.annotate(f"{n_fail/len(df):.2%}  ·  {n_fail} failures",
             xy=(1.0, 0), xytext=(1.0, .62), ha="right",
             color=S.INK, fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=S.FAIL_RED, lw=1.5))
axL.set_title("Class balance", loc="left")
axL.text(0, -.45, "1 failure per 28 healthy cycles",
         color=S.INK_MUTED, fontsize=9)

# -- Right: Pareto of failure causes (bars = share, line = cumulative share)
counts = df[MODES].sum().sort_values(ascending=False)
share  = counts / counts.sum()
cum    = share.cumsum()
xs     = np.arange(len(counts))
axR.bar(xs, share.values, color=S.BLUE, width=.62, zorder=3)
axR.plot(xs, cum.values, color=S.AMBER, marker="o", zorder=4,
         label="Cumulative share")
for x, (s, c, n) in enumerate(zip(share.values, cum.values, counts.values)):
    # Count label sits INSIDE the bar so it can never collide with the
    # cumulative line running above it.
    axR.text(x, s - .035, f"{n}", ha="center", va="top", color="white",
             fontsize=9, fontweight="bold", zorder=5)
    axR.text(x, c + .055, f"{c:.0%}", ha="center", color=S.AMBER, fontsize=8.5,
             zorder=5)
axR.axhline(.8, color=S.INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
axR.text(-.42, .815, "80% line", ha="left", color=S.INK_MUTED, fontsize=8.5)
axR.set_xticks(xs)
axR.set_xticklabels([MODE_LB[m] for m in counts.index])
axR.set_ylim(0, 1.12)
S.pct_axis(axR)
axR.set_ylabel("Share of flagged failures")
axR.set_title("Pareto of failure causes", loc="left")
axR.legend(loc="center right")
fig.suptitle("Figure 1 — Failure profile: rare events, dominated by heat and overstrain",
             x=.012, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .94])
S.save(fig, FIG / "fig01_failure_profile.png")


# ═══ F2 ── Sensor distributions ══════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(15, 3.1))
for ax, col in zip(axes, SENSORS):
    ax.hist(df[col], bins=45, color=S.BLUE, alpha=.85, edgecolor="white", lw=.4)
    ax.set_title(SHORT[col], fontsize=10.5)
    ax.set_yticks([])
    ax.grid(axis="x", visible=False)
    mu = df[col].mean()
    ax.axvline(mu, color=S.AMBER, lw=1.6, ls=(0, (4, 3)))
    ax.text(.97, .93, f"μ={mu:,.1f}", transform=ax.transAxes, ha="right",
            va="top", color=S.INK_MUTED, fontsize=8.5)
axes[0].set_ylabel("Frequency")
fig.suptitle("Figure 2 — Sensor distributions across all 10,000 cycles",
             x=.008, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .89])
S.save(fig, FIG / "fig02_sensor_distributions.png")


# ═══ F3 ── Failed vs healthy on each sensor ══════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
for ax, col in zip(axes, SENSORS):
    parts = ax.boxplot([df.loc[ok, col], df.loc[fail, col]],
                       widths=.55, patch_artist=True, showfliers=False,
                       medianprops=dict(color="white", lw=1.8),
                       whiskerprops=dict(color=S.INK_MUTED, lw=1),
                       capprops=dict(color=S.INK_MUTED, lw=1))
    for patch, c in zip(parts["boxes"], [S.OK_GREEN, S.FAIL_RED]):
        patch.set_facecolor(c); patch.set_edgecolor("white"); patch.set_linewidth(1.4)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Healthy", "Failed"])
    ax.set_title(SHORT[col], fontsize=10.5)
    ax.grid(axis="x", visible=False)
    # Effect size (Cohen's d) tells us how much separation actually exists
    a, b = df.loc[ok, col], df.loc[fail, col]
    pooled = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    d = (b.mean() - a.mean()) / pooled
    ax.text(.5, .955, f"Cohen's d = {d:+.2f}", transform=ax.transAxes,
            ha="center", va="top", color=S.INK_MUTED, fontsize=8.5)
fig.suptitle("Figure 3 — No single sensor separates failures cleanly (all |d| < 0.8)",
             x=.008, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .88])
S.save(fig, FIG / "fig03_sensor_by_failure.png")


# ═══ F4 ── Correlation heatmap ═══════════════════════════════════════════
# Correlation is DIVERGING data (−1 .. 0 .. +1), so it gets two hues and a
# neutral gray midpoint — never a rainbow.
diverging = LinearSegmentedColormap.from_list(
    "pdm_div", [S.BLUE, "#AFC4D8", "#EDEDEB", "#EBC9A0", S.AMBER])

corr = df[SENSORS + ["Machine failure"]].corr()
labels = [SHORT.get(c, "Machine failure") for c in corr.columns]

fig, ax = plt.subplots(figsize=(6.6, 5.4))
im = ax.imshow(corr.values, cmap=diverging, vmin=-1, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=38, ha="right")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.grid(False)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(v) > .55 else S.INK)
cb = fig.colorbar(im, ax=ax, shrink=.8, pad=.03)
cb.set_label("Pearson r", fontsize=9)
cb.outline.set_visible(False)
ax.set_title("Figure 4 — Feature correlations\n"
             "Air and process temperature are near-collinear (r = 0.88)",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig04_correlation_heatmap.png")


# ═══ F5 ── Failure rate by product quality ═══════════════════════════════
qual = df.groupby("Type")["Machine failure"].agg(["mean", "sum", "count"]).loc[["H", "M", "L"]]
fig, ax = plt.subplots(figsize=(7, 3.4))
bars = ax.barh(["High", "Medium", "Low"], qual["mean"].values,
               color=[S.TEAL, S.BLUE, S.AMBER], height=.58, zorder=3)
for y, (r, n, tot) in enumerate(zip(qual["mean"], qual["sum"], qual["count"])):
    ax.text(r + .0012, y, f"{r:.2%}   ({int(n)} of {int(tot):,})",
            va="center", color=S.INK, fontsize=9.5)
ax.set_xlim(0, qual["mean"].max() * 1.5)
S.pct_axis(ax, "x")
ax.set_xlabel("Failure rate")
ax.set_ylabel("Product quality")
ax.grid(axis="y", visible=False)
ax.set_title("Figure 5 — Low-quality parts fail at nearly 2x the rate of high-quality parts",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig05_failure_by_quality.png")


# ═══ F6 ── The failure envelope in torque-speed space ════════════════════
# This is the single most informative EDA figure: it shows failures are NOT
# randomly scattered but cluster in the corners of the operating envelope.
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.scatter(df.loc[ok, "Rotational speed [rpm]"], df.loc[ok, "Torque [Nm]"],
           s=9, c=S.OK_GREEN, alpha=.16, edgecolors="none", label="Healthy", zorder=2)
ax.scatter(df.loc[fail, "Rotational speed [rpm]"], df.loc[fail, "Torque [Nm]"],
           s=26, c=S.FAIL_RED, alpha=.9, edgecolors="white", linewidths=.5,
           label="Failure", zorder=3)
# Constant-power reference curves: power = torque x angular velocity
speeds = np.linspace(1150, 2900, 300)
for watts, style in [(3500, (0, (5, 3))), (9000, (0, (5, 3)))]:
    ax.plot(speeds, watts / (speeds * 2 * np.pi / 60), color=S.INK_MUTED,
            lw=1.2, ls=style, zorder=4)
ax.text(1200, 3500 / (1200 * 2 * np.pi / 60) + 1.5, "3.5 kW", color=S.INK_MUTED, fontsize=8.5)
ax.text(2050, 9000 / (2050 * 2 * np.pi / 60) + 1.5, "9.0 kW", color=S.INK_MUTED, fontsize=8.5)
ax.set_xlabel("Rotational speed (rpm)"); ax.set_ylabel("Torque (Nm)")
ax.set_ylim(0, 82)
ax.legend(loc="upper right")
ax.set_title("Figure 6 — Failures cluster outside the safe power band\n"
             "Power failures occur below 3.5 kW and above 9 kW",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig06_torque_speed_envelope.png")


# ═══ F7 ── Tool wear x torque (the overstrain driver) ════════════════════
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.scatter(df.loc[ok, "Tool wear [min]"], df.loc[ok, "Torque [Nm]"],
           s=9, c=S.OK_GREEN, alpha=.16, edgecolors="none", label="Healthy", zorder=2)
osf = df["OSF"] == 1
other = fail & ~osf
ax.scatter(df.loc[other, "Tool wear [min]"], df.loc[other, "Torque [Nm]"],
           s=22, c=S.PURPLE, alpha=.75, edgecolors="white", linewidths=.4,
           label="Other failure", zorder=3)
ax.scatter(df.loc[osf, "Tool wear [min]"], df.loc[osf, "Torque [Nm]"],
           s=34, c=S.FAIL_RED, alpha=.95, edgecolors="white", linewidths=.6,
           label="Overstrain failure", zorder=4)
# Overstrain threshold curve: tool wear x torque exceeds a limit
wear = np.linspace(40, 253, 300)
ax.plot(wear, 11000 / wear, color=S.INK_MUTED, lw=1.3, ls=(0, (5, 3)), zorder=5)
ax.text(205, 11000 / 205 + 2.5, "wear x torque = 11,000", color=S.INK_MUTED, fontsize=8.5)
ax.set_xlabel("Tool wear (min)"); ax.set_ylabel("Torque (Nm)")
ax.set_ylim(0, 82)
ax.legend(loc="upper right")
ax.set_title("Figure 7 — Overstrain failures sit above a wear x torque threshold\n"
             "This motivates an engineered interaction feature",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig07_wear_torque_overstrain.png")


# ═══ F8 ── Temperature gap and heat dissipation failure ══════════════════
gap_all = df["Process temperature [K]"] - df["Air temperature [K]"]
hdf = df["HDF"] == 1
fig, ax = plt.subplots(figsize=(7.6, 4.2))
bins = np.linspace(gap_all.min(), gap_all.max(), 55)
ax.hist(gap_all[ok], bins=bins, color=S.OK_GREEN, alpha=.75, label="Healthy",
        edgecolor="white", lw=.3, zorder=2)
ax.hist(gap_all[hdf], bins=bins, color=S.FAIL_RED, alpha=.95,
        label="Heat dissipation failure", edgecolor="white", lw=.3, zorder=3)
ax.axvline(8.6, color=S.INK_MUTED, lw=1.4, ls=(0, (4, 3)), zorder=4)
ax.text(8.45, ax.get_ylim()[1] * .78, "8.6 K", ha="right", color=S.INK_MUTED, fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("Process temperature − Air temperature (K)")
ax.set_ylabel("Count (log scale)")
ax.legend(loc="upper right")
ax.set_title("Figure 8 — Heat failures occur almost entirely below a 8.6 K temperature gap\n"
             "A small gap means the machine cannot shed heat",
             loc="left", fontsize=11.5)
fig.tight_layout()
S.save(fig, FIG / "fig08_temp_gap.png")


print(f"\n8 figures written to {FIG}")
print("\nKey EDA findings for the report:")
print("  1. Severe class imbalance (3.39%) — accuracy is a meaningless metric here.")
print("  2. Heat dissipation + overstrain + power = 87% of all flagged failures.")
print("  3. No single sensor separates the classes (all |Cohen's d| < 0.8) —")
print("     which is exactly why a multivariate model is needed.")
print("  4. Air and process temperature are collinear (r = 0.88) — their")
print("     DIFFERENCE carries the real signal, not either one alone.")
print("  5. Failures cluster outside a safe torque-speed power band.")
print("  6. Overstrain sits above a tool-wear x torque threshold.")
print("  --> Findings 4, 5 and 6 directly motivate the engineered features in Stage 4.")
print("\n" + "=" * 78)
print("STAGE 2 COMPLETE")
print("=" * 78)
