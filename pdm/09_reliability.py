"""
STAGE 9 — RELIABILITY ANALYSIS
==============================
Added after the October 9 review with Dr. Elahi, who asked for:
  · product lifetime in use, to express failure rate as a probability
  · a reliability block diagram
  · the dependency between failures of different machine parts

The dataset rows are in production order, and tool wear resets to zero at
every tool change, so the data contains 120 real tool lifetimes. That turns
the tool into a proper life-data problem: some tools fail, some are changed
before failing (right-censored), and the last one is still running.

Outputs
  fig18_tool_life.png             Kaplan-Meier survival + fitted Weibull
  fig19_failure_rate_by_age.png   chance of failure per cycle as the tool ages
  fig20_tool_replacement_age.png  cost-optimal age to change the tool
  fig21_reliability_block.png     series block diagram of the four components
  fig22_failure_dependency.png    which failures share causes and occur together
  results/09_reliability.json, 09_reliability_summary.txt
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import pdm_style as S
import reliability_core as rc

S.apply_style()
ROOT = Path(__file__).parent
FIG = ROOT / "outputs" / "figures"
RES = ROOT / "outputs" / "results"
FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)

COST_PLANNED, COST_FAILURE = 1500, 12000   # same assumptions as stage 8
MISSION = 10                               # cycles, for the block diagram

df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")
print("=" * 76)
print("STAGE 9 — RELIABILITY ANALYSIS")
print("=" * 76)

# ── 9.1 tool lifetimes ────────────────────────────────────────────────────
tools = rc.tool_lives(df)
oc = tools["outcome"].value_counts()
print(f"\n[9.1] Tool lifetimes found by splitting at wear resets: {len(tools)}")
for k, v in oc.items():
    print(f"      {k:15s} {v}")
beta, eta = rc.weibull_fit(tools["life"], tools["failed"])
b10, mlife = rc.b_life(0.10, beta, eta), rc.mean_life(beta, eta)
print(f"      Weibull shape β = {beta:.2f} (β > 1 means wear-out: risk rises with age)")
print(f"      Weibull scale η = {eta:.1f} min")
print(f"      B10 life        = {b10:.0f} min  (10% of tools have failed by here)")
print(f"      Mean life       = {mlife:.0f} min")
for a in (180, 200, 220):
    print(f"      P(survive next 10 min | age {a}) = {rc.conditional_survival(a, 10, beta, eta):.3f}")

t_km, s_km = rc.kaplan_meier(tools["life"], tools["failed"])
x = np.linspace(150, 260, 400)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.step(t_km, s_km, where="post", color=S.BLUE, label="Observed (Kaplan-Meier)")
ax.plot(x, rc.R(x, beta, eta), color=S.AMBER, label=f"Weibull fit (β={beta:.1f}, η={eta:.0f} min)")
cens = tools[tools["failed"] == 0]
ax.plot(cens["life"], np.interp(cens["life"], t_km, s_km), "|", color=S.INK_MUTED, ms=9,
        label="Tool changed before failing (censored)")
ax.axvline(b10, color=S.FAIL_RED, lw=1, ls="--")
ax.text(b10 + 1.5, 0.04, f"B10 = {b10:.0f} min", ha="left", color=S.FAIL_RED, fontsize=9)
ax.set_xlim(150, 260); ax.set_ylim(0, 1.03)
ax.set_xlabel("Tool age (wear minutes)"); ax.set_ylabel("Probability the tool is still working")
ax.set_title("Tool life: 120 tools, 46 failures")
ax.legend(loc="upper right")
S.save(fig, FIG / "fig18_tool_life.png")

# ── 9.2 failure probability as the tool ages ─────────────────────────────
by = rc.failure_prob_by_wear(df, 20)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(by.index + 10, 100 * by["Machine failure"], "o-", color=S.INK, label="Any failure")
for m, col in [("TWF", S.AMBER), ("OSF", S.PURPLE), ("HDF", S.FAIL_RED), ("PWF", S.BLUE)]:
    ax.plot(by.index + 10, 100 * by[m], "o-", color=col, lw=1.4, ms=4, label=rc.MODE_LABEL[m])
ax.set_xlabel("Tool age (wear minutes, 20-minute bands)")
ax.set_ylabel("Chance of failure in one cycle (%)")
ax.set_title("Failure rate climbs sharply once the tool passes 200 minutes")
ax.legend(ncol=2)
S.save(fig, FIG / "fig19_failure_rate_by_age.png")
print("\n[9.2] Failure chance per cycle by tool age")
for a, row in by.iterrows():
    print(f"      {a:3d}-{a + 19:3d} min   {100 * row['Machine failure']:5.1f}%   ({int(row['cycles'])} cycles)")

# ── 9.3 when to change the tool ──────────────────────────────────────────
ar = rc.age_replacement(beta, eta, COST_PLANNED, COST_FAILURE)
rtf = rc.age_replacement(beta, eta, COST_PLANNED, COST_FAILURE, grid=np.array([400.0]))
observed_T = float(tools.loc[~tools["in_service"], "life"].median())
obs = rc.age_replacement(beta, eta, COST_PLANNED, COST_FAILURE, grid=np.array([observed_T]))
print(f"\n[9.3] Age replacement (planned change ${COST_PLANNED:,}, failure ${COST_FAILURE:,})")
print(f"      Cost-optimal change age  : {ar['T_opt']:.0f} min "
      f"(tools failing first: {100 * ar['p_fail_opt']:.1f}%)")
print(f"      Cost per tool-minute     : ${ar['cost_opt']:.2f} optimal, "
      f"${obs['cost_opt']:.2f} at the data's typical {observed_T:.0f} min, "
      f"${rtf['cost_opt']:.2f} run to failure")
sens = []
for cp in (250, 500, 1500, 3000):
    a = rc.age_replacement(beta, eta, cp, COST_FAILURE)
    sens.append((cp, a["T_opt"], a["p_fail_opt"]))
    print(f"      if a planned change costs ${cp:>5,}: change at {a['T_opt']:.0f} min")

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(ar["grid"], ar["cost_rate"], color=S.TEAL)
ax.axvline(ar["T_opt"], color=S.TEAL, ls="--", lw=1)
ax.scatter([ar["T_opt"]], [ar["cost_opt"]], color=S.TEAL, zorder=3)
ax.annotate(f"Change at {ar['T_opt']:.0f} min\n${ar['cost_opt']:.2f} per minute",
            (ar["T_opt"], ar["cost_opt"]), xytext=(ar["T_opt"] - 75, ar["cost_opt"] + 12),
            arrowprops=dict(arrowstyle="->", color=S.INK_MUTED), fontsize=9)
ax.axhline(rtf["cost_opt"], color=S.FAIL_RED, ls=":", lw=1)
ax.text(62, rtf["cost_opt"] + 1, f"Run to failure: ${rtf['cost_opt']:.2f} per minute", color=S.FAIL_RED, fontsize=9)
ax.set_ylim(0, max(rtf["cost_opt"] * 1.25, 30))
ax.set_xlabel("Planned tool change age (wear minutes)"); ax.set_ylabel("Expected cost per minute of tool use ($)")
ax.set_title("Best time to change the tool")
S.save(fig, FIG / "fig20_tool_replacement_age.png")

# ── 9.4 failure rates and the block diagram ──────────────────────────────
rates = rc.mode_rates(df)
bd = rc.block_diagram(rates, MISSION)
print(f"\n[9.4] Failure rate per mode (10,000 cycles)")
print(rates[["mode", "component", "failures", "per_1000_cycles", "mtbf_cycles"]]
      .round(1).to_string(index=False))
print(f"      Series system reliability over {MISSION} cycles: {bd['R_system']:.3f}")
print(f"      Weakest block: {bd['weakest']}")

fig, ax = plt.subplots(figsize=(10, 3.3))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.2); ax.axis("off"); ax.grid(False)
n = len(bd["blocks"])
xs = np.linspace(1.15, 9.45, n)
ax.annotate("", xy=(xs[0] - 0.85, 1.6), xytext=(0.05, 1.6), arrowprops=dict(arrowstyle="->", color=S.INK_MUTED))
ax.text(0.05, 1.8, "In", fontsize=9, color=S.INK_MUTED)
for i, (bk, xc) in enumerate(zip(bd["blocks"], xs)):
    weak = bk["component"] == bd["weakest"]
    ax.add_patch(FancyBboxPatch((xc - 0.78, 1.05), 1.56, 1.1, boxstyle="round,pad=0.02",
                                fc="#FDEDE1" if weak else "#EEF4FA", ec=S.FAIL_RED if weak else S.BLUE, lw=1.6))
    ax.text(xc, 1.83, bk["component"].replace(" & ", " &\n"), ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(xc, 1.25, f"R = {bk['R']:.3f}", ha="center", fontsize=8.5, color=S.INK_MUTED)
    ax.text(xc, 0.8, bk["mode"], ha="center", fontsize=8, color=S.INK_MUTED)
    if i < n - 1:
        ax.annotate("", xy=(xs[i + 1] - 0.8, 1.6), xytext=(xc + 0.8, 1.6), arrowprops=dict(arrowstyle="->", color=S.INK_MUTED))
ax.annotate("", xy=(10.55, 1.6), xytext=(xs[-1] + 0.8, 1.6), arrowprops=dict(arrowstyle="->", color=S.INK_MUTED))
ax.text(10.3, 1.8, "Out", fontsize=9, color=S.INK_MUTED)
ax.text(5.3, 2.75, f"Series system: the machine runs only if every block works.   "
        f"R(system, {MISSION} cycles) = {bd['R_system']:.3f}", ha="center", fontsize=10, weight="bold")
ax.text(5.3, 0.25, "Block reliability over the mission uses each mode's observed failure rate. "
        "Highlighted: the weakest block.", ha="center", fontsize=8.5, color=S.INK_MUTED)
S.save(fig, FIG / "fig21_reliability_block.png")

# ── 9.5 dependency map ───────────────────────────────────────────────────
co = rc.co_occurrence(df)
print("\n[9.5] P(column failure | row failure) — failures that occur together")
print(co.to_string())

readings = ["Product quality", "Air temperature", "Process temperature",
            "Rotational speed", "Torque", "Tool wear"]
modes4 = ["TWF", "OSF", "PWF", "HDF"]
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off"); ax.grid(False)
ry = dict(zip(readings, np.linspace(6.2, 0.8, len(readings))))
my = dict(zip(modes4, np.linspace(5.8, 1.2, 4)))
for rd, y in ry.items():
    ax.text(1.6, y, rd, ha="right", va="center", fontsize=9.5)
for m, y in my.items():
    ax.add_patch(FancyBboxPatch((3.95, y - 0.33), 2.3, 0.66, boxstyle="round,pad=0.02", fc="#EEF4FA", ec=S.BLUE))
    ax.text(5.1, y, rc.MODE_LABEL[m], ha="center", va="center", fontsize=9, weight="bold")
    ax.text(8.25, y, rc.COMPONENT[m], ha="left", va="center", fontsize=9.5)
    ax.annotate("", xy=(8.15, y), xytext=(7.55, y), arrowprops=dict(arrowstyle="->", color=S.INK_MUTED))
    for rd in rc.DRIVERS[m]:
        ax.plot([1.7, 3.93], [ry[rd], y], color=S.TEAL, lw=1.1, alpha=.8)
# co-occurrence arcs between modes
for a_, b_ in [("OSF", "PWF"), ("OSF", "HDF"), ("TWF", "OSF")]:
    p = max(co.loc[a_, b_], co.loc[b_, a_])
    ya, yb = my[a_], my[b_]
    ax.annotate("", xy=(6.3, yb), xytext=(6.3, ya),
                arrowprops=dict(arrowstyle="<->", color=S.FAIL_RED, connectionstyle="arc3,rad=-0.45", lw=1.2))
    ax.text(6.42 + 0.235 * abs(ya - yb), (ya + yb) / 2 + 0.12, f"{100 * p:.0f}%", color=S.FAIL_RED, fontsize=8.5, va="center")
ax.text(1.0, 6.85, "Readings", weight="bold", fontsize=10)
ax.text(4.5, 6.85, "Failure modes", weight="bold", fontsize=10)
ax.text(8.25, 6.85, "Component", weight="bold", fontsize=10)
ax.text(5.0, 0.05, "Teal lines: readings each failure rule depends on (shared readings couple the parts).  "
        "Red arcs: how often two failures occur on the same cycle.", ha="center", fontsize=8.5, color=S.INK_MUTED)
ax.set_title("How failures of different parts depend on each other")
S.save(fig, FIG / "fig22_failure_dependency.png")

# ── save results ─────────────────────────────────────────────────────────
out = {
    "tools": int(len(tools)), "tool_outcomes": {k: int(v) for k, v in oc.items()},
    "weibull_beta": beta, "weibull_eta": eta, "b10": b10, "mean_life": mlife,
    "replace_age_opt": ar["T_opt"], "cost_rate_opt": ar["cost_opt"],
    "cost_rate_observed": obs["cost_opt"], "observed_change_age": observed_T,
    "cost_rate_run_to_failure": rtf["cost_opt"],
    "replace_sensitivity": [{"cost_planned": c, "T_opt": t, "p_fail": p} for c, t, p in sens],
    "mission_cycles": MISSION, "R_system": bd["R_system"], "weakest": bd["weakest"],
    "blocks": bd["blocks"], "rates": rates.to_dict("records"),
    "co_occurrence": co.to_dict(),
    "fail_by_age": {int(k): float(v) for k, v in by["Machine failure"].items()},
}
(RES / "09_reliability.json").write_text(json.dumps(out, indent=2, default=float))
print("\nSaved figures 18-22 and results/09_reliability.json")
print("=" * 76)
print("STAGE 9 COMPLETE — 5 figures written")
print("=" * 76)
