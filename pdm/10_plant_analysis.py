"""
STAGE 10 — ROOT CAUSE, DEMAND SCENARIOS, SPARE PARTS AND BOTTLENECKS
====================================================================
Completes the October 9 feedback from Dr. Elahi:
  · root cause analysis of every recorded failure, with a fishbone diagram
  · what happens to failures and cost when usage or demand changes, and a
    drift check that says when the model must be retrained
  · the critical spare parts (Pareto / ABC) with reorder points
  · a simulated production line that finds the bottleneck and tests backup
    plans

What is measured and what is assumed is stated in plant_core.py.

Outputs
  fig23_fishbone.png            fig24_root_cause_pareto.png
  fig25_demand_scenarios.png    fig26_drift_check.png
  fig27_spare_parts_abc.png     fig28_line_bottleneck.png
  fig29_backup_plans.png
  results/10_plant.json and CSV tables
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import pdm_style as S
import reliability_core as rc
import plant_core as pc

S.apply_style()
ROOT = Path(__file__).parent
FIG, RES = ROOT / "outputs" / "figures", ROOT / "outputs" / "results"
MODELS, PROC = ROOT / "outputs" / "models", ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("STAGE 10 — ROOT CAUSE, DEMAND, SPARE PARTS, BOTTLENECKS")
print("=" * 78)

df = pc.prepare(pd.read_csv(ROOT / "data" / "ai4i2020.csv"))
tools = rc.tool_lives(df)
BETA, ETA = rc.weibull_fit(tools["life"], tools["failed"])

# model used for alarms: the best model from stage 5
best = open(RES / "05_best_model.txt").read().splitlines()[0].strip()
model = joblib.load(MODELS / f"{best.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
scaler = joblib.load(PROC / "scaler.joblib")
FEATS = pd.read_csv(PROC / "feature_names.csv").iloc[:, 0].tolist()
train_raw = pd.read_csv(PROC / "X_train_raw.csv")


def engineer(X):
    X = X.copy()
    X["Power [W]"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["Temp_diff [K]"] = X["Process temperature [K]"] - X["Air temperature [K]"]
    X["Wear_x_Torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]
    X["Temp_power_ratio"] = X["Temp_diff [K]"] / (X["Power [W]"] / 1000)
    return X


def risk(X):
    return model.predict_proba(scaler.transform(engineer(X[pc.BASE])[FEATS]))[:, 1]


# ── 10.1 root cause of every recorded failure ────────────────────────────
par = pc.root_cause_pareto(df)
rct = pc.root_cause_table(df)
print("\n[10.1] Root cause of the 373 recorded failure flags (339 failed cycles)")
for _, r in par.iterrows():
    print(f"      {r['failures']:4d}  {100 * r['share']:5.1f}%  cum {100 * r['cumulative']:5.1f}%  [{r['kind']}]  {r['cause']}")
kind = rct.groupby("kind").size()
print("      By type:", ", ".join(f"{k} {v} ({100 * v / len(rct):.0f}%)" for k, v in kind.items()))
rate = df.groupby("Type")["Machine failure"].mean()
hot = df["Air temperature [K]"] >= df["Air temperature [K]"].quantile(.75)
hdf_hot, hdf_cool = df.loc[hot, "HDF"].mean(), df.loc[~hot, "HDF"].mean()
print(f"      Heat failures: {100 * hdf_hot:.2f}% of cycles in the warmest quarter vs {100 * hdf_cool:.2f}% otherwise")

# fishbone
fig, ax = plt.subplots(figsize=(12, 6.2)); ax.axis("off"); ax.grid(False)
ax.set_xlim(0, 12); ax.set_ylim(0, 6.2)
ax.annotate("", xy=(10.15, 3.1), xytext=(0.4, 3.1), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=S.INK))
ax.text(10.25, 3.1, "Machine\nfailure\n(3.39% of\ncycles)", va="center", fontsize=11, weight="bold", color=S.FAIL_RED)
n = lambda k: int(par.loc[par["cause"] == k, "failures"].sum())
bones = [
    ("Method (settings)", 1.6, True, [f"Speed under 1,380 rpm with small\ntemperature gap ({n('cooling')} heat failures)",
                                      f"Torque × speed above 9,000 W ({n('power_high')})",
                                      f"Very high speed, light load,\nunder 3,500 W ({n('power_low')})"]),
    ("Machine (components)", 4.6, True, [f"Worn tool under load,\nwear × torque over limit ({n('worn_tool')})",
                                         f"Tool kept past its replacement\nage of 200–240 min ({n('tool_life')})"]),
    ("Material (product grade)", 7.6, True, [f"L grade fails {100 * rate['L']:.2f}% vs\nH grade {100 * rate['H']:.2f}%",
                                             "Lower overstrain limit for L grade\n(11,000 vs 13,000 min·Nm)"]),
    ("Environment", 1.6, False, [f"Warmest quarter of air temps:\nheat failures {100 * hdf_hot:.2f}% vs {100 * hdf_cool:.2f}%",
                                 "Air and process temperature\nmove together (r = 0.88)"]),
    ("Measurement", 4.6, False, ["Torque and speed sensors set\nthe power reading (calibrate first)",
                                 "Wear counter must reset at\nevery tool change"]),
    ("People / planning", 7.6, False, ["Setpoints programmed outside\nthe safe power band",
                                       f"Random, no identifiable cause ({n('random')})"]),
]
for title, x, top, items in bones:
    y0, y1 = (5.6, 3.1) if top else (0.6, 3.1)
    ax.plot([x, x + 1.5], [y0, y1], color=S.BLUE, lw=1.8)
    ax.text(x, y0 + (0.18 if top else -0.25), title, ha="center", fontsize=10.5, weight="bold", color=S.BLUE)
    for j, it in enumerate(items):
        f = (j + 1) / (len(items) + 1)
        xb, yb = x + 1.5 * f, y0 + (y1 - y0) * f
        ax.plot([xb - 0.25, xb], [yb, yb], color=S.INK_MUTED, lw=1)
        ax.text(xb - 0.3, yb, it, ha="right", va="center", fontsize=8.3, color=S.INK)
ax.set_title("Cause-and-effect (fishbone) diagram built from the 339 recorded failures", pad=4)
S.save(fig, FIG / "fig23_fishbone.png")

fig, ax = plt.subplots(figsize=(9, 4.8))
labels = ["Slow spindle,\nsmall temp gap", "Worn tool\nunder load", "Load too high\n(>9,000 W)", "Tool past\nreplacement age",
          "Load too light\n(<3,500 W)", "Random", "Torque too high\nfor grade"]
lab = dict(zip(["cooling", "worn_tool", "power_high", "tool_life", "power_low", "random", "overload"], labels))
col = [S.BLUE if k == "Operating setting" else S.AMBER if k == "Component condition" else S.INK_MUTED for k in par["kind"]]
ax.bar(range(len(par)), par["failures"], color=col)
ax2 = ax.twinx(); ax2.plot(range(len(par)), 100 * par["cumulative"], "o-", color=S.FAIL_RED); ax2.set_ylim(0, 105)
ax2.set_ylabel("Cumulative share (%)"); ax2.grid(False); ax2.spines["right"].set_visible(True)
ax.set_xticks(range(len(par))); ax.set_xticklabels([lab[c] for c in par["cause"]], fontsize=8.5)
ax.set_ylabel("Failure flags"); ax.set_title("Root causes ranked: settings (blue) vs worn components (amber)")
S.save(fig, FIG / "fig24_root_cause_pareto.png")

# ── 10.2 demand and usage scenarios ──────────────────────────────────────
SCEN = [
    ("Today", {}),
    ("Load +10%", {"load_pct": 10}),
    ("Load +20%", {"load_pct": 20}),
    ("Speed +10%", {"speed_pct": 10}),
    ("More L grade\n(80/15/5)", {"mix": (80, 15, 5)}),
    ("Hotter shop\n(air +3 K, gap −1 K)", {"ambient_k": 3, "gap_k": -1}),
    ("Load +20% with\ntool change 179 min", {"load_pct": 20, "change_age": 179}),
]
rows = []
for name, kw in SCEN:
    Xs = pc.apply_scenario(df, **kw)
    s = pc.scenario_summary(Xs, risk(Xs), BETA, ETA)
    drift = pc.drift_report(train_raw, Xs)
    rows.append({"scenario": name.replace("\n", " "), **{k: v for k, v in s.items() if k != "by_mode"},
                 **{f"fail_{m}": v for m, v in s["by_mode"].items()}, "max_psi": drift["psi"].max(),
                 "verdict": pc.drift_verdict(drift).split(":")[0]})
sc = pd.DataFrame(rows)
sc.to_csv(RES / "10_demand_scenarios.csv", index=False)
print("\n[10.2] Demand / usage scenarios (10,000 cycles a year)")
for _, r in sc.iterrows():
    print(f"      {r['scenario']:34s} failures {r['failures']:6.0f}  recall {100 * r['recall']:5.1f}%  "
          f"predictive ${r['predictive_cost'] / 1e6:5.2f}M  vs reactive ${r['reactive_cost'] / 1e6:5.2f}M  drift {r['max_psi']:.2f} {r['verdict']}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(sc))
short = ["Today", "Load\n+10%", "Load\n+20%", "Speed\n+10%", "More\nL grade", "Hotter\nshop", "Load +20%\n+ tool\nchange"]
bottom = np.zeros(len(sc))
for m, c in [("HDF", S.FAIL_RED), ("PWF", S.BLUE), ("OSF", S.PURPLE), ("TWF", S.AMBER), ("RNF", S.INK_MUTED)]:
    axes[0].bar(x, sc[f"fail_{m}"], bottom=bottom, color=c, label=rc.MODE_LABEL[m]); bottom += sc[f"fail_{m}"].to_numpy()
axes[0].plot(x, sc["failures"], "D", color=S.INK, ms=5, label="Failed cycles (any mode)")
axes[0].set_xticks(x); axes[0].set_xticklabels(short, fontsize=8); axes[0].set_ylabel("Expected failures per year")
axes[0].set_title("Failures by mode (a cycle can fail in more than one)"); axes[0].legend(fontsize=7.5, ncol=2)
axes[1].bar(x - 0.2, sc["reactive_cost"] / 1e6, 0.4, color=S.FAIL_RED, label="Run to failure")
axes[1].bar(x + 0.2, sc["predictive_cost"] / 1e6, 0.4, color=S.TEAL, label="Predictive (threshold 0.42)")
for i, r in sc.iterrows():
    axes[1].text(i + 0.2, r["predictive_cost"] / 1e6 + 0.1, f"{100 * r['recall']:.0f}%", ha="center", fontsize=7.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(short, fontsize=8); axes[1].set_ylabel("Annual cost ($ million)")
axes[1].set_title("Annual cost; labels = share of failures still caught"); axes[1].legend(fontsize=8)
S.save(fig, FIG / "fig25_demand_scenarios.png")

dr = pc.drift_report(train_raw, pc.apply_scenario(df, load_pct=20, ambient_k=3, gap_k=-1))
fig, ax = plt.subplots(figsize=(8, 3.8))
names = ["Product grade", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
ax.barh(names, dr["psi"], color=[S.FAIL_RED if v >= .25 else S.AMBER if v >= .1 else S.TEAL for v in dr["psi"]])
ax.axvline(0.1, color=S.AMBER, ls="--", lw=1); ax.axvline(0.25, color=S.FAIL_RED, ls="--", lw=1)
ax.text(0.095, 5.75, "watch ", color=S.AMBER, fontsize=8.5, ha="right"); ax.text(0.26, 5.75, " retrain", color=S.FAIL_RED, fontsize=8.5)
ax.set_xlabel("Population stability index vs training data"); ax.invert_yaxis()
ax.set_title("Drift check for 'load +20% in a hotter shop': which readings moved")
S.save(fig, FIG / "fig26_drift_check.png")
print("      Drift check, load +20% in a hotter shop:", ", ".join(f"{n} {v:.2f}" for n, v in zip(names, dr["psi"])))

# ── 10.3 spare parts ─────────────────────────────────────────────────────
base = sc.iloc[0]
fails = {m: base[f"fail_{m}"] for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]}
mean_step = df["Type_encoded"].map(pc.WEAR_STEP).mean()
tool_changes = 10000 / (179 / mean_step)
sp = pc.spare_parts(fails, tool_changes)
sp.to_csv(RES / "10_spare_parts.csv", index=False)
print(f"\n[10.3] Spare parts (assumed prices), {tool_changes:.0f} planned tool changes a year")
for _, r in sp.iterrows():
    print(f"      {r['class']}  {r['part']:26s} {r['annual_qty']:6.1f}/yr  ${r['annual_value']:8,.0f}  "
          f"{100 * r['cumulative']:5.1f}%  reorder at {r['reorder_point']}")
nA = int((sp["class"] == "A").sum())
print(f"      Class A: {nA} of {len(sp)} parts ({100 * nA / len(sp):.0f}%) carry "
      f"{100 * sp.loc[sp['class'] == 'A', 'share'].sum():.0f}% of the spend")

fig, ax = plt.subplots(figsize=(10, 4.6))
cc = {"A": S.FAIL_RED, "B": S.AMBER, "C": S.TEAL}
ax.bar(range(len(sp)), sp["annual_value"] / 1000, color=[cc[c] for c in sp["class"]])
ax2 = ax.twinx(); ax2.plot(range(len(sp)), 100 * sp["cumulative"], "o-", color=S.INK); ax2.set_ylim(0, 105)
ax2.axhline(80, color=S.INK_MUTED, ls=":", lw=1); ax2.set_ylabel("Cumulative share of spend (%)"); ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.set_xticks(range(len(sp))); ax.set_xticklabels(sp["part"], fontsize=8, rotation=30, ha="right")
ax.set_ylabel("Expected annual spend ($ thousand)")
ax.set_title("Spare parts ranked by annual spend: class A (red), B (amber), C (teal)")
S.save(fig, FIG / "fig27_spare_parts_abc.png")

# ── 10.4 production line and bottleneck ──────────────────────────────────
stations = pc.station_profiles(df, BETA, ETA)
plans, detail = pc.line_plans(df, BETA, ETA)
plans.to_csv(RES / "10_line_plans.csv", index=False)
print(f"\n[10.4] Simulated 4-station line, 80 hours, {pc.PARTS_PER_RUN} parts per recorded cycle (assumption)")
for s in stations:
    print(f"      {s['name']:14s} torque {s['torque_range']:9s}  failure per recorded cycle {100 * s['p_fail_cycle']:5.2f}%")
st0 = detail["Run to failure (today)"]["stats"]
print(st0.round(3).to_string(index=False))
for _, r in plans.iterrows():
    print(f"      {r['plan']:34s} {r['parts_per_hour']:5.1f} parts/h ({100 * r['gain_vs_today']:+5.1f}%)  bottleneck {r['bottleneck']}")

fig, ax = plt.subplots(figsize=(9, 4.2))
left = np.zeros(len(st0))
for k, c, lab_ in [("busy", S.TEAL, "Working"), ("down", S.FAIL_RED, "Down for repair"),
                   ("starved", "#C9D3DB", "Starved (no input)"), ("blocked", S.AMBER, "Blocked (buffer full)")]:
    ax.barh(st0["station"], st0[k] * 100, left=left, color=c, label=lab_); left += st0[k].to_numpy() * 100
ax.invert_yaxis(); ax.set_xlabel("Share of time (%)"); ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax.set_title(f"Where time goes on the line today: bottleneck = {detail['Run to failure (today)']['bottleneck']}")
S.save(fig, FIG / "fig28_line_bottleneck.png")

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh(plans["plan"], plans["parts_per_hour"], xerr=plans["sd"], color=[S.FAIL_RED] + [S.TEAL] * (len(plans) - 1))
for i, r in plans.iterrows():
    ax.text(r["parts_per_hour"] + r["sd"] + 1.5, i, f"{r['parts_per_hour']:.1f}/h  ({100 * r['gain_vs_today']:+.0f}%)", va="center", fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, plans["parts_per_hour"].max() * 1.3); ax.set_xlabel("Line output (parts per hour)")
ax.set_title("Backup plans, each added on top of the previous one")
S.save(fig, FIG / "fig29_backup_plans.png")

out = {"root_cause": par.to_dict("records"), "by_kind": kind.to_dict(),
       "scenarios": sc.to_dict("records"), "drift_example": dr.to_dict("records"),
       "spare_parts": sp.to_dict("records"), "tool_changes_per_year": tool_changes,
       "stations": stations, "line_plans": plans.to_dict("records")}
(RES / "10_plant.json").write_text(json.dumps(out, indent=2, default=float))
print("\nSaved figures 23-29 and results/10_*.csv, 10_plant.json")
print("=" * 78)
print("STAGE 10 COMPLETE — 7 figures written")
print("=" * 78)
