"""
plant_core.py — root cause, demand scenarios, drift, spare parts and the
production-line simulation
=========================================================================
Shared by the pipeline (10_plant_analysis.py) and the Foreman app.

Added after the October 9 review with Dr. Elahi, who asked for:
  · root cause analysis (a setting, a temperature, a motor not used adaptively)
  · what happens when usage or demand changes, and whether predictions adapt
  · the critical ~20% of parts held as inventory
  · finding the bottleneck in the flow of material, with a backup plan

WHAT IS MEASURED AND WHAT IS ASSUMED
  Measured from the data: the failure rules (they reproduce the HDF, PWF and
  OSF labels on 99.9-100% of rows), the tool life model, the root cause of
  each of the 339 recorded failures, and the drift statistics.
  Assumed (stated, adjustable): part prices, lead times and usage per repair;
  the layout, cycle times and repair times of the simulated production line;
  annual demand. These are labelled as assumptions wherever they are shown.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

BASE = ["Type_encoded", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
OSF_LIMIT = {0: 11000, 1: 12000, 2: 13000}
WEAR_STEP = {0: 2.0, 1: 3.0, 2: 5.0}   # wear minutes added per cycle, by grade
RNF_P = 0.001
# In the data 12 of the 27 cycles with a gap of exactly 8.6 K (shown to one
# decimal) at low speed still fail by heat dissipation.
HDF_EDGE_P = 12 / 27


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Type_encoded" not in d and "Type" in d:
        d["Type_encoded"] = d["Type"].map({"L": 0, "M": 1, "H": 2})
    return d


# ── documented failure rules, as expected probabilities per cycle ────────
def failure_probabilities(df: pd.DataFrame, beta: float, eta: float) -> pd.DataFrame:
    """Probability that each failure mode occurs on each cycle, using the
    dataset's documented rules. HDF, PWF and OSF are deterministic (0 or 1);
    tool wear failure uses the Weibull tool life model; random failure 0.1%."""
    d = prepare(df)
    rpm = d["Rotational speed [rpm]"]
    power = d["Torque [Nm]"] * rpm * 2 * np.pi / 60
    gap = (d["Process temperature [K]"] - d["Air temperature [K]"]).round(1)
    slow = rpm < 1380
    hdf = np.where(slow & (gap < 8.6), 1.0, np.where(slow & np.isclose(gap, 8.6), HDF_EDGE_P, 0.0))
    pwf = ((power < 3500) | (power > 9000)).astype(float).to_numpy()
    lim = d["Type_encoded"].map(OSF_LIMIT)
    osf = (d["Tool wear [min]"] * d["Torque [Nm]"] > lim).astype(float).to_numpy()
    age = d["Tool wear [min]"].clip(lower=0.5).to_numpy(float)
    step = d["Type_encoded"].map(WEAR_STEP).to_numpy(float)
    R = lambda t: np.exp(-(t / eta) ** beta)
    twf = 1 - R(age) / np.maximum(R(np.maximum(age - step, 0)), 1e-300)
    rnf = np.full(len(d), RNF_P)
    out = pd.DataFrame({"TWF": twf, "HDF": hdf, "PWF": pwf, "OSF": osf, "RNF": rnf}, index=d.index)
    out["any"] = 1 - np.prod(1 - out[["TWF", "HDF", "PWF", "OSF", "RNF"]].to_numpy(), axis=1)
    return out


# ── root cause ────────────────────────────────────────────────────────────
CAUSES = {
    "power_high":  ("Method", "Operating setting", "Load too high for the speed: torque × speed above the drive's 9,000 W band"),
    "power_low":   ("Method", "Operating setting", "Load too light for the drive: power below 3,500 W at very high speed"),
    "cooling":     ("Method", "Operating setting", "Spindle run too slowly (under 1,380 rpm) while the temperature gap is small, so heat is not carried away"),
    "worn_tool":   ("Machine", "Component condition", "Worn tool under load: wear × torque past the grade's limit with the tool near end of life"),
    "overload":    ("Method", "Operating setting", "Torque set too high for the product grade: wear × torque past the limit on a tool that is not yet worn"),
    "tool_life":   ("Machine", "Component condition", "Tool kept past its replacement age (200–240 min window)"),
    "random":      ("Unknown", "No identifiable cause", "Random failure, independent of every reading"),
}


SHORT = {"power_high": "Load too high", "power_low": "Load too light",
         "cooling": "Spindle too slow for the heat", "worn_tool": "Worn tool under load",
         "overload": "Torque too high for grade", "tool_life": "Tool past replacement age",
         "random": "Random"}

# What should have adapted, for each cause. A setting that stayed fixed while
# conditions changed is fixed by changing the setting, not by a new part.
ADAPT = {
    "power_high": "Change the setting: ease torque or speed so power stays under 9,000 W. No part needs replacing for this.",
    "power_low":  "Change the setting: add load or lower the speed so power stays above 3,500 W.",
    "cooling":    "Change the setting: when the shop warms up and the temperature gap closes, run the spindle above 1,380 rpm, but keep power under 9,000 W.",
    "worn_tool":  "Replace the tool, then check the spindle bearings that carried the load.",
    "overload":   "Change the setting: lower the torque for this product grade.",
    "tool_life":  "Replace the tool, and plan tool changes before 200 min of wear.",
    "random":     "No setting to change. Repair, log it and keep running.",
}


def root_cause_of(readings: dict, mode: str) -> str:
    """Root cause key for one failure, from its mode and readings."""
    rpm, tq = readings["Rotational speed [rpm]"], readings["Torque [Nm]"]
    power = tq * rpm * 2 * np.pi / 60
    if mode == "PWF":
        return "power_high" if power > 9000 else "power_low"
    if mode == "HDF":
        return "cooling"
    if mode == "OSF":
        return "worn_tool" if readings["Tool wear [min]"] >= 180 else "overload"
    if mode == "TWF":
        return "tool_life"
    return "random"


def root_cause_table(df: pd.DataFrame) -> pd.DataFrame:
    """Root cause of every recorded failure mode in the data."""
    d = prepare(df)
    rows = []
    for m in ["HDF", "PWF", "OSF", "TWF", "RNF"]:
        for _, r in d[d[m] == 1].iterrows():
            key = root_cause_of(r[BASE].to_dict(), m)
            cat, kind, text = CAUSES[key]
            rows.append({"mode": m, "cause": key, "category": cat, "kind": kind, "text": text,
                         "grade": "LMH"[int(r["Type_encoded"])], "air": r["Air temperature [K]"]})
    return pd.DataFrame(rows)


def root_cause_pareto(df: pd.DataFrame) -> pd.DataFrame:
    t = root_cause_table(df)
    p = t.groupby(["cause"]).size().rename("failures").reset_index()
    p["text"] = p["cause"].map(lambda k: CAUSES[k][2])
    p["kind"] = p["cause"].map(lambda k: CAUSES[k][1])
    p = p.sort_values("failures", ascending=False).reset_index(drop=True)
    p["share"] = p["failures"] / p["failures"].sum()
    p["cumulative"] = p["share"].cumsum()
    return p


# ── demand / usage scenarios ──────────────────────────────────────────────
def apply_scenario(df: pd.DataFrame, load_pct=0.0, speed_pct=0.0, mix=None,
                   ambient_k=0.0, gap_k=0.0, change_age=None, seed=42) -> pd.DataFrame:
    """Return the cycles as they would look under a changed way of running
    the machine. load_pct / speed_pct scale torque and speed; mix sets the
    share of L/M/H product; ambient_k raises the air temperature and gap_k
    changes the process-to-air gap (negative = cooling keeps up less); change_age
    caps tool wear (tools changed at that age)."""
    d = prepare(df)[BASE].copy()
    rng = np.random.default_rng(seed)
    d["Torque [Nm]"] = (d["Torque [Nm]"] * (1 + load_pct / 100)).clip(lower=1)
    d["Rotational speed [rpm]"] = (d["Rotational speed [rpm]"] * (1 + speed_pct / 100)).clip(lower=100)
    if mix is not None:
        p = np.asarray(mix, float); p = p / p.sum()
        d["Type_encoded"] = rng.choice([0, 1, 2], size=len(d), p=p)
    d["Air temperature [K]"] += ambient_k
    d["Process temperature [K]"] += ambient_k + gap_k
    if change_age:
        d["Tool wear [min]"] = d["Tool wear [min]"] % change_age
    return d


def scenario_summary(df_s: pd.DataFrame, risk: np.ndarray, beta: float, eta: float,
                     threshold=0.42, cycles_per_year=10000,
                     c_miss=12000, c_catch=1500, c_false=400) -> dict:
    """Expected failures, alarms and annual cost under a scenario.
    Failures come from the documented rules; alarms from the trained model."""
    fp = failure_probabilities(df_s, beta, eta)
    p = fp["any"].to_numpy()
    alarm = np.asarray(risk) >= threshold
    scale = cycles_per_year / len(df_s)
    fails = p.sum() * scale
    caught = (p * alarm).sum() * scale
    missed = fails - caught
    false = ((1 - p) * alarm).sum() * scale
    reactive = fails * c_miss
    predictive = caught * c_catch + missed * c_miss + false * c_false
    by_mode = {m: float(fp[m].sum() * scale) for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]}
    return {"failures": fails, "caught": caught, "missed": missed, "false_alarms": false,
            "recall": caught / fails if fails else float("nan"),
            "reactive_cost": reactive, "predictive_cost": predictive,
            "saving": reactive - predictive, "by_mode": by_mode}


# ── drift: has the way the machine runs changed? ──────────────────────────
def psi(ref: np.ndarray, new: np.ndarray, bins=10) -> float:
    """Population stability index between training data and new data."""
    ref = np.asarray(ref, float); new = np.asarray(new, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(ref.min(), ref.max() + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(new, edges)[0] / max(len(new), 1)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def drift_report(ref: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in BASE:
        if c == "Type_encoded":
            a = ref[c].value_counts(normalize=True).reindex([0, 1, 2], fill_value=1e-4).clip(lower=1e-4)
            b = new[c].value_counts(normalize=True).reindex([0, 1, 2], fill_value=1e-4).clip(lower=1e-4)
            v = float(((b - a) * np.log(b / a)).sum())
        else:
            v = psi(ref[c], new[c])
        state = "stable" if v < 0.1 else "moderate shift" if v < 0.25 else "major shift"
        rows.append({"reading": c, "psi": v, "state": state})
    out = pd.DataFrame(rows)
    return out


def drift_verdict(rep: pd.DataFrame) -> str:
    if (rep["psi"] >= 0.25).any():
        return "Major shift: the machine is running differently from the training data. Retrain the model before trusting its alarms."
    if (rep["psi"] >= 0.1).any():
        return "Moderate shift: keep using the model but watch its alarms, and retrain at the next review."
    return "Stable: new data looks like the training data."


# ── spare parts: what to keep in stock ────────────────────────────────────
# Assumed catalog (USD, lead time in days) and expected quantity used per
# repair of each failure mode. Tool changes also consume inserts.
PARTS = [
    # part, unit cost, lead time, {mode: qty per event}, per planned tool change
    ("Cutting tool insert set",    85,  7, {"TWF": 1, "OSF": 1}, 1),
    ("Collet",                    120, 14, {"TWF": .3, "OSF": .5}, .05),
    ("Spindle bearing set",      1800, 30, {"OSF": .5, "HDF": .2}, 0),
    ("Coolant and air filters",    45,  7, {"HDF": 1}, 0),
    ("Coolant pump",              650, 21, {"HDF": .3}, 0),
    ("Temperature sensor",        210, 14, {"HDF": .2}, 0),
    ("Variable frequency drive", 2400, 45, {"PWF": .25}, 0),
    ("Drive belt and coupling",   160, 10, {"PWF": .6}, 0),
    ("Torque sensor / encoder",   480, 21, {"PWF": .3}, 0),
    ("Motor rewind",             3200, 30, {"PWF": .1}, 0),
]


def poisson_quantile(mu: float, q: float = 0.95) -> int:
    """Smallest k with P(N <= k) >= q for N ~ Poisson(mu)."""
    if mu <= 0:
        return 0
    k, p, term = 0, math.exp(-mu), math.exp(-mu)
    while p < q and k < 10000:
        k += 1; term *= mu / k; p += term
    return k


def spare_parts(failures_by_mode: dict, tool_changes_per_year: float, service=0.95) -> pd.DataFrame:
    rows = []
    for name, cost, lead, use, per_change in PARTS:
        demand = sum(failures_by_mode.get(m, 0) * q for m, q in use.items()) + per_change * tool_changes_per_year
        lt_demand = demand * lead / 365
        rop = poisson_quantile(lt_demand, service)
        rows.append({"part": name, "unit_cost": cost, "lead_days": lead, "annual_qty": demand,
                     "annual_value": demand * cost, "reorder_point": rop,
                     "safety_stock": max(rop - lt_demand, 0)})
    t = pd.DataFrame(rows).sort_values("annual_value", ascending=False).reset_index(drop=True)
    t["share"] = t["annual_value"] / t["annual_value"].sum()
    t["cumulative"] = t["share"].cumsum()
    prev = t["cumulative"].shift(fill_value=0)
    t["class"] = np.where(prev < 0.80, "A", np.where(prev < 0.95, "B", "C"))
    return t


# ── production line: where is the bottleneck? ─────────────────────────────
PARTS_PER_RUN = 25   # assumption: each recorded cycle = one production run of 25 parts


def station_profiles(df: pd.DataFrame, beta: float, eta: float, change_age=None) -> list[dict]:
    """Four simulated stations, each running one quarter of the real
    operating data split by torque, so each inherits a measured failure
    rate. Rates are per part, treating each recorded cycle as a run of
    PARTS_PER_RUN parts. change_age applies a tool-change policy."""
    d = prepare(df)
    if change_age:
        d = d.copy(); d["Tool wear [min]"] = d["Tool wear [min]"] % change_age
    fp = failure_probabilities(d, beta, eta)
    q1, q2, q3 = d["Torque [Nm]"].quantile([.25, .5, .75])
    tq = d["Torque [Nm]"]
    slices = [
        ("1 Roughing",    tq >= q3,                1.0),
        ("2 Semi-finish", (tq >= q2) & (tq < q3),  0.9),
        ("3 Finishing",   (tq >= q1) & (tq < q2),  0.8),
        ("4 Polishing",   tq < q1,                 0.9),
    ]
    out = []
    for name, mask, ct in slices:
        p = float(fp.loc[mask, "any"].mean())
        out.append({"name": name, "cycle_min": ct, "p_fail_cycle": p, "p_fail": p / PARTS_PER_RUN,
                    "torque_range": f"{tq[mask].min():.0f}-{tq[mask].max():.0f} Nm", "cycles_in_data": int(mask.sum())})
    return out


def line_plans(df: pd.DataFrame, beta: float, eta: float, recall=0.912, false_alarm_cycle=73 / 2000,
               reps=6, minutes=4800) -> tuple[pd.DataFrame, dict]:
    """Compare the line under run-to-failure and four backup plans."""
    base = station_profiles(df, beta, eta)
    fa = false_alarm_cycle / PARTS_PER_RUN
    r0 = simulate_line(base, minutes=minutes, reps=reps)
    neck = r0["bottleneck"]
    plans = [
        ("Run to failure (today)", dict(stations=base)),
        ("Predictive maintenance", dict(stations=base, recall=recall, false_alarm=fa)),
        ("+ change tools at 179 min", dict(stations=station_profiles(df, beta, eta, change_age=179), recall=recall, false_alarm=fa)),
        ("+ larger buffers (5 to 20)", dict(stations=station_profiles(df, beta, eta, change_age=179), recall=recall, false_alarm=fa, buffer=20)),
        (f"+ standby machine at {neck.split(' ', 1)[1]}", dict(stations=station_profiles(df, beta, eta, change_age=179), recall=recall, false_alarm=fa, buffer=20, standby=neck)),
    ]
    rows, detail = [], {}
    for name, kw in plans:
        r = r0 if name.startswith("Run to failure") else simulate_line(minutes=minutes, reps=reps, **kw)
        rows.append({"plan": name, "parts_per_hour": r["throughput"], "sd": r["throughput_sd"], "bottleneck": r["bottleneck"]})
        detail[name] = r
    t = pd.DataFrame(rows)
    t["gain_vs_today"] = t["parts_per_hour"] / t.loc[0, "parts_per_hour"] - 1
    return t, detail


def simulate_line(stations: list[dict], minutes=4800, buffer=5, recall=0.0, false_alarm=0.0,
                  repair_unplanned=120, repair_planned=30, inspect=15, standby=None,
                  reps=8, seed=7) -> dict:
    """Time-stepped simulation of a serial line with buffers between
    stations. Each finished cycle can end in a failure (station's p_fail).
    With predictive maintenance a share `recall` of failures is caught early
    and becomes a short planned stop; false alarms add short inspections.
    `standby` names a station that gets a second, parallel machine."""
    dt = 0.1
    steps = int(minutes / dt)
    n = len(stations)
    agg = {"throughput": [], "busy": np.zeros(n), "down": np.zeros(n), "starved": np.zeros(n), "blocked": np.zeros(n)}
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        machines = [2 if standby == s["name"] else 1 for s in stations]
        prog = [[0.0] * m for m in machines]       # progress on current part, per machine
        down = [[0.0] * m for m in machines]       # minutes of repair left
        holding = [[False] * m for m in machines]  # finished part waiting to move on
        working = [[False] * m for m in machines]
        buf = [0] * (n - 1)
        done = 0
        busy = np.zeros(n); dn = np.zeros(n); st = np.zeros(n); bl = np.zeros(n)
        for _ in range(steps):
            for i in range(n - 1, -1, -1):
                s = stations[i]
                for k in range(machines[i]):
                    w = 1 / machines[i]
                    if down[i][k] > 0:
                        down[i][k] -= dt; dn[i] += dt * w; continue
                    if holding[i][k]:
                        if i == n - 1:
                            done += 1; holding[i][k] = False
                        elif buf[i] < buffer:
                            buf[i] += 1; holding[i][k] = False
                        else:
                            bl[i] += dt * w; continue
                    if not working[i][k]:
                        if i == 0:
                            working[i][k] = True
                        elif buf[i - 1] > 0:
                            buf[i - 1] -= 1; working[i][k] = True
                        else:
                            st[i] += dt * w; continue
                    prog[i][k] += dt; busy[i] += dt * w
                    if prog[i][k] >= s["cycle_min"] - 1e-9:
                        prog[i][k] = 0.0; working[i][k] = False; holding[i][k] = True
                        u = rng.random()
                        if u < s["p_fail"]:
                            down[i][k] = repair_planned if rng.random() < recall else repair_unplanned
                        elif rng.random() < false_alarm:
                            down[i][k] = inspect
        agg["throughput"].append(done / (minutes / 60))
        agg["busy"] += busy / minutes; agg["down"] += dn / minutes
        agg["starved"] += st / minutes; agg["blocked"] += bl / minutes
    for k in ("busy", "down", "starved", "blocked"):
        agg[k] = agg[k] / reps
    tp = np.array(agg["throughput"])
    stats = pd.DataFrame({"station": [s["name"] for s in stations], "busy": agg["busy"], "down": agg["down"],
                          "starved": agg["starved"], "blocked": agg["blocked"]})
    # Active-period method: the station that is active (working or under
    # repair) for the largest share of time is the bottleneck.
    stats["active"] = stats["busy"] + stats["down"]
    return {"throughput": float(tp.mean()), "throughput_sd": float(tp.std()), "stats": stats,
            "bottleneck": stats.loc[stats["active"].idxmax(), "station"]}
