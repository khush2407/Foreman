"""
reliability_core.py — reliability engineering on the AI4I 2020 data
====================================================================
Shared by the pipeline (09_reliability.py) and the Foreman app.

WHY THIS IS POSSIBLE
  The rows are in production order. Tool wear climbs 2, 3 or 5 minutes per
  cycle (by product grade) and drops back to 0 when the tool is changed.
  Splitting the data at those resets gives 120 individual tool lifetimes:
  46 end in a tool wear failure (TWF on the last cycle), 73 end in a
  planned change before failing, and the last one is still in service when
  the data stops. That is classic right-censored life data.

WHAT IT COMPUTES
  · tool life: Kaplan-Meier survival and a censored Weibull fit
  · B10 life, mean life, hazard, conditional survival
  · the cost-optimal age to replace the tool (age replacement policy)
  · failure rate and MTBF per failure mode / component
  · a series reliability block diagram of the four components
  · which failure modes occur together and which readings they share
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma

MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
COMPONENT = {
    "TWF": "Cutting tool",
    "OSF": "Spindle & bearings",
    "HDF": "Motor housing & cooling",
    "PWF": "Drive system",
    "RNF": "Other / random",
}
MODE_LABEL = {
    "TWF": "Tool wear failure", "HDF": "Heat dissipation failure",
    "PWF": "Power failure", "OSF": "Overstrain failure", "RNF": "Random failure",
}
# Readings each failure rule is written in (Matzka, 2020 documentation).
DRIVERS = {
    "TWF": ["Tool wear"],
    "OSF": ["Tool wear", "Torque", "Product quality"],
    "HDF": ["Air temperature", "Process temperature", "Rotational speed"],
    "PWF": ["Torque", "Rotational speed"],
    "RNF": [],
}


# ── tool lifetimes ────────────────────────────────────────────────────────
def tool_lives(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tool: life in wear-minutes, whether it ended in failure."""
    w = df["Tool wear [min]"].to_numpy()
    tool = np.concatenate([[0], np.cumsum(np.diff(w) < 0)])
    d = df.assign(tool=tool)
    g = d.groupby("tool").agg(
        life=("Tool wear [min]", "max"),
        cycles=("UDI", "size"),
        failed=("TWF", "last"),
        machine_failures=("Machine failure", "sum"),
    ).reset_index()
    g["in_service"] = False
    g.loc[g.index[-1], "in_service"] = True      # data ends mid-life
    g.loc[g["in_service"], "failed"] = 0
    g["outcome"] = np.where(g["failed"] == 1, "failed",
                            np.where(g["in_service"], "still running", "planned change"))
    return g


def kaplan_meier(t, e):
    t = np.asarray(t, float); e = np.asarray(e, int)
    order = np.argsort(t); t, e = t[order], e[order]
    times, surv, s = [0.0], [1.0], 1.0
    for u in np.unique(t[e == 1]):
        at_risk = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        s *= 1 - d / at_risk
        times.append(u); surv.append(s)
    return np.array(times), np.array(surv)


def weibull_fit(t, e):
    """Two-parameter Weibull by maximum likelihood with right censoring."""
    t = np.asarray(t, float); e = np.asarray(e, int)

    def nll(p):
        b, n = np.exp(p)
        z = (t / n) ** b
        ll = e * (np.log(b) - np.log(n) + (b - 1) * np.log(t / n)) - z
        return -ll.sum()

    res = minimize(nll, x0=[np.log(5.0), np.log(t.mean())], method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    beta, eta = np.exp(res.x)
    return float(beta), float(eta)


def R(t, beta, eta):
    return np.exp(-(np.asarray(t, float) / eta) ** beta)


def hazard(t, beta, eta):
    t = np.asarray(t, float)
    return beta / eta * (t / eta) ** (beta - 1)


def b_life(p, beta, eta):
    """Age by which a fraction p of tools have failed (B10 for p=0.10)."""
    return eta * (-math.log(1 - p)) ** (1 / beta)


def mean_life(beta, eta):
    return eta * gamma(1 + 1 / beta)


def conditional_survival(age, extra, beta, eta):
    """P(tool survives another `extra` minutes | it has survived `age`)."""
    return float(R(age + extra, beta, eta) / R(age, beta, eta))


def age_replacement(beta, eta, cost_planned, cost_failure, grid=None):
    """Cost per minute of tool use if tools are replaced at age T (or at
    failure if earlier). Classic age-replacement model (Barlow & Proschan)."""
    if grid is None:
        grid = np.arange(60, 261, 1.0)
    out = []
    for T in grid:
        x = np.linspace(0, T, 400)
        expected_life = np.trapezoid(R(x, beta, eta), x)
        F = 1 - R(T, beta, eta)
        out.append((T, (cost_planned * (1 - F) + cost_failure * F) / expected_life, F))
    a = np.array(out)
    i = int(np.argmin(a[:, 1]))
    return {"grid": a[:, 0], "cost_rate": a[:, 1], "p_fail": a[:, 2],
            "T_opt": float(a[i, 0]), "cost_opt": float(a[i, 1]), "p_fail_opt": float(a[i, 2])}


# ── failure modes, rates and the block diagram ───────────────────────────
def mode_rates(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for m in MODES:
        k = int(df[m].sum())
        rows.append({"mode": m, "label": MODE_LABEL[m], "component": COMPONENT[m],
                     "failures": k, "per_1000_cycles": 1000 * k / n,
                     "mtbf_cycles": n / k if k else float("inf")})
    return pd.DataFrame(rows)


def block_diagram(rates: pd.DataFrame, mission_cycles: int = 100) -> dict:
    """Series system: any component failing stops the machine. Each block's
    reliability over the mission uses its observed constant failure rate."""
    blocks = []
    for _, r in rates.iterrows():
        lam = r["failures"] / 10000
        blocks.append({"mode": r["mode"], "component": r["component"],
                       "R": math.exp(-lam * mission_cycles), "lambda": lam})
    Rs = math.prod(b["R"] for b in blocks)
    weakest = min(blocks, key=lambda b: b["R"])
    return {"mission_cycles": mission_cycles, "blocks": blocks, "R_system": Rs,
            "weakest": weakest["component"]}


def co_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """P(column mode | row mode): how often one failure comes with another."""
    m = df[MODES].astype(int)
    counts = m.T @ m
    diag = np.diag(counts).astype(float)
    cond = counts.div(np.where(diag == 0, 1, diag), axis=0)
    return cond.round(3)


def failure_prob_by_wear(df: pd.DataFrame, width: int = 20) -> pd.DataFrame:
    bins = np.arange(0, 261, width)
    lab = pd.cut(df["Tool wear [min]"], bins, right=False)
    g = df.groupby(lab, observed=False)
    out = g[["Machine failure"] + MODES].mean()
    out["cycles"] = g.size()
    out.index = [int(i.left) for i in out.index]
    return out
