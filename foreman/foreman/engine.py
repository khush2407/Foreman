"""
engine.py — prediction and attribution
======================================
Loads the trained models and turns a set of sensor readings into:
  · overall failure risk
  · a probability for each failure mode
  · which readings drove the result, and in which direction
  · the urgency tier implied by the cost model

Nothing here calls a language model. This layer produces the evidence;
the assistant layer only ever describes what it is given.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
KB_PATH = ROOT / "kb" / "kb.json"

BASE_FEATURES = [
    "Type_encoded",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Readings a technician would recognise, in the order the UI shows them.
READING_LABELS = {
    "Type_encoded":            "Product quality",
    "Air temperature [K]":     "Air temperature",
    "Process temperature [K]": "Process temperature",
    "Rotational speed [rpm]":  "Rotational speed",
    "Torque [Nm]":             "Torque",
    "Tool wear [min]":         "Tool wear",
    "Power [W]":               "Mechanical power",
    "Temp_diff [K]":           "Temperature differential",
    "Wear_x_Torque":           "Wear x torque",
    "Temp_power_ratio":        "Thermal headroom ratio",
}

UNITS = {
    "Air temperature":        "K",
    "Process temperature":    "K",
    "Rotational speed":       "rpm",
    "Torque":                 "Nm",
    "Tool wear":              "min",
    "Mechanical power":       "W",
    "Temperature differential": "K",
    "Wear x torque":          "min·Nm",
    "Thermal headroom ratio": "",
    "Product quality":        "",
}

QUALITY = {0: "L (low)", 1: "M (medium)", 2: "H (high)"}

# TWF is excluded from the mode display as a *prediction* — see kb.json.
# Its classifier scores poorly because the dataset fires it at a random
# wear time, so the wear clock is reported instead.
PREDICTABLE_MODES = ["HDF", "PWF", "OSF"]

# Overstrain limit depends on product quality (Matzka, 2020). The data
# agrees: the smallest wear x torque among OSF rows is 11,003 (L),
# 12,337 (M) and 13,235 (H).
OSF_LIMIT = {0: 11000, 1: 12000, 2: 13000}

COLS = BASE_FEATURES


def fmt(v, unit: str = "") -> str:
    """One number format for the whole app, so the same reading never
    shows as 9,000.0 W in one panel and 9000 W in another.
      temperatures       one decimal  (300.1 K, 8.6 K)
      100 and above,     no decimals  (2,170 rpm, 9,045 W, 0 min)
      or counted units
      small, unitless    two decimals (1.38)
      everything else    one decimal  (42.5 Nm)"""
    if isinstance(v, str):
        return v
    v = float(v)
    if unit == "K":
        s = f"{v:,.1f}"
    elif abs(v) >= 100 or unit in ("rpm", "min", "W", "min·Nm"):
        s = f"{v:,.0f}"
    elif not unit and abs(v) < 10:
        s = f"{v:.2f}"
    else:
        s = f"{v:,.1f}"
    return f"{s} {unit}".strip()


def engineer(row: dict) -> dict:
    """Add the four derived features to a set of raw readings."""
    out = dict(row)
    omega = out["Rotational speed [rpm]"] * 2 * np.pi / 60
    out["Power [W]"]        = out["Torque [Nm]"] * omega
    out["Temp_diff [K]"]    = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Wear_x_Torque"]    = out["Tool wear [min]"] * out["Torque [Nm]"]
    out["Temp_power_ratio"] = out["Temp_diff [K]"] / (out["Power [W]"] / 1000) \
        if out["Power [W]"] else 0.0
    return out


@lru_cache(maxsize=1)
def load():
    """Load every artefact once and cache it."""
    features = pd.read_csv(MODELS / "feature_names.csv")["feature"].tolist()
    binary = joblib.load(next(p for p in [MODELS / "XGBoost.joblib", MODELS / "Gradient_Boosting.joblib"] if p.exists()))
    scaler = joblib.load(MODELS / "scaler.joblib")
    modes = joblib.load(MODELS / "mode_models.joblib")
    train = pd.read_csv(MODELS / "X_train_engineered.csv")
    kb = json.loads(KB_PATH.read_text())
    return {
        "features": features,
        "binary": binary,
        "scaler": scaler,
        "modes": modes["modes"],
        "mode_features": modes["features"],
        "medians": train[features].median().values,
        "train": train,
        "kb": kb,
    }


def _vector(row: dict, features: list[str]) -> pd.DataFrame:
    """One row as a DataFrame with the column names the scaler was fitted
    on. Passing a bare array works but makes sklearn warn on every call."""
    full = engineer(row)
    return pd.DataFrame([[full[f] for f in features]], columns=features, dtype=float)


def tier_for(risk: float, kb: dict) -> dict:
    for t in kb["policy"]["tiers"]:
        if risk < t["max_risk"]:
            return t
    return kb["policy"]["tiers"][-1]


def attribute(row: dict, top_n: int = 5) -> list[dict]:
    """Which readings pushed this prediction, and which way.

    Uses SHAP when it is installed. Otherwise falls back to local ablation:
    replace one feature at a time with the training median and measure how
    far the predicted risk moves. Both answer the same question; SHAP is
    exact, ablation is an approximation that needs no extra dependency.
    """
    A = load()
    X = _vector(row, A["features"])
    Xs = A["scaler"].transform(X)
    base_p = float(A["binary"].predict_proba(Xs)[0, 1])

    try:
        import shap
        expl = shap.TreeExplainer(A["binary"])
        vals = expl.shap_values(Xs)
        if isinstance(vals, list):
            vals = vals[1]
        contrib = np.ravel(vals)
        method = "shap"
    except Exception:
        med = A["scaler"].transform(
            pd.DataFrame([A["medians"]], columns=A["features"]))
        contrib = np.zeros(len(A["features"]))
        for i in range(len(A["features"])):
            probe = Xs.copy()
            probe[0, i] = med[0, i]
            contrib[i] = base_p - float(A["binary"].predict_proba(probe)[0, 1])
        method = "ablation"

    full = engineer(row)
    out = []
    for f, c in zip(A["features"], contrib):
        name = READING_LABELS.get(f, f)
        val = full[f]
        out.append({
            "feature": f,
            "name": name,
            "value": QUALITY.get(int(val), val) if f == "Type_encoded" else val,
            "unit": UNITS.get(name, ""),
            "contribution": float(c),
            "direction": "toward failure" if c > 0 else "toward healthy",
        })
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    for d in out:
        d["method"] = method
    return out[:top_n]


def predict(row: dict) -> dict:
    """Score one set of readings. Returns everything the UI and the
    assistant need, and nothing that required a language model."""
    A = load()
    full = engineer(row)

    Xs = A["scaler"].transform(_vector(row, A["features"]))
    risk = float(A["binary"].predict_proba(Xs)[0, 1])

    modes = {}
    for code in PREDICTABLE_MODES:
        m = A["modes"][code]
        xm = m["scaler"].transform(_vector(row, A["mode_features"]))
        modes[code] = float(m["model"].predict_proba(xm)[0, 1])

    top_mode = max(modes, key=modes.get) if modes else None
    top_p = modes.get(top_mode, 0.0)

    env = A["kb"]["operating_envelope"]
    wear = full["Tool wear [min]"]
    wear_state = ("replace" if wear >= env["tool_wear_min"]["replace"]
                  else "approaching" if wear >= env["tool_wear_min"]["warn"]
                  else "ok")

    tier = tier_for(risk, A["kb"])
    flags = envelope_flags(full, env)
    mode = top_mode if top_p >= .5 else None

    return {
        "readings": full,
        "risk": risk,
        "tier": tier,
        "modes": modes,
        "top_mode": mode,
        "top_mode_p": top_p,
        "wear_state": wear_state,
        "wear_minutes": wear,
        "factors": attribute(row),
        "envelope_flags": flags,
        "root_causes": root_cause(full, flags, mode),
    }


def root_cause_keys(full: dict, flags: list[dict], top_mode: str | None) -> list[str]:
    """Which root causes explain this cycle. Each broken limit points to
    one cause; with no limit broken, the predicted mode decides."""
    from .plant import root_cause_of
    keys = []
    for f in flags:
        w = f["what"]
        if w == "Mechanical power":
            keys.append("power_high" if f["state"] == "above" else "power_low")
        elif w == "Temperature differential":
            keys.append("cooling")
        elif w == "Wear x torque":
            keys.append(root_cause_of(full, "OSF"))
        elif w == "Tool wear":
            keys.append("tool_life")
    if not keys and top_mode:
        keys.append(root_cause_of(full, top_mode))
    return list(dict.fromkeys(keys))


def root_cause(full: dict, flags: list[dict], top_mode: str | None) -> list[dict]:
    """Root cause of the alarm, split into an operating setting that should
    have adapted or a component that has worn. Empty when nothing is wrong."""
    from .plant import CAUSES, ADAPT
    out = []
    for k in root_cause_keys(full, flags, top_mode):
        cat, kind, text = CAUSES[k]
        out.append({"key": k, "category": cat, "kind": kind, "text": text, "adapt": ADAPT[k]})
    return out


def score_many(df: pd.DataFrame) -> pd.DataFrame:
    """Score many cycles at once. Same model and same rules as predict(),
    but vectorised and without per-row attribution, so a full shift or the
    whole 10,000-cycle dataset scores in about a second."""
    A = load()
    env = A["kb"]["operating_envelope"]
    d = df[BASE_FEATURES].astype(float).reset_index(drop=True)
    d["Power [W]"] = d["Torque [Nm]"] * d["Rotational speed [rpm]"] * 2 * np.pi / 60
    d["Temp_diff [K]"] = d["Process temperature [K]"] - d["Air temperature [K]"]
    d["Wear_x_Torque"] = d["Tool wear [min]"] * d["Torque [Nm]"]
    d["Temp_power_ratio"] = np.where(d["Power [W]"] != 0,
                                     d["Temp_diff [K]"] / (d["Power [W]"] / 1000), 0.0)
    risk = A["binary"].predict_proba(A["scaler"].transform(d[A["features"]]))[:, 1]
    probs = {}
    for code in PREDICTABLE_MODES:
        m = A["modes"][code]
        probs[code] = m["model"].predict_proba(m["scaler"].transform(d[A["mode_features"]]))[:, 1]
    P = pd.DataFrame(probs)
    top = P.idxmax(axis=1).where(P.max(axis=1) >= .5)
    out = d.copy()
    out["risk"] = risk
    out["tier"] = [tier_for(r, A["kb"])["name"] for r in risk]
    out["top_mode"] = top
    causes = []
    for rec, tm in zip(d.to_dict("records"), top):
        flags = envelope_flags(rec, env)
        causes.append(", ".join(root_cause_keys(rec, flags, tm if isinstance(tm, str) else None)))
    out["root_cause"] = causes
    return out


def envelope_flags(full: dict, env: dict) -> list[dict]:
    """Which documented operating limits this cycle is outside of.

    These are hard rules from the dataset documentation, not model output,
    so they are reported as facts rather than predictions. Values are
    compared at the precision they are displayed at, so the app never says
    "8.6 K is below 8.6 K".
    """
    flags = []

    def add(what, value, unit, state, limit, note=""):
        text = (f"{what} at {fmt(value, unit)} is {state} the limit of "
                f"{fmt(limit, unit)}" + (f", {note}" if note else ""))
        flags.append({"what": what, "value": value, "unit": unit,
                      "state": state, "limit": limit, "note": note,
                      "text": text})

    p = round(full["Power [W]"])
    if p < env["power_w"]["min"]:
        add("Mechanical power", p, "W", "below", env["power_w"]["min"])
    elif p > env["power_w"]["max"]:
        add("Mechanical power", p, "W", "above", env["power_w"]["max"])

    # Heat dissipation needs BOTH conditions: a small temperature gap and
    # a slow spindle. A small gap at normal speed is not the failure rule.
    td = round(full["Temp_diff [K]"], 1)
    rpm = full["Rotational speed [rpm]"]
    slow = env.get("speed_rpm_heat_risk_below", 1380)
    if td < env["temp_diff_k"]["min"] and rpm < slow:
        add("Temperature differential", td, "K", "below",
            env["temp_diff_k"]["min"],
            f"with speed {fmt(rpm, 'rpm')} under {fmt(slow, 'rpm')}")

    q = int(full.get("Type_encoded", 0))
    lim = OSF_LIMIT.get(q, 11000)
    wt = round(full["Wear_x_Torque"])
    if wt > lim:
        add("Wear x torque", wt, "min·Nm", "above", lim,
            f"set for {QUALITY[q].split()[0]}-grade stock")

    w = full["Tool wear [min]"]
    if w >= env["tool_wear_min"]["replace"]:
        add("Tool wear", w, "min", "past", env["tool_wear_min"]["replace"],
            "inside the 200–240 min replacement window")
    return flags


def component_mode(p: dict) -> str | None:
    """Which component to mark on the schematic. The predicted mode wins;
    with none, a tool in its replacement window is still worth pointing at."""
    if p["top_mode"]:
        return p["top_mode"]
    return "TWF" if p["wear_state"] == "replace" else None


@lru_cache(maxsize=1)
def example_cases() -> dict:
    """Typical rows from the real data, one per situation.

    Each failure example is a row where only that one mode fired, chosen
    as the median case on the quantity its rule is written in. Taking the
    first matching row instead picks edge cases — the first HDF row has a
    temperature gap of exactly 8.6 K, right on the boundary.
    """
    df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")
    df["Type_encoded"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    full = pd.DataFrame([engineer(r) for r in df[COLS].to_dict("records")])
    full.index = df.index
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    only = lambda m: (df[m] == 1) & (df[modes].sum(axis=1) == 1)

    def median_row(mask, key):
        sub = full[mask]
        i = (sub[key] - sub[key].median()).abs().idxmin()
        return {c: float(df.loc[i, c]) for c in COLS}

    lim = full["Type_encoded"].map(OSF_LIMIT)
    full["_osf_ratio"] = full["Wear_x_Torque"] / lim
    healthy = ((df["Machine failure"] == 0) & (df["Tool wear [min]"] < 60) &
               (full["Temp_diff [K]"] > 9.5) &
               full["Power [W]"].between(4500, 7500))

    return {
        "Healthy cycle":            median_row(healthy, "Power [W]"),
        "Heat dissipation failure": median_row(only("HDF"), "Temp_diff [K]"),
        "Power failure":            median_row(only("PWF") & (full["Power [W]"] > 9000),
                                               "Power [W]"),
        "Overstrain failure":       median_row(only("OSF"), "_osf_ratio"),
        "Worn tool, still running": median_row(
            (df["Machine failure"] == 0) & df["Tool wear [min]"].between(205, 235),
            "Tool wear [min]"),
    }


def what_if(row: dict, feature: str, values) -> list[tuple[float, float]]:
    """Re-score with one raw reading swept across a range.

    This is what turns the tool from a diagnosis into a decision aid: it
    answers 'what would I have to change to clear this alarm?'.
    """
    out = []
    for v in values:
        probe = dict(row)
        probe[feature] = float(v)
        A = load()
        Xs = A["scaler"].transform(_vector(probe, A["features"]))
        out.append((float(v), float(A["binary"].predict_proba(Xs)[0, 1])))
    return out


def suggest_fix(row: dict, target: float = 0.15) -> list[dict]:
    """Find the smallest single change to each adjustable reading that
    would bring risk under the target. Returns only changes that work.

    Only moves an operator can actually make are searched:
      speed   up or down (raising it clears heat risk, lowering it clears
              high power)
      torque  down only — easing the load. A model can sometimes score a
              higher torque as safer, but nobody would act on that.
      wear    replace the tool, i.e. back to 0 min, and only once the tool
              is near its replacement window. Wear cannot be dialled back
              by ten minutes.
    """
    A = load()
    env = A["kb"]["operating_envelope"]
    base = predict(row)["risk"]
    if base < target:
        return []          # nothing to clear
    fixes = []

    def keep(feat, v, p, label=None):
        fixes.append({
            "feature": feat,
            "name": label or READING_LABELS[feat],
            "from": row[feat], "to": float(v),
            "unit": {"Rotational speed [rpm]": "rpm", "Torque [Nm]": "Nm",
                     "Tool wear [min]": "min"}[feat],
            "risk_from": base, "risk_to": p,
            "delta": float(v) - row[feat],
        })

    for feat, lo, hi in [("Rotational speed [rpm]", 1168, 2886),
                         ("Torque [Nm]", 3.8, row["Torque [Nm]"])]:
        cur = row[feat]
        grid = np.linspace(lo, hi, 90)
        ok = [(v, p) for v, p in what_if(row, feat, grid)
              if p < target and abs(v - cur) > 1e-9]
        if ok:
            v, p = min(ok, key=lambda t: abs(t[0] - cur))
            keep(feat, v, p)

    if row["Tool wear [min]"] >= env["tool_wear_min"]["warn"]:
        (v, p), = what_if(row, "Tool wear [min]", [0.0])
        if p < target:
            keep("Tool wear [min]", v, p, label="Replace the tool, tool wear")

    fixes.sort(key=lambda d: abs(d["delta"] / max(abs(d["from"]), 1)))
    return fixes
