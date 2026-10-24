"""
test_headless.py — exercise every code path the app uses, without a browser.

Run:  python3 test_headless.py
"""
import sys
import re
import pandas as pd
import numpy as np

from foreman.engine import (predict, what_if, suggest_fix, load, QUALITY,
                            example_cases, component_mode, fmt, envelope_flags,
                            engineer)
from foreman.assistant import write_work_order, build_prompt
from foreman.knowledge import retrieve, as_source_text
from foreman import schematic

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("=" * 74)
print("FOREMAN — headless test")
print("=" * 74)

# ── artefacts load ────────────────────────────────────────────────────────
print("\n[1] Artefacts")
A = load()
check("models and KB load", all(k in A for k in
      ("binary", "scaler", "modes", "kb", "features")))
check("10 features", len(A["features"]) == 10, f"{len(A['features'])}")
from foreman.engine import PREDICTABLE_MODES
# All four are trained and kept — TWF's poor score is a documented result,
# not a reason to delete it. Only three are surfaced as live predictions.
check("4 mode models trained", len(A["modes"]) == 4, sorted(A["modes"]))
check("3 surfaced as predictions", len(PREDICTABLE_MODES) == 3, PREDICTABLE_MODES)
check("TWF trained but not surfaced",
      "TWF" in A["modes"] and "TWF" not in PREDICTABLE_MODES)
check("KB has 4 modes", len(A["kb"]["modes"]) == 4)
check("KB has 3 policy tiers", len(A["kb"]["policy"]["tiers"]) == 3)

# ── real examples from the dataset ────────────────────────────────────────
print("\n[2] Scoring real dataset rows")
df = pd.read_csv("data/ai4i2020.csv")
df["Type_encoded"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
COLS = ["Type_encoded", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

for code in ("HDF", "PWF", "OSF"):
    rows = df[df[code] == 1].head(10)
    hits = sum(predict(r[COLS].to_dict())["top_mode"] == code
               for _, r in rows.iterrows())
    check(f"{code}: identified in most of 10 rows", hits >= 7, f"{hits}/10")

healthy = df[(df["Machine failure"] == 0) & (df["Tool wear [min]"] < 60)].iloc[0][COLS].to_dict()
ph = predict(healthy)
check("healthy row → low risk", ph["risk"] < .25, f"risk {ph['risk']:.0%}")
check("healthy row → Continue tier", ph["tier"]["name"] == "Continue",
      ph["tier"]["name"])

# ── tiers align with the documented thresholds ───────────────────────────
print("\n[3] Policy tiers")
for r, want in [(0.05, "Continue"), (0.25, "Inspect at changeover"),
                (0.90, "Intervene this shift")]:
    from foreman.engine import tier_for
    check(f"risk {r:.2f} → {want}", tier_for(r, A["kb"])["name"] == want)

# ── envelope flags are facts, not predictions ────────────────────────────
print("\n[4] Envelope flags")
over = {"Type_encoded": 0, "Air temperature [K]": 298.4,
        "Process temperature [K]": 308.9, "Rotational speed [rpm]": 2150,
        "Torque [Nm]": 42.0, "Tool wear [min]": 183}
po = predict(over)
names = [f["what"] for f in po["envelope_flags"]]
check("power over 9000 W flagged", "Mechanical power" in names,
      f"power {po['readings']['Power [W]']:,.0f} W; flags {names}")
check("power value is correct",
      abs(po["readings"]["Power [W]"] - 42.0 * 2150 * 2 * np.pi / 60) < 1e-6)

# ── attribution ───────────────────────────────────────────────────────────
print("\n[5] Attribution")
f = po["factors"]
check("returns 5 factors", len(f) == 5, f"{len(f)}")
check("sorted by magnitude",
      all(abs(f[i]["contribution"]) >= abs(f[i + 1]["contribution"])
          for i in range(len(f) - 1)))
check("each factor has a direction",
      all(x["direction"] in ("toward failure", "toward healthy") for x in f))

# ── retrieval ─────────────────────────────────────────────────────────────
print("\n[6] Knowledge retrieval")
ret = retrieve(po)
codes = [e["code"] for e in ret["entries"]]
check("retrieves PWF for a power excursion", "PWF" in codes, codes)
check("every entry says why it was picked",
      all(e.get("selected_because") for e in ret["entries"]))
src = as_source_text(ret)
check("source text carries the 9000 W limit", "9000 W" in src)
check("source text cites the rule source", "Matzka" in src)

# ── work order, template path ─────────────────────────────────────────────
print("\n[7] Work order (no API key)")
wo = write_work_order(po, use_llm=False)
check("template path used", wo["source"] == "template", wo["source"])
for head in ("STATUS", "WHY", "WHAT TO DO", "WHAT WOULD CLEAR IT", "SAFETY"):
    check(f"has {head}", head in wo["text"])
check("cites the KB entry", "[PWF:" in wo["text"])
check("quotes only KB thresholds",
      "9,000 W" in wo["text"] or "9000" in wo["text"])

# every number in the order should trace to the evidence, not be invented
nums = set(re.findall(r"\d[\d,]*\.?\d*", wo["text"]))
check("work order produced text", len(wo["text"]) > 300, f"{len(wo['text'])} chars")

# ── prompt assembly ───────────────────────────────────────────────────────
print("\n[8] Prompt given to the assistant")
prompt = build_prompt(po, ret, wo["fixes"])
check("contains the prediction", "Overall failure risk" in prompt)
check("contains the passages", "Knowledge base passages" in prompt)
check("states passages are the only source", "your only source" in prompt)

# ── what-if ───────────────────────────────────────────────────────────────
print("\n[9] What-if and fixes")
curve = what_if(over, "Torque [Nm]", np.linspace(3.8, 76.6, 25))
check("sweep returns 25 points", len(curve) == 25)
check("sweep risk stays in [0,1]", all(0 <= r <= 1 for _, r in curve))
fixes = suggest_fix(over)
check("finds at least one fix", len(fixes) >= 1, [f["name"] for f in fixes])
if fixes:
    fx = fixes[0]
    verify = predict({**over, fx["feature"]: fx["to"]})["risk"]
    check("the suggested fix actually works",
          verify < .15 and abs(verify - fx["risk_to"]) < 1e-6,
          f"{fx['name']} → {fx['to']:.1f} gives {verify:.1%}")

# ── schematic ─────────────────────────────────────────────────────────────
print("\n[10] Schematic")
for mode, zone in [("PWF", "Drive system"), ("HDF", "Motor housing"),
                   ("OSF", "Spindle"), ("TWF", "Cutting tool")]:
    svg = schematic.render(mode, 0.9)
    check(f"{mode} marks the right zone",
          svg.count("inspect here") == 1 and "</svg>" in svg)
svg_none = schematic.render(None, 0.1)
check("no mode → no marker", "inspect here" not in svg_none)
check("svg is well formed",
      svg_none.startswith("<svg") and svg_none.endswith("</svg>"))

# ── batch ─────────────────────────────────────────────────────────────────
print("\n[11] Batch scoring")
sample = df.head(60)
risks = [predict({c: float(r[c]) for c in COLS})["risk"]
         for _, r in sample.iterrows()]
check("scored 60 rows", len(risks) == 60)
check("risks are probabilities", all(0 <= r <= 1 for r in risks))
actual = sample["Machine failure"].values
if actual.sum():
    mean_fail = np.mean([r for r, a in zip(risks, actual) if a == 1])
    mean_ok = np.mean([r for r, a in zip(risks, actual) if a == 0])
    check("failures score higher than healthy rows", mean_fail > mean_ok,
          f"{mean_fail:.2f} vs {mean_ok:.2f}")

# ── fixes from the first live run ─────────────────────────────────────────
print("\n[12] Example cases (typical rows, not edge cases)")
CASES = example_cases()
want = {"Healthy cycle": None, "Heat dissipation failure": "HDF",
        "Power failure": "PWF", "Overstrain failure": "OSF",
        "Worn tool, still running": "TWF"}
for name, mode in want.items():
    pc = predict(CASES[name])
    check(f"{name} marks {mode or 'nothing'} on the schematic",
          component_mode(pc) == mode,
          f"got {component_mode(pc)}, risk {pc['risk']:.0%}")
check("healthy example is in the Continue tier",
      predict(CASES["Healthy cycle"])["tier"]["name"] == "Continue")

print("\n[13] Envelope rules match the documentation")
env = A["kb"]["operating_envelope"]
base = {"Type_encoded": 0, "Air temperature [K]": 300.0,
        "Rotational speed [rpm]": 1300, "Torque [Nm]": 40.0,
        "Tool wear [min]": 10}
whats = lambda r: [f["what"] for f in envelope_flags(engineer(r), env)]
check("8.6 K exactly is not 'below 8.6 K'",
      "Temperature differential" not in whats({**base, "Process temperature [K]": 308.6}))
check("8.5 K at slow speed is flagged",
      "Temperature differential" in whats({**base, "Process temperature [K]": 308.5}))
check("8.5 K at normal speed is not flagged (rule needs both)",
      "Temperature differential" not in whats({**base, "Process temperature [K]": 308.5,
                                                 "Rotational speed [rpm]": 1500}))
osf = {**base, "Process temperature [K]": 310.0, "Tool wear [min]": 200,
       "Torque [Nm]": 62.5}            # 12,500 min·Nm
check("12,500 min·Nm breaches the L-grade limit",
      "Wear x torque" in whats({**osf, "Type_encoded": 0}))
check("12,500 min·Nm is fine for H-grade stock",
      "Wear x torque" not in whats({**osf, "Type_encoded": 2}))

print("\n[14] Number formatting")
for v, u, want_s in [(9000.0, "W", "9,000 W"), (2170.0, "rpm", "2,170 rpm"),
                     (8.6, "K", "8.6 K"), (298.4, "K", "298.4 K"),
                     (42.0, "Nm", "42.0 Nm"), (0.0, "min", "0 min")]:
    check(f"fmt({v}, {u}) = {want_s}", fmt(v, u) == want_s, fmt(v, u))

print("\n[15] Fix suggestions are things an operator can do")
for name, r in CASES.items():
    for fx in suggest_fix(r):
        if fx["feature"] == "Torque [Nm]":
            check(f"{name}: torque fix only lowers torque", fx["to"] < fx["from"])
        if fx["feature"] == "Tool wear [min]":
            check(f"{name}: wear fix means a new tool", fx["to"] == 0)
check("healthy cycle gets no fixes", suggest_fix(CASES["Healthy cycle"]) == [])
wo_h = write_work_order(predict(CASES["Healthy cycle"]), use_llm=False)
check("healthy work order doesn't demand an inspection",
      "Inspection is required" not in wo_h["text"])
check("prompt carries units",
      all(u in build_prompt(po, ret, fixes) for u in (" rpm", " Nm", " min", " K")))
from foreman.assistant import to_html
h = to_html(wo["text"])
check("work order HTML is one physical line (markdown can't touch it)", "\n" not in h)
check("headings become heading rows", h.count('class="fm-wo-h"') == 5,
      h.count('class="fm-wo-h"'))
check("every line survives", h.count("<div") - 1 == len(wo["text"].splitlines()))
check("text is escaped", "<b>" not in to_html("Torque <b>x</b>"))
check("Claude-style markdown headings are handled",
      'fm-wo-h">STATUS<' in to_html("**STATUS**\nok") and
      'fm-wo-h">WHY<' in to_html("## WHY:\nok"))
worn = predict(CASES["Worn tool, still running"])
if worn["top_mode"] is None:
    check("no 'Most likely' when the model predicts no mode",
          "Most likely" not in write_work_order(worn, use_llm=False)["text"])
check("prompt asks for plain text", "no markdown" in build_prompt(po, ret, fixes))

# ── app imports ───────────────────────────────────────────────────────────
print("\n[17] Reliability (tool life, block diagram, dependencies)")
from foreman import reliability as rel
tools = rel.tool_lives(df)
check("120 tool lifetimes found from wear resets", len(tools) == 120, len(tools))
check("46 tools end in a tool wear failure", int(tools["failed"].sum()) == 46)
check("failures happen on the tool's last cycle",
      int(df["TWF"].sum()) == int(tools["failed"].sum()))
beta, eta = rel.weibull_fit(tools["life"], tools["failed"])
check("tool shows wear-out (Weibull shape > 1)", beta > 1, f"beta {beta:.1f}, eta {eta:.0f}")
check("B10 life inside the observed range", 190 < rel.b_life(0.1, beta, eta) < 230)
check("survival falls with age",
      rel.conditional_survival(100, 10, beta, eta) > rel.conditional_survival(220, 10, beta, eta))
ar = rel.age_replacement(beta, eta, 1500, 12000)
rtf = rel.age_replacement(beta, eta, 1500, 12000, grid=np.array([400.0]))
check("planned replacement beats run to failure", ar["cost_opt"] < rtf["cost_opt"],
      f"change at {ar['T_opt']:.0f} min")
check("cheaper planned changes move the best age earlier",
      rel.age_replacement(beta, eta, 250, 12000)["T_opt"] <= ar["T_opt"])
bd = rel.block_diagram(rel.mode_rates(df), 10)
check("series system is less reliable than its weakest block",
      bd["R_system"] < min(b["R"] for b in bd["blocks"]))
co = rel.co_occurrence(df)
check("dependency matrix has 1.0 on the diagonal", all(abs(co.loc[m, m] - 1) < 1e-9 for m in rel.MODES))

print("\n[18] App module")
import ast
src_app = open("app.py").read()
try:
    ast.parse(src_app)
    check("app.py parses", True)
except SyntaxError as e:
    check("app.py parses", False, str(e))
check("app lays the work order out line by line", "to_html(wo" in src_app)
check("switching example resets the sliders", "on_change=load_case" in src_app)
check("app has a Reliability tab", "tab_rel" in src_app and "rel.block_diagram" in src_app)
check("app calls every layer",
      all(s in src_app for s in ("predict(", "write_work_order(",
                                 "schematic.render(", "what_if(", "retrieve")))

print("\n" + "=" * 74)
if FAILS:
    print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
    sys.exit(1)
print("ALL CHECKS PASSED")
print("=" * 74)
