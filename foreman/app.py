"""
Foreman — AI maintenance assistant
==================================
ISE 298 · Khush Hemant Patel · Supervisor: Dr. Behin Elahi

Run with:  streamlit run app.py
"""
import os
import sys
import html
from pathlib import Path

# Make "foreman" mean the package next to this file, even when the app is
# launched from the repository root (as Streamlit Community Cloud does),
# where the outer foreman/ folder would otherwise shadow it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st

from foreman.engine import (predict, what_if, load, example_cases, score_many,
                            component_mode, fmt, QUALITY, READING_LABELS)
from foreman.assistant import write_work_order, to_html
from foreman.knowledge import retrieve
from foreman import schematic
from foreman import reliability as rel
from foreman import plant

# Hosted on Streamlit Community Cloud the app starts from the repository
# root, not this folder. Move here so data/ and models/ resolve the same way
# locally and online.
os.chdir(Path(__file__).resolve().parent)

# On Streamlit Community Cloud the API key lives in the app's Secrets, not
# in the shell. Copy it into the environment so the rest of the code is the
# same locally and hosted.
try:
    for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "FOREMAN_MODEL",
               "APP_PASSWORD"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

st.set_page_config(page_title="Foreman", page_icon="🔧", layout="wide")

# ── optional access gate (hosted version) ────────────────────────────────
# Set APP_PASSWORD in the app's Secrets to stop strangers from spending the
# Claude credits. Locally, with no password set, the app opens directly.
_pw = os.environ.get("APP_PASSWORD")
if _pw and not st.session_state.get("unlocked"):
    st.markdown("### Foreman")
    st.caption("AI-Enabled Predictive Maintenance · ISE 298 · San José State University")
    entered = st.text_input("Access code", type="password")
    if entered == _pw:
        st.session_state["unlocked"] = True
        st.rerun()
    elif entered:
        st.error("That code is not right.")
    st.stop()

# ── styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 3.2rem; max-width: 1280px; }
  h1, h2, h3 { letter-spacing: -.01em; }
  .fm-title { font-size: 2.1rem; font-weight: 700; margin-bottom: .1rem; }
  .fm-sub { color: #6C7A85; font-size: .95rem; margin-bottom: 1.4rem; }
  .fm-card { border: 1px solid rgba(128,144,152,.28); border-radius: 5px;
             padding: 15px 17px; height: 100%; }
  .fm-lab { font-family: ui-monospace, monospace; font-size: .68rem;
            letter-spacing: .09em; text-transform: uppercase;
            color: #0E7C7B; margin-bottom: .55rem; }
  .fm-wo { font-family: ui-monospace, "IBM Plex Mono", monospace;
           font-size: .80rem; line-height: 1.6;
           border: 1px solid rgba(128,144,152,.28); border-radius: 5px;
           padding: 14px 17px; }
  .fm-wo-h { font-weight: 700; color: #0E7C7B; letter-spacing: .05em;
             margin-top: .15rem; }
  .fm-wo-gap { height: .7rem; }
  .fm-tier { display: inline-block; padding: 5px 13px; border-radius: 3px;
             font-weight: 700; font-size: .92rem; }
  .fm-bar-row { display: flex; align-items: center; gap: 9px;
                margin-bottom: 7px; font-size: .81rem; }
  .fm-bar-name { width: 172px; flex-shrink: 0; text-align: right;
                 color: #55666E; }
  .fm-bar-track { flex: 1; height: 17px; position: relative;
                  background: rgba(128,144,152,.12); border-radius: 2px; }
  .fm-bar-fill { position: absolute; top: 0; height: 100%; border-radius: 2px; }
  .fm-bar-val { width: 118px; flex-shrink: 0; white-space: nowrap;
                font-family: ui-monospace, monospace; font-size: .74rem;
                color: #6C7A85; }
  .fm-flag { font-size: .82rem; padding: 7px 11px; border-radius: 3px;
             margin-bottom: 6px; background: rgba(194,87,24,.10);
             border-left: 3px solid #C2571A; }
  .fm-src { font-size: .74rem; color: #6C7A85; margin-top: .5rem; }
  .fm-rc { font-size: .84rem; padding: 8px 11px; border-radius: 3px;
           margin-bottom: 6px; border-left: 3px solid #2C6FAF;
           background: rgba(44,111,175,.08); }
  .fm-rc-k { font-family: ui-monospace, monospace; font-size: .66rem;
             letter-spacing: .08em; text-transform: uppercase; opacity: .75; }
  div[data-testid="stMetricValue"] { font-size: 1.9rem; }
</style>
""", unsafe_allow_html=True)


# ── example cases: typical rows from the real dataset ────────────────────
CASES = example_cases()


@st.cache_data
def tool_life_model():
    """Fit the tool life model once: 120 tool lifetimes found in the data."""
    df = pd.read_csv("data/ai4i2020.csv")
    tools = rel.tool_lives(df)
    beta, eta = rel.weibull_fit(tools["life"], tools["failed"])
    km_t, km_s = rel.kaplan_meier(tools["life"], tools["failed"])
    return {"df": df, "tools": tools, "beta": beta, "eta": eta,
            "km_t": km_t, "km_s": km_s, "rates": rel.mode_rates(df),
            "co": rel.co_occurrence(df), "by_age": rel.failure_prob_by_wear(df, 20)}


LIFE = tool_life_model()

SLIDERS = {   # session key -> dataset column
    "q": "Type_encoded", "air": "Air temperature [K]",
    "proc": "Process temperature [K]", "rpm": "Rotational speed [rpm]",
    "tq": "Torque [Nm]", "wear": "Tool wear [min]",
}
CAST = {"q": int, "rpm": int, "wear": int}


def load_case():
    """Copy the chosen example into the sliders. Runs when the dropdown
    changes or Reset is pressed, so edits never leak between examples."""
    c = CASES[st.session_state["case"]]
    for key, col in SLIDERS.items():
        st.session_state[key] = CAST.get(key, float)(c[col])


if "case" not in st.session_state:
    st.session_state["case"] = "Power failure"
    load_case()

# ── header ────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="fm-title">Foreman</div>', unsafe_allow_html=True)
    st.markdown('<div class="fm-sub">Predicts machine failure, says which '
                'component and why, and writes the work order.</div>',
                unsafe_allow_html=True)
with right:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    st.caption("Assistant: **Claude**" if has_key
               else "Assistant: **built-in writer**  \n`ANTHROPIC_API_KEY` not set")

tab_one, tab_batch, tab_rel, tab_dem, tab_line, tab_about = st.tabs(
    ["Single cycle", "Score a shift", "Reliability", "Demand & spares",
     "Production line", "How it works"])


# ══════════════════════════════════════════════════════════════════════════
# SINGLE CYCLE
# ══════════════════════════════════════════════════════════════════════════
with tab_one:
    with st.sidebar:
        st.markdown("### Sensor readings")
        st.selectbox("Load an example", list(CASES.keys()), key="case",
                     on_change=load_case)
        c = CASES[st.session_state["case"]]

        q = st.select_slider("Product quality", options=[0, 1, 2],
                             format_func=lambda v: QUALITY[v], key="q")
        air = st.slider("Air temperature (K)", 295.0, 305.0, step=.1, key="air")
        proc = st.slider("Process temperature (K)", 305.0, 314.0, step=.1,
                         key="proc")
        rpm = st.slider("Rotational speed (rpm)", 1160, 2890, step=1, key="rpm")
        tq = st.slider("Torque (Nm)", 3.0, 77.0, step=.1, key="tq")
        wear = st.slider("Tool wear (min)", 0, 253, step=1, key="wear")

        edited = any(abs(float(st.session_state[k]) - float(c[col])) > 1e-6
                     for k, col in SLIDERS.items())
        if edited:
            st.caption("Edited from the example.")
            st.button("Reset to example", on_click=load_case)

        st.divider()
        use_llm = st.toggle("Use Claude to write the order", value=has_key,
                            disabled=not has_key)
        st.caption("Move any slider and everything below re-scores.")

    row = {
        "Type_encoded": q, "Air temperature [K]": air,
        "Process temperature [K]": proc, "Rotational speed [rpm]": rpm,
        "Torque [Nm]": tq, "Tool wear [min]": wear,
    }
    p = predict(row)
    risk, tier = p["risk"], p["tier"]

    TONE = {"none": ("#0E7C7B", "rgba(14,124,123,.13)"),
            "low":  ("#B07A12", "rgba(176,122,18,.14)"),
            "high": ("#C2571A", "rgba(194,87,24,.14)")}
    fg, bg = TONE[tier["urgency"]]

    # ── top row: risk, tier, derived values ──────────────────────────────
    k1, k2, k3, k4 = st.columns([1.1, 1.5, 1.2, 1.2])
    k1.metric("Failure risk", f"{risk:.0%}")
    with k2:
        st.markdown(f'<div class="fm-lab">Action</div>'
                    f'<div class="fm-tier" style="color:{fg};background:{bg}">'
                    f'{tier["name"]}</div>', unsafe_allow_html=True)
    k3.metric("Mechanical power", fmt(p["readings"]["Power [W]"], "W"))
    k4.metric("Temp differential", fmt(p["readings"]["Temp_diff [K]"], "K"))

    st.write("")

    # ── where + why ──────────────────────────────────────────────────────
    col_where, col_why = st.columns([1.25, 1])

    with col_where:
        st.markdown('<div class="fm-lab">Where to look</div>',
                    unsafe_allow_html=True)
        alarm = load()["kb"]["policy"]["tiers"][1]["max_risk"]
        st.markdown(schematic.render(component_mode(p), risk, alarm),
                    unsafe_allow_html=True)

        st.markdown('<div class="fm-lab" style="margin-top:.7rem">'
                    'Failure mode probability</div>', unsafe_allow_html=True)
        kb_modes = load()["kb"]["modes"]
        bars = []
        for code, prob in sorted(p["modes"].items(), key=lambda t: -t[1]):
            w = max(prob * 100, 0.8)
            col = "#C2571A" if prob >= .5 else "#7E8E97"
            bars.append(
                f'<div class="fm-bar-row"><div class="fm-bar-name">'
                f'{kb_modes[code]["label"]}</div>'
                f'<div class="fm-bar-track"><div class="fm-bar-fill" '
                f'style="width:{w:.1f}%;background:{col}"></div></div>'
                f'<div class="fm-bar-val">{prob:.0%}</div></div>')
        st.markdown("".join(bars), unsafe_allow_html=True)

        ws = p["wear_state"]
        wear_txt = {"replace": ("#C2571A", "inside the 200–240 min replacement window"),
                    "approaching": ("#B07A12", "approaching the replacement window"),
                    "ok": ("#0E7C7B", "within normal range")}[ws]
        st.markdown(
            f'<div class="fm-bar-row"><div class="fm-bar-name">Tool wear clock</div>'
            f'<div class="fm-bar-track"><div class="fm-bar-fill" '
            f'style="width:{min(wear/253*100,100):.1f}%;background:{wear_txt[0]}">'
            f'</div></div><div class="fm-bar-val">{wear:.0f} min</div></div>'
            f'<div class="fm-src">Tool wear is read from the clock, not predicted — '
            f'the dataset fires this mode at a random point in the window. '
            f'Currently {wear_txt[1]}. Tool life model: '
            f'<b>{rel.conditional_survival(max(wear, 1), 10, LIFE["beta"], LIFE["eta"]):.0%}</b> '
            f'chance this tool lasts another 10 minutes.</div>',
            unsafe_allow_html=True)

    with col_why:
        st.markdown('<div class="fm-lab">Why — what drove this</div>',
                    unsafe_allow_html=True)
        fac = p["factors"]
        span = max(abs(f["contribution"]) for f in fac) or 1
        rows = []
        for f in fac:
            pct = abs(f["contribution"]) / span * 50
            pos = f["contribution"] > 0
            colr = "#C2571A" if pos else "#0E7C7B"
            style = (f"left:50%;width:{pct:.1f}%" if pos
                     else f"right:50%;width:{pct:.1f}%")
            val = fmt(f["value"], f["unit"])
            rows.append(
                f'<div class="fm-bar-row"><div class="fm-bar-name">{f["name"]}</div>'
                f'<div class="fm-bar-track">'
                f'<div style="position:absolute;left:50%;top:0;bottom:0;'
                f'width:1px;background:rgba(128,144,152,.45)"></div>'
                f'<div class="fm-bar-fill" style="{style};background:{colr}"></div>'
                f'</div><div class="fm-bar-val">{val}</div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.markdown(f'<div class="fm-src">Left of centre pushes toward healthy, '
                    f'right pushes toward failure. Method: {fac[0]["method"]}.</div>',
                    unsafe_allow_html=True)

        if p["envelope_flags"]:
            st.markdown('<div class="fm-lab" style="margin-top:.9rem">'
                        'Outside documented limits</div>', unsafe_allow_html=True)
            for f in p["envelope_flags"]:
                st.markdown(f'<div class="fm-flag">{html.escape(f["text"])}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="fm-lab" style="margin-top:.9rem">'
                        'Outside documented limits</div>', unsafe_allow_html=True)
            st.markdown('<div class="fm-src">No reading is outside a '
                        'documented operating limit.</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="fm-lab" style="margin-top:.9rem">Root cause</div>',
                    unsafe_allow_html=True)
        if p["root_causes"]:
            for c in p["root_causes"]:
                tone = "#2C6FAF" if c["kind"] == "Operating setting" else "#B07A12"
                st.markdown(
                    f'<div class="fm-rc" style="border-left-color:{tone}">'
                    f'<div class="fm-rc-k" style="color:{tone}">{c["kind"]} · {c["category"]}</div>'
                    f'{html.escape(c["text"])}.<br><b>{html.escape(c["adapt"])}</b></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="fm-src">None. Every reading is inside its '
                        'documented limits.</div>', unsafe_allow_html=True)

    st.write("")

    # ── work order + what-if ─────────────────────────────────────────────
    col_wo, col_fix = st.columns([1.35, 1])

    with col_wo:
        st.markdown('<div class="fm-lab">Work order</div>', unsafe_allow_html=True)
        with st.spinner("Writing…"):
            wo = write_work_order(p, use_llm=use_llm)
        st.markdown(to_html(wo["text"]), unsafe_allow_html=True)

        cited = ", ".join(e["code"] for e in wo["retrieved"]["entries"]) or "none"
        origin = (f'written by Claude ({wo["model"]})' if wo["source"] == "claude"
                  else "written by the built-in writer")
        st.markdown(f'<div class="fm-src">Grounded in knowledge base entries: '
                    f'<b>{cited}</b> · {origin}</div>', unsafe_allow_html=True)
        if wo.get("error"):
            st.caption(f"Claude call failed, fell back to the template — {wo['error']}")

        with st.expander("Show what the assistant was given"):
            st.code(wo["prompt"], language=None, wrap_lines=True)

    with col_fix:
        st.markdown('<div class="fm-lab">What would clear it</div>',
                    unsafe_allow_html=True)
        fixes = wo["fixes"]
        if risk < .15:
            st.markdown("Nothing to clear. Risk is already under 15%.")
        elif fixes:
            for f in fixes[:3]:
                st.markdown(
                    f'**{f["name"]}** {fmt(f["from"], f["unit"])} → '
                    f'**{fmt(f["to"], f["unit"])}**  \nrisk '
                    f'{f["risk_from"]:.0%} → **{f["risk_to"]:.0%}**')
        else:
            st.markdown("No single operating change brings risk under 15%. "
                        "Inspection is required.")

        st.markdown('<div class="fm-lab" style="margin-top:1rem">'
                    'Risk across the range</div>', unsafe_allow_html=True)
        sweep_on = st.selectbox(
            "Change one reading, keep the rest fixed",
            ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"],
            format_func=lambda f: READING_LABELS[f])
        lo, hi = {"Torque [Nm]": (3.8, 76.6),
                  "Rotational speed [rpm]": (1168, 2886),
                  "Tool wear [min]": (0, 253)}[sweep_on]
        grid = np.linspace(lo, hi, 60)
        curve = what_if(row, sweep_on, grid)
        unit = sweep_on.split("[")[1].rstrip("]")
        xlab = f"{READING_LABELS[sweep_on]} ({unit})"
        st.line_chart(
            pd.DataFrame({xlab: [v for v, _ in curve],
                          "Failure risk": [r for _, r in curve]}).set_index(xlab),
            height=200, x_label=xlab, y_label="Failure risk (0–1)")
        st.caption(f"Now: {fmt(row[sweep_on], unit)}, risk {risk:.0%}. "
                   f"The alarm threshold is {alarm:.0%}.")


# ══════════════════════════════════════════════════════════════════════════
# BATCH
# ══════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("Upload a CSV of cycles and get a worklist ranked by risk, the root "
                "cause behind each alarm, and a check that the data still looks like "
                "what the model learned from. Needs the six sensor columns; the AI4I "
                "file works as-is.")
    up = st.file_uploader("CSV", type="csv", label_visibility="collapsed")
    use_sample = st.toggle("Or use the AI4I 2020 dataset (10,000 cycles)", key="use_sample")

    if up or use_sample:
        bdf = pd.read_csv(up) if up else pd.read_csv("data/ai4i2020.csv")
        if "Type" in bdf.columns and "Type_encoded" not in bdf.columns:
            bdf["Type_encoded"] = bdf["Type"].map({"L": 0, "M": 1, "H": 2})

        need = plant.BASE
        missing = [c for c in need if c not in bdf.columns]

        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            bdf = bdf.dropna(subset=need).head(50000)
            sc = score_many(bdf)
            labels = load()["kb"]["modes"]
            res = pd.DataFrame({
                "Row": np.arange(1, len(sc) + 1),
                "Risk": sc["risk"],
                "Action": sc["tier"],
                "Likely mode": [labels[m]["label"] if isinstance(m, str) else "—"
                                for m in sc["top_mode"]],
                "Root cause": [", ".join(plant.SHORT[k] for k in rc.split(", "))
                               if rc else "—" for rc in sc["root_cause"]],
                "Torque": sc["Torque [Nm]"],
                "Speed": sc["Rotational speed [rpm]"],
                "Tool wear": sc["Tool wear [min]"],
            }).sort_values("Risk", ascending=False)

            a, b, c2, d2 = st.columns(4)
            a.metric("Cycles scored", f"{len(res):,}")
            b.metric("Need attention this shift",
                     int((res["Action"] == "Intervene this shift").sum()))
            c2.metric("Inspect at changeover",
                      int((res["Action"] == "Inspect at changeover").sum()))
            drift = plant.drift_report(load()["train"][plant.BASE], bdf[need].astype(float))
            verdict = plant.drift_verdict(drift)
            d2.metric("Data drift", verdict.split(":")[0])

            st.dataframe(
                res.head(500).style.format({"Risk": "{:.1%}", "Torque": "{:.1f}",
                                            "Speed": "{:.0f}", "Tool wear": "{:.0f}"}),
                width="stretch", height=380, hide_index=True)
            if len(res) > 500:
                st.caption(f"Showing the 500 highest-risk of {len(res):,} cycles.")

            r1, r2 = st.columns([1.1, 1])
            with r1:
                st.markdown('<div class="fm-lab">Root causes behind the alarms</div>',
                            unsafe_allow_html=True)
                alarm = load()["kb"]["policy"]["tiers"][1]["max_risk"]
                keys = [k for rc in sc.loc[sc["risk"] >= alarm, "root_cause"]
                        for k in (rc.split(", ") if rc else ["none"])]
                if keys:
                    cnt = pd.Series(keys).value_counts()
                    names = {**plant.SHORT, "none": "No limit broken yet"}
                    kinds = {k: plant.CAUSES[k][1] for k in plant.CAUSES}
                    kinds["none"] = "—"
                    t = pd.DataFrame({"Cause": [names[k] for k in cnt.index],
                                      "Type": [kinds[k] for k in cnt.index],
                                      "Alarms": cnt.values})
                    t["Share"] = t["Alarms"] / t["Alarms"].sum()
                    st.dataframe(t.style.format({"Share": "{:.0%}"}), width="stretch",
                                 hide_index=True)
                    setting = t.loc[t["Type"] == "Operating setting", "Alarms"].sum() / t["Alarms"].sum()
                    st.caption(f"{setting:.0%} of alarm causes are operating settings that "
                               "should have been changed, not worn parts.")
                else:
                    st.caption("No alarms in this data.")
            with r2:
                st.markdown('<div class="fm-lab">Does this data look like the training data?</div>',
                            unsafe_allow_html=True)
                dv = drift.copy()
                dv["reading"] = [READING_LABELS[c] for c in dv["reading"]]
                st.dataframe(dv.rename(columns={"reading": "Reading", "psi": "PSI", "state": "State"})
                             .style.format({"PSI": "{:.3f}"}), width="stretch", hide_index=True)
                st.caption(verdict + " PSI under 0.1 is stable, 0.1 to 0.25 a moderate "
                           "shift, above 0.25 a major one.")


# ══════════════════════════════════════════════════════════════════════════
# RELIABILITY
# ══════════════════════════════════════════════════════════════════════════
with tab_rel:
    B, E = LIFE["beta"], LIFE["eta"]
    tools = LIFE["tools"]
    oc = tools["outcome"].value_counts()
    st.markdown("The rows are in production order and tool wear resets at every "
                "tool change, so the data holds **120 real tool lifetimes**: "
                f"{oc.get('failed', 0)} ended in failure, {oc.get('planned change', 0)} "
                "were changed before failing, and 1 is still running.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weibull shape β", f"{B:.1f}", help="Above 1 means wear-out: risk rises with age")
    m2.metric("B10 life", f"{rel.b_life(0.10, B, E):.0f} min", help="10% of tools have failed by this age")
    m3.metric("Mean tool life", f"{rel.mean_life(B, E):.0f} min")
    cp = st.session_state.get("rel_cp", 1500)
    ar = rel.age_replacement(B, E, cp, 12000)
    m4.metric("Best age to change", f"{ar['T_opt']:.0f} min")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="fm-lab">Tool survival curve</div>', unsafe_allow_html=True)
        x = np.arange(150, 261, 1.0)
        km = np.interp(x, LIFE["km_t"], LIFE["km_s"], right=0)
        st.line_chart(pd.DataFrame({"Observed (Kaplan-Meier)": km,
                                    "Weibull fit": rel.R(x, B, E)}, index=x),
                      height=260, x_label="Tool age (wear minutes)",
                      y_label="Chance still working")
    with c2:
        st.markdown('<div class="fm-lab">Chance of failure per cycle by tool age</div>',
                    unsafe_allow_html=True)
        by = LIFE["by_age"]
        # Numeric index (band start) keeps the bars in age order; text labels
        # like "100-119" would be sorted alphabetically.
        st.bar_chart(pd.DataFrame({"Failure chance (%)": 100 * by["Machine failure"].values},
                                  index=pd.Index(by.index.astype(int), name="band")),
                     height=260, x_label="Tool age, start of 20-minute band (minutes)", y_label="%")

    st.markdown('<div class="fm-lab" style="margin-top:.6rem">When to change the tool</div>',
                unsafe_allow_html=True)
    a1, a2 = st.columns([1, 1.6])
    with a1:
        st.slider("Cost of a planned tool change ($)", 100, 5000, 1500, 100, key="rel_cp")
        st.caption("A failure is priced at \\$12,000, as in the cost model.")
        rtf = rel.age_replacement(B, E, cp, 12000, grid=np.array([400.0]))
        st.markdown(f"Change at **{ar['T_opt']:.0f} min**: only "
                    f"**{ar['p_fail_opt']:.1%}** of tools fail first, and cost per minute "
                    f"of tool use drops from **\\${rtf['cost_opt']:.2f}** (run to failure) "
                    f"to **\\${ar['cost_opt']:.2f}**.")
        age = st.slider("Check a tool at age (min)", 0, 250, 200, 5, key="rel_age")
        st.markdown(f"Chance it lasts another 10 min: "
                    f"**{rel.conditional_survival(max(age, 1), 10, B, E):.0%}**; "
                    f"another 20 min: **{rel.conditional_survival(max(age, 1), 20, B, E):.0%}**")
    with a2:
        st.line_chart(pd.DataFrame({"Cost per minute ($)": ar["cost_rate"]}, index=ar["grid"]),
                      height=250, x_label="Planned change age (minutes)", y_label="$ per minute")

    st.markdown('<div class="fm-lab" style="margin-top:.6rem">Reliability block diagram</div>',
                unsafe_allow_html=True)
    mission = st.slider("Mission length (cycles)", 1, 50, 10, key="rel_mission")
    bd = rel.block_diagram(LIFE["rates"], mission)
    blocks = []
    for bk in bd["blocks"]:
        weak = bk["component"] == bd["weakest"]
        blocks.append(
            f'<div style="flex:1;border:1.6px solid {"#C2571A" if weak else "#2C6FAF"};'
            f'background:{"rgba(194,87,24,.10)" if weak else "rgba(44,111,175,.07)"};'
            f'border-radius:5px;padding:10px 6px;text-align:center">'
            f'<div style="font-weight:700;font-size:.85rem">{bk["component"]}</div>'
            f'<div style="font-family:ui-monospace,monospace;font-size:.8rem;margin-top:6px">'
            f'R = {bk["R"]:.3f}</div><div style="font-size:.7rem;opacity:.7">{bk["mode"]}</div></div>')
    st.markdown('<div style="display:flex;align-items:center;gap:8px">'
                + '<div style="opacity:.6">→</div>'.join(blocks) + '</div>'
                + '<div style="height:.8rem"></div>', unsafe_allow_html=True)
    st.markdown(f"Series system: the machine runs only if every block works. "
                f"**R(system, {mission} cycles) = {bd['R_system']:.3f}**. Weakest block: "
                f"**{bd['weakest']}**.")

    st.markdown('<div class="fm-lab" style="margin-top:.6rem">How failures depend on each other</div>',
                unsafe_allow_html=True)
    d1, d2 = st.columns([1.2, 1])
    with d1:
        co = LIFE["co"].copy()
        co.index = [rel.MODE_LABEL[m] for m in co.index]
        co.columns = [m for m in co.columns]
        st.dataframe(co.style.format("{:.0%}"), width="stretch")
        st.caption("Row = failure that happened; column = chance the other failure "
                   "happened on the same cycle.")
    with d2:
        st.markdown("**Shared readings couple the parts**")
        for m in ["TWF", "OSF", "PWF", "HDF"]:
            st.markdown(f"- {rel.COMPONENT[m]}: {', '.join(rel.DRIVERS[m])}")
        st.caption("Torque links the spindle and the drive; tool wear links the "
                   "cutting tool and the spindle; speed links the drive and cooling.")


# ══════════════════════════════════════════════════════════════════════════
# DEMAND & SPARES
# ══════════════════════════════════════════════════════════════════════════
BEST_AGE = int(round(rel.age_replacement(LIFE["beta"], LIFE["eta"], 1500, 12000)["T_opt"]))
MIXES = {"As in the data (60% L, 30% M, 10% H)": None,
         "More L grade (80 / 15 / 5)": (80, 15, 5),
         "More H grade (30 / 40 / 30)": (30, 40, 30)}


@st.cache_data
def run_scenario(load_pct, speed_pct, mix, ambient_k, gap_k, change_age, cycles):
    """Re-run the 10,000 real cycles as they would look under a changed way
    of working, then count failures, alarms, cost and drift."""
    base = plant.prepare(LIFE["df"])
    xs = plant.apply_scenario(base, load_pct, speed_pct, mix, ambient_k, gap_k, change_age)
    s = plant.scenario_summary(xs, score_many(xs)["risk"].to_numpy(),
                               LIFE["beta"], LIFE["eta"], cycles_per_year=cycles)
    drift = plant.drift_report(load()["train"][plant.BASE], xs)
    mean_step = base["Type_encoded"].map(plant.WEAR_STEP).mean()
    life = change_age or rel.mean_life(LIFE["beta"], LIFE["eta"])
    spares = plant.spare_parts(s["by_mode"], cycles / (life / mean_step))
    return s, drift, spares


with tab_dem:
    st.markdown("What happens when the way the machine is used changes? Each setting "
                "below re-runs the 10,000 real cycles as they would look under the change. "
                "Failures come from the dataset's documented failure rules; alarms come "
                "from the trained model, so you can see whether it still catches them.")
    s1, s2 = st.columns([1, 2.1])
    with s1:
        cycles = st.slider("Demand (cycles a year)", 5000, 20000, 10000, 1000, key="dm_cycles")
        load_pct = st.slider("Load (torque) change (%)", -20, 30, 0, 5, key="dm_load")
        speed_pct = st.slider("Speed change (%)", -15, 15, 0, 5, key="dm_speed")
        mix = MIXES[st.selectbox("Product mix", list(MIXES), key="dm_mix")]
        ambient = st.slider("Hotter shop: air temperature (+K)", 0.0, 5.0, 0.0, 0.5, key="dm_air")
        gap = st.slider("Cooling keeps up less: gap change (K)", -2.0, 0.0, 0.0, 0.5, key="dm_gap")
        change = st.toggle(f"Change tools at {BEST_AGE} min", key="dm_change")
    S0, _, sp0 = run_scenario(0, 0, None, 0.0, 0.0, None, cycles)
    S1, drift, sp1 = run_scenario(load_pct, speed_pct, mix, ambient, gap,
                                  BEST_AGE if change else None, cycles)
    with s2:
        m1, m2, m3, m4 = st.columns(4)
        df_ = S1["failures"] - S0["failures"]
        dc_ = S1["predictive_cost"] - S0["predictive_cost"]
        m1.metric("Failures a year", f"{S1['failures']:,.0f}",
                  f"{df_:+,.0f} vs today" if abs(df_) >= 1 else None, delta_color="inverse")
        m2.metric("Caught by the model", f"{S1['recall']:.0%}")
        m3.metric("Cost, run to failure", f"${S1['reactive_cost'] / 1e6:,.2f}M")
        m4.metric("Cost, predictive", f"${S1['predictive_cost'] / 1e6:,.2f}M",
                  f"{'+' if dc_ >= 0 else '-'}${abs(dc_) / 1e6:,.2f}M vs today"
                  if abs(dc_) >= 5000 else None, delta_color="inverse")
        bm = pd.DataFrame({"Today": S0["by_mode"], "This scenario": S1["by_mode"]})
        bm.index = [rel.MODE_LABEL[m] for m in bm.index]
        # Horizontal bars keep the mode names readable. In a horizontal
        # chart x_label names the category axis and y_label the value axis.
        st.bar_chart(bm, stack=False, horizontal=True, height=260, x_label="",
                     y_label="Expected failures a year", color=["#9AA7B0", "#C2571A"])
        verdict = plant.drift_verdict(drift)
        worst = drift.loc[drift["psi"].idxmax()]
        (st.error if worst["psi"] >= .25 else st.warning if worst["psi"] >= .1 else st.success)(
            f"**Drift check.** {verdict} Largest shift: {READING_LABELS[worst['reading']]} "
            f"(PSI {worst['psi']:.2f}).")
    st.caption("Costs: \\$12,000 per missed failure, \\$1,500 per planned fix, \\$400 per "
               "false alarm, as in the cost model. A cycle can fail in more than one mode.")

    st.markdown('<div class="fm-lab" style="margin-top:.8rem">Spare parts to keep in stock '
                '(Pareto / ABC)</div>', unsafe_allow_html=True)
    a_share = sp1.loc[sp1["class"] == "A", "share"].sum()
    st.markdown(f"**{(sp1['class'] == 'A').sum()} of {len(sp1)} parts** (class A) carry "
                f"**{a_share:.0%}** of the yearly spend on spares. These are the critical "
                "few to stock and track closely; class C can be ordered as needed.")
    show = sp1[["class", "part", "annual_qty", "unit_cost", "annual_value", "share",
                "lead_days", "reorder_point"]].rename(columns={
        "class": "Class", "part": "Part", "annual_qty": "Used a year", "unit_cost": "Unit cost",
        "annual_value": "Spend a year", "share": "Share", "lead_days": "Lead time (days)",
        "reorder_point": "Reorder at"})
    st.dataframe(show.style.format({"Used a year": "{:.1f}", "Unit cost": "${:,.0f}",
                                    "Spend a year": "${:,.0f}", "Share": "{:.0%}"}),
                 width="stretch", hide_index=True)
    st.caption("Use per year comes from the predicted failures by mode and the planned tool "
               "changes. Reorder at = stock level that covers lead-time demand 95% of the "
               "time. Part prices, lead times and parts per repair are assumptions.")


# ══════════════════════════════════════════════════════════════════════════
# PRODUCTION LINE
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def line_study():
    base = plant.prepare(LIFE["df"])
    plans, detail = plant.line_plans(base, LIFE["beta"], LIFE["eta"], reps=6)
    stations = plant.station_profiles(base, LIFE["beta"], LIFE["eta"])
    stations_tc = plant.station_profiles(base, LIFE["beta"], LIFE["eta"], change_age=BEST_AGE)
    return plans, detail["Run to failure (today)"]["stats"], stations, stations_tc


@st.cache_data
def line_custom(predictive, tool_change, buffer, standby, repair):
    plans, _, stations, stations_tc = line_study()
    return plant.simulate_line(stations_tc if tool_change else stations, buffer=buffer,
                               recall=0.912 if predictive else 0.0,
                               false_alarm=(73 / 2000) / plant.PARTS_PER_RUN if predictive else 0.0,
                               standby=standby, repair_unplanned=repair, reps=6)


with tab_line:
    plans, stats0, stations, stations_tc = line_study()
    st.markdown("A four-station line in series, with a buffer of 5 parts between stations. "
                "Each station runs one quarter of the real cycles, split by torque, so it "
                "inherits a measured failure rate. A failure stops the station for 2 hours; "
                "a failure the model catches early becomes a 30-minute planned stop.")
    t1, t2 = st.columns([1, 1.25])
    with t1:
        st.markdown('<div class="fm-lab">Stations</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Station": s_["name"], "Torque": s_["torque_range"],
            "Cycle (min)": s_["cycle_min"], "Failure chance per run": s_["p_fail_cycle"]}
            for s_ in stations]).style.format({"Failure chance per run": "{:.2%}",
                                               "Cycle (min)": "{:.1f}"}),
            width="stretch", hide_index=True)
        neck = plans.loc[0, "bottleneck"]
        st.markdown(f"Bottleneck today: **{neck}**. It is busy or under repair more of the "
                    "time than any other station, so every station after it waits for parts.")
    with t2:
        st.markdown('<div class="fm-lab">Where time goes today</div>', unsafe_allow_html=True)
        ut = stats0.set_index("station")[["busy", "down", "starved", "blocked"]] * 100
        ut.columns = ["Working", "Down", "Starved", "Blocked"]
        st.bar_chart(ut, horizontal=True, height=250, x_label="", y_label="Share of time (%)",
                     color=["#0E7C7B", "#C2571A", "#C9D2D8", "#D99A2B"])

    st.markdown('<div class="fm-lab" style="margin-top:.6rem">Backup plans, each added on '
                'top of the previous one</div>', unsafe_allow_html=True)
    pv = plans.rename(columns={"plan": "Plan", "parts_per_hour": "Parts per hour",
                               "gain_vs_today": "vs today", "bottleneck": "Bottleneck"})
    st.dataframe(pv[["Plan", "Parts per hour", "vs today", "Bottleneck"]]
                 .style.format({"Parts per hour": "{:.1f}", "vs today": "{:+.0%}"}),
                 width="stretch", hide_index=True)
    st.caption(f"A standby machine at the bottleneck gives the largest gain, and the "
               f"bottleneck then moves to {plans['bottleneck'].iloc[-1]}: that is where the "
               "next backup plan should go. Starved = waiting for parts from upstream; "
               "blocked = the buffer downstream is full. Each recorded cycle is treated as "
               "a run of 25 parts; the layout, cycle times and repair times are assumptions.")

    st.markdown('<div class="fm-lab" style="margin-top:.6rem">Try your own plan</div>',
                unsafe_allow_html=True)
    with st.form("line_form"):
        f1, f2, f3 = st.columns(3)
        pm = f1.toggle("Predictive maintenance", value=True)
        tc = f1.toggle(f"Change tools at {BEST_AGE} min", value=False)
        buf = f2.slider("Buffer between stations (parts)", 1, 30, 5)
        rep = f2.slider("Repair time after a failure (min)", 60, 240, 120, 30)
        sb = f3.selectbox("Standby machine at", ["None"] + [s_["name"] for s_ in stations])
        go = f3.form_submit_button("Run the line")
    r = line_custom(pm, tc, buf, None if sb == "None" else sb, rep)
    today = plans.loc[0, "parts_per_hour"]
    g1, g2, g3 = st.columns(3)
    g1.metric("Line output", f"{r['throughput']:.1f} parts/h",
              f"{r['throughput'] / today - 1:+.0%} vs today")
    g2.metric("Bottleneck", r["bottleneck"])
    g3.metric("Bottleneck down for repair",
              f"{r['stats'].set_index('station').loc[r['bottleneck'], 'down']:.0%} of the time")


# ══════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════
with tab_about:
    A = load()
    perf = pd.read_csv("models/mode_performance.csv")

    st.markdown("""
#### The five layers

1. **Input** — the readings on the left, a CSV, or an example case.
2. **Predict** — gradient boosting gives overall risk; four separate
   classifiers give the mode.
3. **Explain** — feature attribution says which readings drove it; the
   mode maps to a physical component.
4. **Ground** — the matching maintenance procedures are retrieved from the
   knowledge base.
5. **Generate** — the assistant writes the work order from *only* those
   passages, and the cost model sets the urgency tier.

#### Why the assistant can be trusted

It receives the prediction, the factors and the retrieved passages, and is
instructed to use nothing else. It cannot state a threshold that does not
appear in the knowledge base. The work order names the entries it drew on,
and the full prompt is visible under the work order on the first tab.

#### From one machine to the plant

- **Root cause.** Each alarm is traced to the documented rule it breaks and
  sorted into an operating setting that should have adapted (speed, load)
  or a component that has worn (tool, spindle). The work order says which.
- **Demand & spares.** Changes in load, speed, product mix, shop
  temperature or volume re-run the real cycles, show the failures and cost
  that follow, and warn when the data has drifted far enough that the
  model should be retrained. Predicted failures set the yearly demand for
  each spare part, ranked by ABC class.
- **Production line.** A simulated four-station line finds the bottleneck
  and compares backup plans: predictive maintenance, planned tool changes,
  larger buffers and a standby machine.
""")

    st.markdown("#### Failure mode models")
    st.dataframe(
        perf[["code", "label", "positives", "recall", "precision", "pr_auc"]]
        .style.format({"recall": "{:.3f}", "precision": "{:.3f}",
                       "pr_auc": "{:.3f}"}),
        width="stretch", hide_index=True)

    st.markdown("""
The three deterministic modes are recovered almost exactly, because the
engineered features encode the quantities their rules are written in — a
temperature difference, a power, a wear-torque product.

**Tool wear failure scores poorly on purpose.** The dataset fires it at a
*randomly chosen* wear time between 200 and 240 minutes, so two identical
rows can disagree. Foreman therefore reports the wear clock directly
instead of a model probability for that mode. Random failures (RNF) are
excluded entirely for the same reason — they are noise by construction,
and they put a real ceiling on achievable recall.
""")

    st.markdown("#### Knowledge base")
    st.caption(A["kb"]["_about"]["authorship"])
    for code, e in A["kb"]["modes"].items():
        with st.expander(f"{code} — {e['label']}"):
            st.markdown(f"**Component:** {e['component']}")
            st.markdown(f"**Documented trigger:** {e['trigger_rule']}")
            st.caption(f"Source: {e['rule_source']}")
            st.markdown("**Inspection:**")
            for i, s in enumerate(e["inspection"], 1):
                st.markdown(f"{i}. {s}")
            st.markdown(f"**Safety:** {e['safety']}")
            st.caption("Procedure sources: " + "; ".join(e["sources"]))
