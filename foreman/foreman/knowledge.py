"""
knowledge.py — retrieval from the maintenance knowledge base
============================================================
The grounding layer. Given a prediction, it pulls the procedures that
apply and hands them to the assistant as source material.

This is deliberately simple retrieval — the knowledge base has four
entries, so a vector index would be theatre. What matters is the
contract: the assistant receives only these passages, and every claim it
makes has to come from them.
"""
from __future__ import annotations
from .engine import load


def retrieve(prediction: dict) -> dict:
    """Select the knowledge base entries relevant to this prediction.

    Selection rules, in order:
      1. The highest-probability predicted mode, if it clears 0.5.
      2. Tool wear, whenever the wear clock is in its replacement window —
         this is checked directly rather than predicted, because the
         dataset fires tool-wear failures at a random point in that window.
      3. Any mode whose documented operating limit this cycle has crossed,
         even if the classifier did not flag it.
    """
    kb = load()["kb"]
    picked, why = [], {}

    top = prediction.get("top_mode")
    if top:
        picked.append(top)
        why[top] = f"model predicts this mode at {prediction['modes'][top]:.0%}"

    if prediction["wear_state"] in ("replace", "approaching") and "TWF" not in picked:
        picked.append("TWF")
        why["TWF"] = (f"tool wear at {prediction['wear_minutes']:.0f} min — "
                      f"{'inside' if prediction['wear_state'] == 'replace' else 'approaching'} "
                      "the 200-240 min replacement window")

    flag_to_mode = {
        "Mechanical power":         "PWF",
        "Temperature differential": "HDF",
        "Wear x torque":            "OSF",
        "Tool wear":                "TWF",
    }
    for f in prediction["envelope_flags"]:
        code = flag_to_mode.get(f["what"])
        if code and code not in picked:
            picked.append(code)
            why[code] = f["text"]

    entries = []
    for code in picked:
        e = dict(kb["modes"][code])
        e["code"] = code
        e["selected_because"] = why[code]
        entries.append(e)

    return {
        "entries": entries,
        "envelope": kb["operating_envelope"],
        "policy_note": kb["policy"]["_source"],
        "kb_version": kb["_about"]["version"],
    }


def as_source_text(retrieved: dict) -> str:
    """Flatten the retrieved entries into the passage block the assistant
    is allowed to draw on. Anything not in this string is not available
    to the model."""
    if not retrieved["entries"]:
        return ("No failure-mode procedure applies to this cycle.\n"
                "General safety: Follow standard lockout procedure before "
                "any inspection.")

    parts = []
    for e in retrieved["entries"]:
        lines = [
            f"### {e['code']} — {e['label']}",
            f"Retrieved because: {e['selected_because']}",
            f"Component: {e['component']}",
            f"Summary: {e['plain_summary']}",
            f"Documented trigger: {e['trigger_rule']}  [{e['rule_source']}]",
        ]
        if e.get("predictability_note"):
            lines.append(f"Note: {e['predictability_note']}")
        lines.append("Inspection steps:")
        lines += [f"  {i}. {s}" for i, s in enumerate(e["inspection"], 1)]
        lines.append("Parts typically involved: " + ", ".join(e["parts"]))
        lines.append("Safety: " + e["safety"])
        lines.append("Sources: " + "; ".join(e["sources"]))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
