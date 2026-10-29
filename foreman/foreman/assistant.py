"""
assistant.py — the generative layer
===================================
Turns the evidence into a work order a technician can act on.

THE CONTRACT
  The model receives the prediction, the ranked factors, and the retrieved
  knowledge base passages. It is instructed to use nothing else. It may
  not state a threshold, a part name or a procedure that does not appear
  in the passages it was given. That is what makes the output citable
  rather than plausible-sounding.

WITHOUT AN API KEY
  Everything still works. The template writer below produces the same
  work order from the same inputs, deterministically. The demo never
  depends on a network call succeeding.
"""
from __future__ import annotations
import os
import re
import html
import textwrap
from functools import lru_cache
from .engine import QUALITY, suggest_fix, fmt
from .knowledge import retrieve, as_source_text

# Model id is configurable because published model names change over time.
DEFAULT_MODEL = os.environ.get("FOREMAN_MODEL", "claude-sonnet-5-5")

SYSTEM = """You write maintenance work orders for a machine shop.

You will be given: a failure risk score, a predicted failure mode, the \
sensor readings, which readings drove the prediction, and passages from \
the plant's maintenance knowledge base.

Rules you must follow:
- Use ONLY the supplied passages for procedures, part names, thresholds \
and safety notes. If something is not in the passages, do not say it.
- Never invent a numeric limit. Every threshold you quote must appear in \
the passages verbatim.
- Write for a technician on the floor, not an engineer at a desk. Short \
sentences. Plain words. No preamble, no restating the question.
- Keep the section headings you are given, in order.
- Under ROOT CAUSE, use the root cause given. Say plainly whether it is \
an operating setting that should have been changed or a part that has \
worn, and what should have adapted.
- If the evidence is thin or the readings are all within limits, say so \
plainly rather than manufacturing urgency.
- Write as a foreman talking to a technician. Never mention the \
passages, the knowledge base, the prompt, or what you were or weren't \
given. If nothing applies, just say what to do: keep running, re-score \
next cycle.
- When risk is low and nothing is flagged, keep the whole order under \
80 words.
- Plain text only. No markdown: no asterisks, no # signs, no bold, no \
tables. Put each heading in capitals on its own line and use "1." or "-" \
for list items. Keep units on every number (K, rpm, Nm, min, W).

Length: under 220 words."""


_fmt = fmt  # one number format across the app


def build_prompt(prediction: dict, retrieved: dict, fixes: list) -> str:
    r = prediction["readings"]
    lines = [
        "## Prediction",
        f"Overall failure risk: {prediction['risk']:.0%}",
        f"Urgency tier: {prediction['tier']['name']} — {prediction['tier']['action']}",
    ]
    if prediction["top_mode"]:
        e = next(x for x in retrieved["entries"] if x["code"] == prediction["top_mode"])
        lines.append(f"Predicted mode: {e['label']} "
                     f"({prediction['top_mode_p']:.0%}) — component: {e['component']}")
    else:
        lines.append("Predicted mode: none above 50% confidence")

    lines += ["", "## Readings"]
    for key, label, unit in [
            ("Air temperature [K]", "Air temperature", "K"),
            ("Process temperature [K]", "Process temperature", "K"),
            ("Rotational speed [rpm]", "Speed", "rpm"),
            ("Torque [Nm]", "Torque", "Nm"),
            ("Tool wear [min]", "Tool wear", "min"),
            ("Power [W]", "Mechanical power", "W"),
            ("Temp_diff [K]", "Temperature differential", "K"),
            ("Wear_x_Torque", "Wear x torque", "min·Nm")]:
        lines.append(f"  {label}: {_fmt(r[key], unit)}")
    lines.append(f"  Product quality: {QUALITY.get(int(r['Type_encoded']), '?')}")

    if prediction["envelope_flags"]:
        lines += ["", "## Outside documented limits"]
        for f in prediction["envelope_flags"]:
            lines.append(f"  {f['text']}")
    else:
        lines += ["", "## Outside documented limits", "  none"]

    lines += ["", "## Root cause"]
    if prediction.get("root_causes"):
        for c in prediction["root_causes"]:
            lines.append(f"  {c['kind']} ({c['category']}): {c['text']}")
            lines.append(f"    What should adapt: {c['adapt']}")
    else:
        lines.append("  none: every reading is inside its documented limits")

    lines += ["", "## What drove the prediction"]
    for f in prediction["factors"][:4]:
        lines.append(f"  {f['name']} = {_fmt(f['value'], f['unit'])} "
                     f"({f['direction']})")

    if fixes:
        lines += ["", "## Changes that would bring risk under 15%"]
        for fx in fixes[:3]:
            lines.append(f"  {fx['name']}: {_fmt(fx['from'], fx['unit'])} → "
                         f"{_fmt(fx['to'], fx['unit'])} "
                         f"(risk {fx['risk_from']:.0%} → {fx['risk_to']:.0%})")

    lines += ["", "## Knowledge base passages — your only source", "",
              as_source_text(retrieved)]

    lines += ["", "", "Write the work order with these headings, in order:",
              "STATUS / WHY / ROOT CAUSE / WHAT TO DO / WHAT WOULD CLEAR IT / SAFETY",
              "Plain text only, no markdown.",
              "Under WHAT TO DO, cite the knowledge base entry code in brackets."]
    return "\n".join(lines)


def _template_work_order(prediction: dict, retrieved: dict, fixes: list) -> str:
    """Deterministic fallback. Same inputs, same structure, no network."""
    r = prediction["readings"]
    entries = retrieved["entries"]
    primary = entries[0] if entries else None

    out = ["STATUS"]
    out.append(f"  Risk {prediction['risk']:.0%} — {prediction['tier']['name']}.")
    out.append(f"  {prediction['tier']['action']}")
    if primary and prediction["top_mode"]:
        out.append(f"  Most likely: {primary['label']}. Component: {primary['component']}.")
    elif primary:
        # Retrieved for a reason other than the model (e.g. the wear clock),
        # so don't present it as the predicted failure.
        out.append(f"  Watch the {primary['component'].lower()}: "
                   f"{primary['selected_because']}.")

    out += ["", "WHY"]
    if prediction["envelope_flags"]:
        for f in prediction["envelope_flags"]:
            out.append(f"  {f['text']}.")
    else:
        out.append("  No reading is outside a documented limit.")
    top = prediction["factors"][0]
    out.append(f"  Largest contributor: {top['name']} at "
               f"{_fmt(top['value'], top['unit'])} ({top['direction']}).")

    out += ["", "ROOT CAUSE"]
    if prediction.get("root_causes"):
        for c in prediction["root_causes"]:
            out.append(f"  {c['kind']}: {c['text']}.")
            out.append(f"  {c['adapt']}")
    else:
        out.append("  None found. Every reading is inside its documented limits.")

    out += ["", "WHAT TO DO"]
    if primary:
        for i, step in enumerate(primary["inspection"][:5], 1):
            out.append(f"  {i}. {step}")
        out.append(f"     [{primary['code']}: {primary['label']} procedure]")
    else:
        out.append("  1. No specific procedure applies. Continue monitoring.")

    out += ["", "WHAT WOULD CLEAR IT"]
    if prediction["risk"] < 0.15:
        out.append("  Nothing to clear. Risk is already under 15%.")
    elif fixes:
        for fx in fixes[:3]:
            out.append(f"  {fx['name']} {_fmt(fx['from'], fx['unit'])} → "
                       f"{_fmt(fx['to'], fx['unit'])}: risk falls to {fx['risk_to']:.0%}")
    else:
        out.append("  No single operating change brings risk under 15%. "
                   "Inspection is required.")

    out += ["", "SAFETY"]
    out.append("  " + (primary["safety"] if primary
                       else "Follow standard lockout procedure before any inspection."))
    return "\n".join(out)


def to_html(text: str) -> str:
    """Lay the work order out line by line as HTML.

    Streamlit runs st.markdown through its own renderer: newlines collapse,
    "1." becomes a list, ** becomes bold, and a <pre> block gets swapped
    for its code widget. Building every line as its own <div>, all on one
    physical line, sidesteps all of that and shows the order as written,
    whether the template or Claude wrote it.
    """
    out = []
    for ln in text.splitlines():
        s = ln.strip().replace("**", "")
        if not s:
            out.append('<div class="fm-wo-gap"></div>')
        elif re.fullmatch(r"#*\s*[A-Z][A-Z /&-]{2,40}:?", s):
            out.append(f'<div class="fm-wo-h">{html.escape(s.lstrip("# ").rstrip(":"))}</div>')
        else:
            indent = len(ln) - len(ln.lstrip())
            out.append(f'<div style="padding-left:{indent * .45:.2f}em">'
                       f'{html.escape(s)}</div>')
    return '<div class="fm-wo">' + "".join(out) + "</div>"


def _explain(exc: Exception) -> str:
    """The SDK's own message can be just 'Error code: 400'. Dig out the
    API's reason so the UI says what actually went wrong."""
    detail = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        detail = (body.get("error") or {}).get("message", "") or str(body)
    resp = getattr(exc, "response", None)
    if not detail and resp is not None:
        try:
            detail = resp.text[:300]
        except Exception:
            pass
    return f"{type(exc).__name__}: {detail or exc}"


@lru_cache(maxsize=256)
def _ask_claude(prompt: str, model: str) -> str:
    """One API call per distinct prompt. Streamlit re-runs the whole script
    on every click, so without this the same readings would be sent to
    Claude again and again. Failed calls raise and are not cached."""
    import anthropic
    # Account-linked keys (sk-ant-usr...) that aren't scoped to a workspace
    # must name one on every request, or the API answers 400.
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(
        default_headers={"anthropic-workspace-id": ws} if ws else None)
    msg = client.messages.create(
        model=model, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    # Newer models may return a thinking block before the answer, so take
    # only the text blocks rather than assuming the first block is text.
    text = "".join(getattr(b, "text", "") for b in msg.content
                   if getattr(b, "type", "") == "text").strip()
    if not text:
        raise RuntimeError(f"no text in reply (stop reason: {msg.stop_reason})")
    return text


def write_work_order(prediction: dict, use_llm: bool = True,
                     model: str | None = None) -> dict:
    """Produce the work order. Returns the text plus how it was produced,
    so the UI can be honest about which path ran."""
    retrieved = retrieve(prediction)
    fixes = suggest_fix(prediction["readings"])
    prompt = build_prompt(prediction, retrieved, fixes)

    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return {
                "text": _ask_claude(prompt, model or DEFAULT_MODEL),
                "source": "claude",
                "model": model or DEFAULT_MODEL,
                "retrieved": retrieved,
                "fixes": fixes,
                "prompt": prompt,
            }
        except Exception as exc:
            return {
                "text": _template_work_order(prediction, retrieved, fixes),
                "source": "template",
                "model": None,
                "error": _explain(exc),
                "retrieved": retrieved,
                "fixes": fixes,
                "prompt": prompt,
            }

    return {
        "text": _template_work_order(prediction, retrieved, fixes),
        "source": "template",
        "model": None,
        "retrieved": retrieved,
        "fixes": fixes,
        "prompt": prompt,
    }
