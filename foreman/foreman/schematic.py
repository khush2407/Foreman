"""
schematic.py — where on the machine
===================================
Draws a simple machine schematic and marks the component implicated by
the predicted failure mode.

This is the half of the explanation that a SHAP bar chart cannot give.
"Mechanical power is 9,456 W" tells an engineer something. "Go look at
the drive system" tells a technician what to do with their hands.

Component mapping comes from the dataset's documented failure physics:
  HDF  heat cannot leave    -> cooling path and motor housing
  PWF  power out of band    -> drive system
  OSF  wear x torque strain -> spindle and bearings
  TWF  tool at end of life  -> cutting tool
"""
from __future__ import annotations

# x, y, w, h, title, subtitle
ZONES = {
    "TOOL":    (18,  74, 104, 74, "Cutting tool",   "TWF"),
    "SPINDLE": (138, 60, 128, 102, "Spindle &\nbearings", "OSF"),
    "HOUSING": (282, 60, 130, 102, "Motor housing\n& cooling", "HDF"),
    "DRIVE":   (428, 74, 114, 74, "Drive system",   "PWF"),
}

MODE_TO_ZONE = {"TWF": "TOOL", "OSF": "SPINDLE", "HDF": "HOUSING", "PWF": "DRIVE"}


def render(active_mode: str | None = None, risk: float = 0.0,
           alarm: float = 0.42) -> str:
    """Return an SVG string with the implicated zone marked.

    Colours are passed as CSS custom properties so the drawing follows
    whatever theme the page is using. `alarm` is the cost-optimal
    threshold; above it the marked zone turns orange instead of teal.
    """
    active = MODE_TO_ZONE.get(active_mode or "")
    hot = risk >= alarm

    parts = [
        '<svg viewBox="0 0 560 230" width="100%" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Machine schematic with the at-risk component marked">',
        "<defs>",
        '<marker id="fl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
        'markerHeight="5" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10 z" fill="#8A9BA6"/></marker>',
        "</defs>",
        # bed
        '<rect x="10" y="176" width="540" height="16" rx="3" '
        'fill="#D9E1E4" stroke="#B9C6CC" stroke-width="1"/>',
        '<text x="280" y="204" text-anchor="middle" font-family="system-ui,sans-serif" '
        'font-size="10" fill="#8A9BA6">machine bed</text>',
        # drive shaft through the middle
        '<line x1="122" y1="111" x2="428" y2="111" stroke="#B9C6CC" '
        'stroke-width="3"/>',
    ]

    for key, (x, y, w, h, title, code) in ZONES.items():
        on = key == active
        fill = "#FDEDE1" if (on and hot) else ("#E2F1F0" if on else "#FFFFFF")
        stroke = "#C2571A" if (on and hot) else ("#0E7C7B" if on else "#C6D2D6")
        sw = 2.4 if on else 1.2
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

        lines = title.split("\n")
        ty = y + h / 2 - (len(lines) - 1) * 7 + 1
        for i, ln in enumerate(lines):
            parts.append(
                f'<text x="{x + w/2}" y="{ty + i*14}" text-anchor="middle" '
                f'font-family="system-ui,sans-serif" font-size="12" '
                f'font-weight="{600 if on else 500}" '
                f'fill="{"#7A3208" if (on and hot) else ("#0A5F5C" if on else "#55666E")}">'
                f'{ln}</text>'
            )
        parts.append(
            f'<text x="{x + w/2}" y="{y + h - 9}" text-anchor="middle" '
            f'font-family="ui-monospace,monospace" font-size="9.5" '
            f'fill="{stroke if on else "#9DAAB0"}">{code}</text>'
        )

        if on:
            cx, top = x + w / 2, y - 10
            parts.append(
                f'<path d="M{cx} {top-22} L{cx} {top-4}" stroke="{stroke}" '
                f'stroke-width="2" marker-end="url(#fl)"/>'
            )
            parts.append(
                f'<text x="{cx}" y="{top-28}" text-anchor="middle" '
                f'font-family="system-ui,sans-serif" font-size="11" '
                f'font-weight="600" fill="{stroke}">inspect here</text>'
            )

    if active is None:
        parts.append(
            '<text x="280" y="30" text-anchor="middle" '
            'font-family="system-ui,sans-serif" font-size="11.5" fill="#8A9BA6">'
            'no single component implicated</text>'
        )

    parts.append("</svg>")
    return "".join(parts)
