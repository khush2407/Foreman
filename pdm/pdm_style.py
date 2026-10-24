"""
pdm_style.py — shared plotting style and palette for the whole project.

Every script imports from here so all figures in the report look like one
family: same fonts, same grid weight, same colors, same export settings.

Palette was validated for colorblind separation (protan/deutan/tritan) and
for >=3:1 contrast against a light surface, so the figures survive both
grayscale printing and CVD readers.
"""
import matplotlib
matplotlib.use("Agg")                     # no display in a headless environment
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Categorical palette: fixed order, never cycled ────────────────────────
# Slot meanings are stable across every figure in the report.
BLUE   = "#2C6FAF"    # slot 1 — Logistic Regression / "normal" class
AMBER  = "#D97706"    # slot 2 — Random Forest / warning
TEAL   = "#0F9D8F"    # slot 3 — Gradient Boosting
PURPLE = "#8B5CF6"    # slot 4 — SVM
CATEGORICAL = [BLUE, AMBER, TEAL, PURPLE]

# Semantic colors — reserved, never reused as "series 5"
FAIL_RED   = "#C1362F"
OK_GREEN   = "#0F9D8F"

# Ink (text always wears text tokens, never the series color)
INK        = "#111827"
INK_MUTED  = "#6B7280"
GRID       = "#E3E7EC"
SURFACE    = "#FFFFFF"

# Sequential ramp for heatmaps: one hue, light -> dark
SEQ_CMAP = "Blues"


def apply_style():
    """Apply the project-wide matplotlib rcParams. Call once per script."""
    plt.rcParams.update({
        "figure.dpi":          110,
        "savefig.dpi":         200,
        "savefig.bbox":        "tight",
        "figure.facecolor":    SURFACE,
        "axes.facecolor":      SURFACE,

        "font.family":         "DejaVu Sans",
        "font.size":           10,
        "axes.titlesize":      12,
        "axes.titleweight":    "bold",
        "axes.titlepad":       12,
        "axes.labelsize":      10,
        "axes.labelcolor":     INK,
        "text.color":          INK,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "xtick.color":         INK_MUTED,
        "ytick.color":         INK_MUTED,

        # Recessive axes and grid — the data should be the loudest thing
        "axes.edgecolor":      GRID,
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "axes.axisbelow":      True,
        "grid.color":          GRID,
        "grid.linewidth":      0.7,

        "legend.frameon":      False,
        "legend.fontsize":     9,

        "lines.linewidth":     2.0,        # 2px lines per mark spec
        "lines.markersize":    6,
    })


def save(fig, path, verbose=True):
    """Save a figure and close it so long scripts don't leak memory."""
    fig.savefig(path)
    plt.close(fig)
    if verbose:
        print(f"    saved  {path}")


def pct_axis(ax, axis="y"):
    """Format an axis as percentages."""
    fmt = mticker.PercentFormatter(xmax=1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
