"""Shared matplotlib style for the IA paper figures (from old_codes/Plots13NRV-3).

Arial is used when a ttf is available (fonts/arial.ttf next to the repo or the
Feynman path from the notebook); otherwise falls back to the platform sans.

Sizes/dpi convention (Romain, 2026-07-14):
  - 1x1 figures: figsize (3.5, 3.5), dpi 150
  - 2-panel figures (1x2 / 2x1): figsize (7, 3.5), dpi 300
  - bigger grids: dpi 300
  - always save with bbox_inches="tight"
"""
import os

import matplotlib
import matplotlib.font_manager as fm

_ARIAL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "fonts", "arial.ttf"),
    "/feynman/work/dap/lceg/rp269101/miniforge3/envs/desienv/fonts/arial.ttf",
]


#: multiplies every font size set by apply_style().  1.0 = the paper figures.
#: Set the PAPER_STYLE_FONTSCALE env var to re-render a paper figure with
#: larger relative text -- e.g. for talks, where the figure is shown much
#: smaller than a journal column.  Scripts that hard-code a fontsize= should
#: multiply it by this value so the whole figure scales together.
FONTSCALE = float(os.environ.get("PAPER_STYLE_FONTSCALE", "1.0"))


def apply_style():
    family = "DejaVu Sans"
    for path in _ARIAL_CANDIDATES:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            family = "Arial"
            break
    matplotlib.rcParams.update({
        "figure.figsize": (3.5, 3.5),
        "figure.dpi": 150,
        "font.family": family,
        "font.size": 10 * FONTSCALE,
        "axes.labelsize": 10 * FONTSCALE,
        "axes.titlesize": 10 * FONTSCALE,
        "xtick.labelsize": 10 * FONTSCALE,
        "ytick.labelsize": 10 * FONTSCALE,
        "legend.fontsize": 10 * FONTSCALE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "text.usetex": False,
        "mathtext.fontset": "stix",
    })


def savefig(fig, out, dpi=None):
    """Save with the paper convention: dpi 150 for 1x1, 300 otherwise."""
    if dpi is None:
        w, h = fig.get_size_inches()
        dpi = 150 if max(w, h) <= 4.0 else 300
    os.makedirs(os.path.dirname(str(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"wrote {out}  (dpi={dpi}, bbox_inches=tight)")
