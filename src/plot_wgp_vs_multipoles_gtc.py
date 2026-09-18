#!/usr/bin/env python3
"""Paper figure: (b1, A1) posterior, wg+ vs multipoles — LRG first z-bin.

pygtc corner plot of the nautilus chains for LRG_zmin_0.40_zmax_0.75 from the
fiducial multipoles fit (nautilus_snr_rmin25-100_zerocross) and the projected
fit (fits_projected_rmin6, the same directory used by the paper correlation-
function figures). Following the wgp sign convention, the projected a1 chain
is flipped (A1 = -a1_wgp) so both posteriors live in the multipole convention.

Run:  /home/rpaviot/unions_IA/.venv/bin/python plot_for_paper/plot_wgp_vs_multipoles_gtc.py
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator
import pygtc

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402

paper_style.apply_style()

import argparse

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--mult", default="results/fits/nautilus_split_rmin30-10_zerocross"
                                   "/LRG_zmin_0.40_zmax_0.75_multipoles_NLA_nautilus.npz")
_ap.add_argument("--proj", default="results/fits/nautilus_split_projected_rmin6"
                                   "/LRG_zmin_0.40_zmax_0.75_projected_NLA_nautilus.npz")
_ap.add_argument("--figure-size", type=float, default=4.04,
                   help="pygtc figure size in inches (4.04 = the "
                        "paper column figure). Use a smaller value "
                        "together with PAPER_STYLE_FONTSCALE for "
                        "talks, so the labels survive the scaling.")
_ap.add_argument("--out", default="plot_for_paper/figA_wgp_vs_multipoles_gtc.png")
_ap.add_argument("--b1-range", type=float, nargs=2, default=None)
_ap.add_argument("--a1-range", type=float, nargs=2, default=None)
_args = _ap.parse_args()

MULT = REPO / _args.mult
PROJ = REPO / _args.proj


def load(path, flip_a1=False):
    d = np.load(path, allow_pickle=True)
    chain = np.asarray(d["chain"], float).copy()   # columns: b1, a1
    logw = np.asarray(d["log_w"], float)
    w = np.exp(logw - logw.max())
    if flip_a1:
        chain[:, 1] = -chain[:, 1]
    mean = (chain * w[:, None]).sum(0) / w.sum()
    std = np.sqrt(((chain - mean) ** 2 * w[:, None]).sum(0) / w.sum())
    return chain, w, mean, std


mult, w_m, mean_m, std_m = load(MULT)
proj, w_p, mean_p, std_p = load(PROJ, flip_a1=True)
for tag, m, s in (("multipoles", mean_m, std_m), ("wgp(flipped)", mean_p, std_p)):
    print(f"{tag:14s}  b1 = {m[0]:.3f} +- {s[0]:.3f}   A1 = {m[1]:.3f} +- {s[1]:.3f}")

# Ranges: default to +-4 sigma around the two posteriors so the panel always
# frames the chains actually plotted, instead of the old hard-coded window.
def _range(col, override):
    if override is not None:
        return tuple(override)
    lo = min(mean_m[col] - 4 * std_m[col], mean_p[col] - 4 * std_p[col])
    hi = max(mean_m[col] + 4 * std_m[col], mean_p[col] + 4 * std_p[col])
    return (lo, hi)


_b1_range = _range(0, _args.b1_range)
_a1_range = _range(1, _args.a1_range)

chainLabels = [
    rf"$\xi_{{g+}}$: $A_1 = {mean_m[1]:.2f} \pm {std_m[1]:.2f}$",
    rf"$w_{{g+}}$: $A_1 = {mean_p[1]:.2f} \pm {std_p[1]:.2f}$",
]
# pygtc's "AandA_column" preset builds a 1.71 in figure (it divides the A&A
# column width by an internal 150 ppi, not 72), so the saved PNG was rendered
# at ~2x when included at \columnwidth and every label came out ~2x too big --
# the legend ran across the top panel and the b1/A1 tick labels collided.
# Build at the true column width instead (4.04 in shrinks to 3.5 in once the
# tight bbox is applied, i.e. 1:1 at \columnwidth) and set the fonts in real
# points, so what is set here is what the reader sees.
GTC = pygtc.plotGTC(
    chains=[mult, proj],
    weights=[w_m, w_p],
    nBins=300,
    paramNames=[r"$b_1$", r"$A_1$"],
    paramRanges=[_b1_range, _a1_range],
    figureSize=_args.figure_size,
    customLabelFont={"size": 9 * paper_style.FONTSCALE},
    customTickFont={"size": 6.5 * paper_style.FONTSCALE},
    customLegendFont={"size": 7 * paper_style.FONTSCALE},
    chainLabels=chainLabels,
)
# Thin the ticks so the b1 and A1 axes stop overlapping at the panel join;
# keep the 1D density axes unlabelled the way pygtc leaves them.
for ax in GTC.axes:
    labelled_y = any(t.get_text() for t in ax.get_yticklabels())
    ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
    if labelled_y:
        ax.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
    else:
        ax.set_yticks([])
out = REPO / _args.out
GTC.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"saved {out}")
