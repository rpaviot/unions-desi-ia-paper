#!/usr/bin/env python3
"""TATT (A_1, A_2) posteriors for the blue samples — nautilus contours.

Two figures:
  * <out>            : the four blue samples with the fiducial A_2 prior
  * <out>_priorcheck : A_2 prior [-3,3] vs [-5,5] vs [-6,6], one sample per
                       call of --check, to expose prior truncation.

Run: /home/rpaviot/unions_IA/.venv/bin/python plot_for_paper/plot_tatt_a1a2_gtc.py
"""
from pathlib import Path
import argparse
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

NARROW = "results/fits/nautilus_split_TATT_rmin30-6_bta1"
WIDE = "results/fits/nautilus_split_TATT_rmin30-6_bta1_a2wide"
# Fiducial since 2026-08-25: [-5,5] still truncates the high-M* blue bin (Fig. 5).
WIDER = "results/fits/nautilus_split_TATT_rmin30-6_bta1_a2wider"

SAMPLES = [
    ("BGS blue, global", "BGS_BLUE_GMM_zmin_0.10_zmax_0.50_multipoles_TATT_nautilus.npz"),
    (r"BGS blue, low $M_\star$", "BGS_BLUE_GMM_zmin_0.10_zmax_0.50_massbin0_multipoles_TATT_nautilus.npz"),
    (r"BGS blue, high $M_\star$", "BGS_BLUE_GMM_zmin_0.10_zmax_0.50_massbin1_multipoles_TATT_nautilus.npz"),
    ("ELG, global", "ELG_LOPnotqso_zmin_0.80_zmax_1.60_multipoles_TATT_nautilus.npz"),
]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", default=WIDER, help="fit directory to plot")
ap.add_argument("--figure-size", type=float, default=4.04,
                  help="pygtc figure size in inches (4.04 = the "
                       "paper column figure). Use a smaller value "
                       "together with PAPER_STYLE_FONTSCALE for "
                       "talks, so the labels survive the scaling.")
ap.add_argument("--out", default="plot_for_paper/figX_tatt_a1a2_gtc.png")
ap.add_argument("--drop", default="ELG, global",
                help="semicolon-separated sample labels to omit from the main panel; "
                     "the ELG posterior fills the A_2 prior and carries no "
                     "information, so it is excluded from the paper figure")
ap.add_argument("--check", default=None,
                help="sample key (global/low/high/elg): overlay narrow vs wide A2 prior")
args = ap.parse_args()


def load(path):
    d = np.load(path, allow_pickle=True)
    free = [str(x) for x in d["free_params"]]
    chain = np.asarray(d["chain"], float)
    logw = np.asarray(d["log_w"], float)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    cols = [free.index("a1"), free.index("a2")]
    c = chain[:, cols]
    mean = (c * w[:, None]).sum(0)
    cov = np.cov(c.T, aweights=w)
    sd = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (sd[0] * sd[1])
    return c, w, mean, sd, rho


if args.check:
    key = {"global": 0, "low": 1, "high": 2, "elg": 3}[args.check]
    label, fname = SAMPLES[key]
    chains, weights, labels = [], [], []
    for tag, d in (("prior $|A_2|<3$", NARROW), ("prior $|A_2|<5$", WIDE),
                   ("prior $|A_2|<6$", WIDER)):
        c, w, m, s, r = load(REPO / d / fname)
        chains.append(c); weights.append(w)
        labels.append(rf"{tag}: $A_2 = {m[1]:.2f} \pm {s[1]:.2f}$")
        print(f"{label:24s} {tag:16s} A1 = {m[0]:+.2f} +- {s[0]:.2f}   "
              f"A2 = {m[1]:+.2f} +- {s[1]:.2f}   rho = {r:+.3f}")
    # distinct filename: --check must never clobber the paper figure
    op = Path(args.out)
    out = REPO / op.with_name(f"{op.stem}_priorcheck_{args.check}{op.suffix}")
    title = label
else:
    chains, weights, labels = [], [], []
    dropped = {x.strip() for x in args.drop.split(";") if x.strip()}
    for label, fname in SAMPLES:
        if label in dropped:
            print(f"omitting {label} (unconstrained in A_2)")
            continue
        c, w, m, s, r = load(REPO / args.dir / fname)
        chains.append(c); weights.append(w)
        labels.append(rf"{label}  ($\rho = {r:.2f}$)")
        print(f"{label:24s} A1 = {m[0]:+.2f} +- {s[0]:.2f}   "
              f"A2 = {m[1]:+.2f} +- {s[1]:.2f}   rho = {r:+.3f}")
    out = REPO / args.out
    title = None

allc = np.vstack(chains)
allw = np.concatenate([w * len(w) for w in weights])
rng = []
for i in range(2):
    m = (allc[:, i] * allw).sum() / allw.sum()
    s = np.sqrt(((allc[:, i] - m) ** 2 * allw).sum() / allw.sum())
    rng.append((m - 3.5 * s, m + 3.5 * s))

GTC = pygtc.plotGTC(
    chains=chains,
    weights=weights,
    nBins=200,
    paramNames=[r"$A_1$", r"$A_2$"],
    paramRanges=rng,
    figureSize=args.figure_size,
    customLabelFont={"size": 9 * paper_style.FONTSCALE},
    customTickFont={"size": 6.5 * paper_style.FONTSCALE},
    customLegendFont={"size": 6.5 * paper_style.FONTSCALE},
    chainLabels=labels,
)
for ax in GTC.axes:
    labelled_y = any(t.get_text() for t in ax.get_yticklabels())
    ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
    if labelled_y:
        ax.yaxis.set_major_locator(MaxNLocator(4, prune="both"))
    else:
        ax.set_yticks([])
if title:
    GTC.suptitle(title, fontsize=9 * paper_style.FONTSCALE, y=1.0)
GTC.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"saved {out}")
