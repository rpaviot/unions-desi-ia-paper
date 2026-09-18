#!/usr/bin/env python3
"""Reproduce the colour-redshift figure (paper Fig. 2 / old_codes/BGS_RED.png).

Density contours of dereddened g-r vs redshift for the DESI samples used in the IA
analysis: BGS Red, BGS Blue (GMM colour split) and LRG. Reads the analysis-ready
parquet catalogues built by build_catalogues.py and uses the precomputed GR_DERED
colour column when present (falling back to FLUX/MW_TRANSMISSION otherwise).

Ports the recipe from old_codes/Plots13NRV-3.ipynb (cell 13): per sample, a smoothed
2D histogram over (z, g-r), with filled 1sigma/2sigma density levels and contour lines.

Usage
-----
    python scripts/plot_colour_redshift.py [--config CONFIG] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent.parent


def gr_dered(df: pd.DataFrame) -> np.ndarray:
    """Dereddened g-r, from the stored column if available else from fluxes."""
    if "GR_DERED" in df:
        return df["GR_DERED"].to_numpy()
    fg = df["FLUX_G"].to_numpy() / df["MW_TRANSMISSION_G"].to_numpy()
    fr = df["FLUX_R"].to_numpy() / df["MW_TRANSMISSION_R"].to_numpy()
    return -2.5 * np.log10(fg / fr)


def sigma_levels(density: np.ndarray, sigmas=(1, 2)) -> list[float]:
    """Density thresholds enclosing the given Gaussian sigma mass fractions."""
    flat = np.sort(density.ravel())[::-1]
    cumsum = np.cumsum(flat) / np.sum(flat)
    out = []
    for s in sigmas[::-1]:
        frac = 1 - np.exp(-0.5 * s ** 2)
        out.append(flat[np.searchsorted(cumsum, frac)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "data_sources.yaml"))
    ap.add_argument("--out", default=str(REPO / "results" / "figures" / "colour_redshift.png"))
    ap.add_argument("--subsample", type=int, default=10,
                    help="Use every Nth galaxy (matches the notebook's z[::10]).")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cat_dir = Path(cfg["dest"]) / cfg["catalogue_build"]["out_subdir"]

    datasets = [
        (cat_dir / "BGS_RED_GMM_clustering.dat.parquet", "BGS Red", "Oranges"),
        (cat_dir / "BGS_BLUE_GMM_clustering.dat.parquet", "BGS Blue", "Blues"),
        (cat_dir / "LRG_clustering.dat.parquet", "LRG", "Reds"),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    step = max(1, args.subsample)
    for path, label, cmap in datasets:
        if not path.exists():
            print(f"[warn] missing {path.name}; skipping {label}")
            continue
        df = pd.read_parquet(path, columns=None)
        z = df["Z"].to_numpy()[::step]
        col = gr_dered(df)[::step]
        m = np.isfinite(z) & np.isfinite(col)

        H, xe, ye = np.histogram2d(z[m], col[m], bins=[200, 200],
                                   range=[[0.1, 0.8], [0.0, 2.5]])
        H = gaussian_filter(H.T, sigma=2)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        lv = sigma_levels(H)
        color = plt.get_cmap(cmap)(0.7)
        ax.contourf(xc, yc, H, levels=[lv[0], lv[1], H.max()], cmap=cmap, alpha=0.3)
        ax.contour(xc, yc, H, levels=lv, colors=[color], linewidths=[0.8, 1.5])
        ax.plot([], [], color=color, label=label, lw=2)
        print(f"  {label}: {m.sum():,} galaxies plotted (from {path.name})")

    ax.set_xlim(0.1, 0.8)
    ax.set_ylim(0.0, 2.2)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel("$g - r$")
    ax.set_title("DESI Y1 Galaxy Samples")
    ax.legend(loc=4, frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nFigure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
