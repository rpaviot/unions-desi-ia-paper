#!/usr/bin/env python
"""Reproduce draft Fig. 6: intrinsic-alignment amplitude A_IA versus median
stellar mass for the *blue* samples, comparing the wedge-multipole and the
projected (wg+) analyses.

Filled symbols  -> BGS blue (GMM separation, 0.1 < z < 0.5)
Open symbols    -> ELG (0.8 < z < 1.6)
Circles         -> multipole analysis  (xi_g+,2)
Triangles       -> projected statistics (wg+)

Style convention (per the draft): the projected wg+ fit reports A1 with the
opposite sign to the multipole fit, so the projected a1 is negated before
plotting onto the common A_IA axis.

Reads the per-sample / per-mass-bin nautilus NLA fit npz files written by
``fit_ia.py``:
  - multipole : results/fits/nautilus_rmin25-100_zerocross/*_multipoles_NLA_nautilus.npz
  - projected : results/fits_projected_rmin6/*_projected_NLA_nautilus.npz

Usage:
    python scripts/plot_aia_mass_blue_compare.py [--out plots/aia_vs_mass_blue_compare.png]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style


def collect(fits_dir, pattern, negate_a1=False, exclude=None):
    """(logM*_median, A1, err) points from every fit npz matching pattern.

    ``exclude``: skip files whose basename contains this substring.
    """
    pts = []
    for f in sorted(glob.glob(os.path.join(fits_dir, pattern))):
        if exclude and exclude in os.path.basename(f):
            continue
        d = np.load(f, allow_pickle=True)
        names = list(d["param_names"].astype(str))
        i = names.index("a1")
        a1 = float(d["bestfit"][i])
        if negate_a1:
            a1 = -a1
        pts.append((float(d["prop_logmstar_median"]), a1, float(d["errors"][i])))
    pts.sort()
    return np.array(pts).reshape(-1, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--multipole-dir",
                   default="results/fits/nautilus_rmin25-100_zerocross")
    p.add_argument("--projected-dir", default="results/fits_projected_rmin6")
    p.add_argument("--out", default="plots/aia_vs_mass_blue_compare.png")
    p.add_argument("--sampler", default="nautilus",
                   help="fit-file suffix sampler (nautilus / minuit)")
    p.add_argument("--dx", type=float, default=0.015,
                   help="horizontal offset (dex) separating the two analyses")
    args = p.parse_args()

    # BGS blue mass bins only (the resolved A_IA(M*) relation); the whole-sample
    # point is the same galaxies and would double-count, so it is excluded.
    bgs_mp = collect(args.multipole_dir,
                     f"BGS_BLUE_GMM_*massbin*_multipoles_NLA_{args.sampler}.npz")
    bgs_pr = collect(args.projected_dir,
                     f"BGS_BLUE_GMM_*massbin*_projected_NLA_{args.sampler}.npz",
                     negate_a1=True)
    # BGS blue whole ("main") sample, un-split by mass.
    bgs_main_mp = collect(args.multipole_dir,
                          f"BGS_BLUE_GMM_zmin_*_multipoles_NLA_{args.sampler}.npz",
                          exclude="massbin")
    bgs_main_pr = collect(args.projected_dir,
                          f"BGS_BLUE_GMM_zmin_*_projected_NLA_{args.sampler}.npz",
                          negate_a1=True, exclude="massbin")
    elg_mp = collect(args.multipole_dir,
                     f"ELG_*_multipoles_NLA_{args.sampler}.npz")
    elg_pr = collect(args.projected_dir,
                     f"ELG_*_projected_NLA_{args.sampler}.npz", negate_a1=True)

    paper_style.apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    dx = args.dx

    # fill encodes sample, marker encodes analysis (per the draft caption).
    if len(bgs_mp):
        ax.errorbar(bgs_mp[:, 0] - dx, bgs_mp[:, 1], yerr=bgs_mp[:, 2], fmt="o",
                    color="royalblue", ms=6, capsize=2, lw=1.2,
                    label=r"BGS blue, multipole ($\tilde\xi_{g+,2}$)")
    if len(bgs_pr):
        ax.errorbar(bgs_pr[:, 0] + dx, bgs_pr[:, 1], yerr=bgs_pr[:, 2], fmt="^",
                    color="royalblue", ms=7, capsize=2, lw=1.2,
                    label=r"BGS blue, projected ($w_{g+}$)")
    if len(bgs_main_mp):
        ax.errorbar(bgs_main_mp[:, 0] - dx, bgs_main_mp[:, 1],
                    yerr=bgs_main_mp[:, 2], fmt="o", mfc="none",
                    mec="royalblue", ecolor="royalblue", ms=7, capsize=2,
                    lw=1.2, mew=1.4,
                    label=r"BGS blue (main), multipole ($\tilde\xi_{g+,2}$)")
    if len(bgs_main_pr):
        ax.errorbar(bgs_main_pr[:, 0] + dx, bgs_main_pr[:, 1],
                    yerr=bgs_main_pr[:, 2], fmt="^", mfc="none",
                    mec="royalblue", ecolor="royalblue", ms=8, capsize=2,
                    lw=1.2, mew=1.4,
                    label=r"BGS blue (main), projected ($w_{g+}$)")
    if len(elg_mp):
        ax.errorbar(elg_mp[:, 0] - dx, elg_mp[:, 1], yerr=elg_mp[:, 2], fmt="o",
                    mfc="none", mec="seagreen", ecolor="seagreen", ms=7,
                    capsize=2, lw=1.2, mew=1.4,
                    label=r"ELG, multipole ($\tilde\xi_{g+,2}$)")
    if len(elg_pr):
        ax.errorbar(elg_pr[:, 0] + dx, elg_pr[:, 1], yerr=elg_pr[:, 2], fmt="^",
                    mfc="none", mec="seagreen", ecolor="seagreen", ms=8,
                    capsize=2, lw=1.2, mew=1.4,
                    label=r"ELG, projected ($w_{g+}$)")

    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\log_{10}(M_\ast\,/\,h^{-2}M_\odot)$")
    ax.set_ylabel(r"$A_{\rm IA}$")
    ax.legend(frameon=False, loc="best", fontsize=6 * paper_style.FONTSCALE)
    fig.tight_layout()
    paper_style.savefig(fig, args.out, dpi=150)
    n = (len(bgs_mp) + len(bgs_pr) + len(bgs_main_mp) + len(bgs_main_pr)
         + len(elg_mp) + len(elg_pr))
    print(f"wrote {args.out}  ({n} points)")

    # Tension between the two methods, matched by median stellar mass.
    print("\nMultipole vs projected (A1 with wg+ sign flipped):")
    print(f"  {'sample':22s} {'logM*':>6s} {'multipole':>16s} "
          f"{'projected':>16s} {'Delta/sigma':>11s}")
    for name, mp, pr in [("BGS blue main", bgs_main_mp, bgs_main_pr),
                         ("BGS blue bins", bgs_mp, bgs_pr),
                         ("ELG", elg_mp, elg_pr)]:
        for (m1, a1, e1), (m2, a2, e2) in zip(mp, pr):
            sig = abs(a1 - a2) / np.hypot(e1, e2)
            print(f"  {name:22s} {m1:6.2f} {a1:7.3f} +/- {e1:5.3f}  "
                  f"{a2:7.3f} +/- {e2:5.3f} {sig:10.2f}")


if __name__ == "__main__":
    main()
